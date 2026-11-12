//! `flacify`: set up the model, render a file, or play one with an A/B switch.
//!
//! The app is the whole product minus the model: decode, analysis, the windowing
//! pipeline, the buffer policy and the A/B switch are Rust. The model runs in a Python
//! sidecar the app installs and owns (ADR-010), which is why `setup` and
//! `worker_config` exist at all.

mod decode;
mod device;
mod limit;
mod player;
mod render;
mod setup;
mod ui;
mod wav;

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use flacify_core::{plan, Probe, SAMPLE_RATE};
use flacify_infer::python::WorkerConfig;

#[derive(Parser)]
#[command(
    name = "flacify",
    version,
    about = "It's not a FLAC, but it's close enough."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Install the Python runtime, the wheels and the model weights (about 6 GB, once).
    Setup,
    /// Report the detected band limit and the playback verdict for a file.
    Probe {
        /// Any format symphonia can decode.
        file: PathBuf,
        /// Measured sustained cost, in seconds of compute per second of audio.
        /// The sidecar measured 0.54 on an M5 Max; this stays a flag, not a constant.
        #[arg(long, default_value_t = 0.3)]
        rtf: f64,
    },
    /// Render the generated high band and write a 32-bit float WAV.
    Enhance {
        file: PathBuf,
        /// Defaults to <name>.flacify.wav next to the input.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Play a file with an instant A/B switch between original and generated.
    Play {
        file: PathBuf,
        /// Render, open the output device, play for a second, report and exit.
        /// This is how a headless machine (and CI) checks the audio path.
        #[arg(long)]
        smoke: bool,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();

    match Cli::parse().command {
        Command::Setup => run_setup(),
        Command::Probe { file, rtf } => probe(&file, rtf),
        Command::Enhance { file, out } => enhance(&file, out),
        Command::Play { file, smoke } => play(&file, smoke),
    }
}

fn run_setup() -> Result<()> {
    let paths = setup::Paths::discover()?;
    let bytes: u64 = setup::plan().iter().map(|step| step.bytes).sum();
    println!("flacify installs into {}", paths.root().display());
    println!(
        "this is a one-time download of about {:.1} GB:",
        bytes as f64 / 1e9
    );
    for step in setup::plan() {
        println!(
            "  {:<10} {} ({:.0} MB)",
            step.id,
            step.label,
            step.bytes as f64 / 1e6
        );
    }
    let progress = setup::Progress::new(|step, fraction| {
        if fraction >= 1.0 {
            println!("  {step}: done");
        } else {
            println!("  {step}: ...");
        }
    });
    setup::ensure(&paths, progress)?;
    println!("ready; try `flacify play <file>`");
    Ok(())
}

fn worker_config(paths: &setup::Paths) -> Result<WorkerConfig> {
    if !paths.is_ready() {
        anyhow::bail!(
            "the model is not installed yet. Run `flacify setup` once (about 6 GB), \
             or set FLACIFY_HOME if it lives somewhere else."
        );
    }
    Ok(WorkerConfig {
        program: paths.venv_python(),
        worker_script: paths.worker_script(),
        model_dir: paths.models_dir(),
        flashsr_dir: Some(paths.flashsr_dir()),
        device: "auto".to_string(),
    })
}

fn probe(path: &Path, rtf: f64) -> Result<()> {
    let loaded = render::load(path)?;
    let (track, analysis) = (&loaded.track, &loaded.analysis);
    let seconds = track.frames as f64 / f64::from(SAMPLE_RATE);
    let plan = plan(&Probe::from_rtf(rtf), 5.12, Some(seconds));

    println!("file      {}", path.display());
    println!(
        "format    {}, {} frames ({seconds:.2} s)",
        loaded.format_label(),
        track.frames
    );
    println!(
        "cutoff    {} Hz  (energy above 21 kHz: {:.4}% of the spectrum)",
        analysis.cutoff_hz,
        analysis.high_band_ratio * 100.0
    );
    if analysis.looks_full_band() {
        println!("verdict   already full-band; there is no top octave to infer");
        return Ok(());
    }
    println!(
        "verdict   the top {:.1} kHz would be inferred, not recovered",
        (f64::from(SAMPLE_RATE) / 2.0 - f64::from(analysis.cutoff_hz)) / 1000.0
    );
    println!("verdict   {}", mode_label(&plan));

    if let Some(warning) = &plan.warning {
        println!("\n{}", warning.headline);
        println!("{}", warning.detail);
    }
    Ok(())
}

fn mode_label(plan: &flacify_core::Plan) -> String {
    match plan.mode {
        flacify_core::Mode::Realtime => format!(
            "realtime: {:.1} s of prebuffer, first audio after {:.1} s",
            plan.prebuffer_secs, plan.time_to_first_audio_secs
        ),
        flacify_core::Mode::Marginal => format!(
            "marginal: {:.1} s of prebuffer, first audio after {:.1} s",
            plan.prebuffer_secs, plan.time_to_first_audio_secs
        ),
        flacify_core::Mode::PreRender => format!(
            "pre-render: {:.1} s of audio rendered before first sound, {:.0} s of wait",
            plan.prebuffer_secs,
            plan.estimated_wait_secs.unwrap_or_default()
        ),
    }
}

fn enhance(file: &Path, out: Option<PathBuf>) -> Result<()> {
    let paths = setup::Paths::discover()?;
    let config = worker_config(&paths)?;
    let out = out.unwrap_or_else(|| {
        let mut name = file.file_name().unwrap_or_default().to_os_string();
        name.push(".flacify.wav");
        file.with_file_name(name)
    });

    let started = Instant::now();
    let mut last_printed = 0u32;
    render::enhance_file(file, &out, &config, &mut |done, total| {
        if total == 0 {
            return;
        }
        let percent = (done * 100 / total) as u32;
        if percent >= last_printed + 5 {
            last_printed = percent;
            eprint!(
                "\r  {percent:3}%  {:.0} s of audio",
                done as f64 / f64::from(SAMPLE_RATE)
            );
        }
    })
    .context("enhancing")?;
    eprintln!();
    println!(
        "wrote {} in {:.1} s",
        out.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn play(file: &Path, smoke: bool) -> Result<()> {
    let paths = setup::Paths::discover()?;
    let config = worker_config(&paths)?;

    if smoke {
        return smoke_play(file, &config);
    }

    let (sender, receiver) = mpsc::channel();
    let progress = Arc::new(ui::Progress::new(0));
    {
        let progress = Arc::clone(&progress);
        let file = file.to_path_buf();
        std::thread::spawn(move || {
            let result = prepare(&file, &config, &progress);
            let _ = sender.send(result);
        });
    }

    ui::run(ui::PlayerApp::new(receiver, progress))
        .map_err(|error| anyhow::anyhow!("{error}"))
        .context("running the window")?;
    Ok(())
}

/// Render, open the device, let it pull for a second, then report. No window, no ears
/// required: it proves the callback runs and the playhead moves.
fn smoke_play(file: &Path, config: &WorkerConfig) -> Result<()> {
    let progress = ui::Progress::new(0);
    let prepared = prepare(file, config, &progress).map_err(|message| anyhow::anyhow!(message))?;
    let engine = device::AudioEngine::start(&prepared.original, prepared.enhanced.as_deref())
        .context("opening the output device")?;
    engine.controls.set_playing(true);
    std::thread::sleep(std::time::Duration::from_millis(1200));

    let position = engine
        .controls
        .position
        .load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "device    {} Hz x {} ch",
        engine.device_rate, engine.device_channels
    );
    println!(
        "played    {position} frames in 1.2 s ({:.2} s of audio)",
        position as f64 / f64::from(engine.device_rate)
    );
    println!("model     {}", prepared.device);
    println!(
        "switch    {}",
        if prepared.enhanced.is_some() {
            "enabled"
        } else {
            "none (full-band source)"
        }
    );
    if position == 0 {
        anyhow::bail!("the device never pulled audio");
    }
    if !engine.controls.is_playing() {
        anyhow::bail!("playback stopped after {position} frames");
    }
    Ok(())
}

/// Decode, analyse, and render the whole track. Runs off the UI thread.
fn prepare(
    file: &Path,
    config: &WorkerConfig,
    progress: &ui::Progress,
) -> Result<ui::Prepared, String> {
    let fail = |error: anyhow::Error| format!("{error:#}");

    progress.set_status("decoding");
    let render::Loaded {
        track, analysis, ..
    } = render::load(file).map_err(fail)?;
    progress.set_total(track.frames);
    progress.set_done(track.frames);
    let audio_secs = track.frames as f64 / f64::from(SAMPLE_RATE);

    if analysis.looks_full_band() {
        return Ok(ui::Prepared {
            track_name: file_name(file),
            original: track.interleaved(),
            enhanced: None,
            cutoff_hz: analysis.cutoff_hz,
            render_secs: 0.0,
            audio_secs,
            peak_before: 0.0,
            limited_samples: 0,
            device: "none (full-band source)".to_string(),
        });
    }

    progress.set_status(format!(
        "enhancing the top {:.1} kHz with the model",
        (f64::from(SAMPLE_RATE) / 2.0 - f64::from(analysis.cutoff_hz)) / 1000.0
    ));
    progress.set_done(0);
    let started = Instant::now();
    let rendered = render::enhance(
        &track,
        Some(analysis.cutoff_hz),
        config,
        render::seed_for(&track),
        &mut |done| progress.set_done(done),
    )
    .map_err(fail)?;
    let render_secs = started.elapsed().as_secs_f64();

    Ok(ui::Prepared {
        track_name: file_name(file),
        original: track.interleaved(),
        enhanced: Some(render::interleave_stereo(&rendered.planar, track.frames)),
        peak_before: rendered.limit.peak_before,
        limited_samples: rendered.limit.limited_samples,
        device: rendered.device,
        cutoff_hz: analysis.cutoff_hz,
        render_secs,
        audio_secs,
    })
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}
