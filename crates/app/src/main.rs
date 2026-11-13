//! `flacify`: set up the model, render a file, or play one with an A/B switch.
//!
//! The app is the whole product minus the model: decode, analysis, the windowing
//! pipeline, the buffer policy and the A/B switch are Rust. The model runs in a Python
//! sidecar the app installs and owns (ADR-010), which is why `setup` and
//! `worker_config` exist at all.

mod cache;
mod decode;
mod device;
mod frontier;
mod limit;
mod loudness;
mod player;
mod render;
mod setup;
mod spectrum;
mod ui;
mod wav;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use flacify_core::{plan, Probe, SAMPLE_RATE};
use flacify_infer::python::WorkerConfig;

use crate::frontier::Frontier;

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
    let cache = cache::Cache::new(paths.cache_dir());
    let model_id = cache::model_id(&paths.models_dir());
    let out = out.unwrap_or_else(|| {
        let mut name = file.file_name().unwrap_or_default().to_os_string();
        name.push(".flacify.wav");
        file.with_file_name(name)
    });

    let started = Instant::now();
    let mut last_printed = 0u32;
    render::enhance_file(
        file,
        &out,
        &config,
        &cache,
        &model_id,
        &mut |done, total| {
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
        },
    )
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
    let cache = Arc::new(cache::Cache::new(paths.cache_dir()));
    let model_id = cache::model_id(&paths.models_dir());
    let started = Instant::now();

    // Decode and analyse first: this is the only work between pressing play and hearing
    // the file. Everything else happens behind the playhead.
    let render::Loaded {
        track, analysis, ..
    } = render::load(file)?;
    let original = track.interleaved();
    let tags = decode::tags(file);
    let info_base = (
        file_name(file),
        tags,
        analysis.cutoff_hz,
        track.frames as f64 / f64::from(SAMPLE_RATE),
    );

    if analysis.looks_full_band() {
        eprintln!(
            "already full-band ({:.4}% of the spectrum above 21 kHz); nothing to generate",
            analysis.high_band_ratio * 100.0
        );
        let engine = device::AudioEngine::start(&original, None)?;
        let info = ui::TrackInfo {
            name: info_base.0,
            title: info_base.1.title,
            artist: info_base.1.artist,
            cutoff_hz: info_base.2,
            audio_secs: info_base.3,
            device: "none".to_string(),
            from_cache: false,
            gain_db: 0.0,
        };
        finish(engine, None, info, smoke, started)
    } else {
        let key = cache::key(
            &render::sample_hash(&track),
            render::WINDOW,
            render::OVERLAP,
            &model_id,
        );
        match cache.load(&key)? {
            Some(cached) => {
                eprintln!(
                    "cache hit: {:.0} s of audio, nothing to render",
                    info_base.3
                );
                let enhanced = render::interleave_stereo(&cached.planar, track.frames);
                let engine = device::AudioEngine::start(&original, Some(&enhanced))?;
                let info = ui::TrackInfo {
                    name: info_base.0,
                    title: info_base.1.title,
                    artist: info_base.1.artist,
                    cutoff_hz: info_base.2,
                    audio_secs: info_base.3,
                    device: cached.meta.device,
                    from_cache: true,
                    gain_db: cached.meta.gain_db,
                };
                finish(engine, None, info, smoke, started)
            }
            None if !device::progressive_supported() => {
                eprintln!(
                    "this output device is not 48 kHz stereo, so the track is rendered \
                     before playback starts"
                );
                let started = Instant::now();
                let (rendered, from_cache) = render::enhance_or_cached(
                    &track,
                    Some(analysis.cutoff_hz),
                    &config,
                    &cache,
                    &model_id,
                    &mut |_| {},
                )?;
                let enhanced = render::interleave_stereo(&rendered.planar, track.frames);
                let engine = device::AudioEngine::start(&original, Some(&enhanced))?;
                let info = ui::TrackInfo {
                    name: info_base.0,
                    title: info_base.1.title,
                    artist: info_base.1.artist,
                    cutoff_hz: info_base.2,
                    audio_secs: info_base.3,
                    device: rendered.device,
                    from_cache,
                    gain_db: rendered.gain_db,
                };
                eprintln!(
                    "rendered {:.0} s of audio in {:.0} s",
                    info_base.3,
                    started.elapsed().as_secs_f64()
                );
                finish(engine, None, info, smoke, started)
            }
            None => {
                let frontier = Frontier::new(track.frames);
                let cutoff = analysis.cutoff_hz;
                let seed = render::seed_for(&track);
                {
                    let frontier = Arc::clone(&frontier);
                    let cache = Arc::clone(&cache);
                    let config = config.clone();
                    std::thread::spawn(move || {
                        let result = render::render_into_frontier(
                            &track,
                            Some(cutoff),
                            &config,
                            seed,
                            &frontier,
                            &mut |_| {},
                        );
                        match result {
                            Ok(render) => {
                                if let Err(error) = cache.store(&key, &render, cutoff) {
                                    tracing::warn!(%error, "could not cache the render");
                                }
                            }
                            Err(error) => tracing::error!(%error, "rendering failed"),
                        }
                    });
                }
                let engine =
                    device::AudioEngine::start_progressive(&original, Arc::clone(&frontier))?;
                let info = ui::TrackInfo {
                    name: info_base.0,
                    title: info_base.1.title,
                    artist: info_base.1.artist,
                    cutoff_hz: info_base.2,
                    audio_secs: info_base.3,
                    device: "rendering".to_string(),
                    from_cache: false,
                    gain_db: 0.0,
                };
                finish(engine, Some(frontier), info, smoke, started)
            }
        }
    }
}

/// Hand the engine to the window, or report what a headless run saw.
///
/// Playback starts once the frontier is two seconds ahead, so the first note is already
/// the generated signal and the frontier stays ahead from there (it runs at about 1.28x
/// realtime once the model is loaded). A patience budget catches the slow machine that
/// never gets there: it starts on the original and the automatic promotion does the rest.
fn finish(
    engine: device::AudioEngine,
    frontier: Option<Arc<Frontier>>,
    info: ui::TrackInfo,
    smoke: bool,
    started: Instant,
) -> Result<()> {
    let rate = engine.device_rate as usize;

    if !smoke {
        return ui::run(ui::PlayerApp::new(engine, frontier, info))
            .map_err(|error| anyhow::anyhow!("{error}"));
    }

    let prebuffer = 2 * rate;
    let patience = 30.0;
    while started.elapsed().as_secs_f64() < patience {
        let ready = match &frontier {
            Some(frontier) => frontier.ready_frames(),
            None => usize::MAX,
        };
        if ready >= prebuffer {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let first_audio = started.elapsed();
    engine.controls.set_playing(true);

    // Then watch for a few seconds: the toggle should become live, and promotion should
    // fire on its own.
    let mut live: Option<f64> = None;
    let mut promoted: Option<f64> = None;
    let mut peak_ready = 0usize;
    let watch = 4.0;
    let until = started.elapsed().as_secs_f64() + watch;
    while started.elapsed().as_secs_f64() < until {
        std::thread::sleep(std::time::Duration::from_millis(100));
        let ready = match &frontier {
            Some(frontier) => frontier.ready_frames(),
            None => (info.audio_secs * rate as f64) as usize,
        };
        peak_ready = peak_ready.max(ready);
        let position = engine.controls.position_frames();
        if live.is_none() && ready > position + rate {
            live = Some(started.elapsed().as_secs_f64());
        }
        if promoted.is_none()
            && engine
                .controls
                .promoted
                .load(std::sync::atomic::Ordering::Relaxed)
        {
            promoted = Some(started.elapsed().as_secs_f64());
        }
    }

    let frames = engine.controls.position_frames();
    println!(
        "first     {:.2} s to first audio (prebuffer included)",
        first_audio.as_secs_f64()
    );
    println!(
        "device    {} Hz x {} ch",
        engine.device_rate, engine.device_channels
    );
    println!(
        "played    {frames} frames ({:.2} s of audio) in {:.0} s of watching",
        frames as f64 / f64::from(engine.device_rate),
        watch
    );
    println!("model     {}", info.device);
    println!(
        "generated {:.1} s ready of {:.1} s",
        peak_ready as f64 / f64::from(engine.device_rate),
        info.audio_secs
    );
    match live {
        Some(at) => println!("live      toggle offered at {at:.1} s"),
        None => println!("live      frontier never got a second ahead"),
    }
    match promoted {
        Some(at) => println!("promoted  automatically at {at:.1} s"),
        None => println!("promoted  not yet"),
    }
    println!(
        "switch    {}",
        if !engine.controls.has_enhanced {
            "none (full-band source)"
        } else if engine.controls.is_enhanced() {
            "generated"
        } else {
            "original"
        }
    );
    if frames == 0 {
        anyhow::bail!("the device never pulled audio");
    }
    Ok(())
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}
