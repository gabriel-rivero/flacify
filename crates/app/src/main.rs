//! `flacify`: set up the model, render a file, or play one with an A/B switch.
//!
//! The app is the whole product minus the model: decode, analysis, the windowing
//! pipeline, the buffer policy and the A/B switch are Rust. The model runs in a Python
//! sidecar the app installs and owns (ADR-010), and [`session`] keeps that sidecar alive
//! for the life of the app so a second file does not load it again.

mod cache;
mod decode;
mod device;
mod diagnose;
mod first_run;
mod flac;
mod frontier;
mod limit;
mod loudness;
mod player;
mod render;
mod render_status;
mod segments;
mod session;
mod setup;
mod source;
mod spectrum;
mod splice;
mod ui;
mod wav;

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use flacify_core::{plan, Probe, SAMPLE_RATE};

use crate::session::{needs_setup, Playback, Session};

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
    Setup {
        /// Install the CUDA build of torch from PyTorch's index instead of PyPI's CPU
        /// wheel. Only meaningful on Windows and Linux with an NVIDIA GPU.
        #[arg(long)]
        cuda: bool,
    },
    /// Report the detected band limit and the playback verdict for a file.
    Probe {
        /// Any format symphonia can decode.
        file: PathBuf,
        /// Measured sustained cost, in seconds of compute per second of audio.
        /// The sidecar measured 0.54 on an M5 Max; this stays a flag, not a constant.
        #[arg(long, default_value_t = 0.3)]
        rtf: f64,
    },
    /// Render the generated high band and write a 32-bit float WAV, or a 24-bit FLAC
    /// when the output name ends in `.flac`.
    Enhance {
        file: PathBuf,
        /// Defaults to <name>.flacify.wav next to the input.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Keep the input's low band instead of the model's (experimental).
        #[arg(long)]
        splice: bool,
    },
    /// Play a file with an instant A/B switch between original and generated.
    Play {
        file: PathBuf,
        /// Keep the input's low band instead of the model's (experimental).
        #[arg(long)]
        splice: bool,
        /// Render, open the output device, play for a second, report and exit.
        /// This is how a headless machine (and CI) checks the audio path.
        #[arg(long)]
        smoke: bool,
    },
    /// Write a text file with versions, paths and install state for a bug report.
    Diagnose {
        /// Defaults to flacify-diagnose.txt in the current directory.
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();

    match Cli::parse().command {
        Command::Setup { cuda } => run_setup(cuda),
        Command::Probe { file, rtf } => probe(&file, rtf),
        Command::Enhance { file, out, splice } => enhance(&file, out, splice),
        Command::Play {
            file,
            smoke,
            splice,
        } => play(&file, smoke, splice),
        Command::Diagnose { out } => {
            let path = diagnose::run(out)?;
            println!("wrote {}", path.display());
            Ok(())
        }
    }
}

fn run_setup(cuda: bool) -> Result<()> {
    if cuda {
        if cfg!(target_os = "macos") {
            anyhow::bail!(
                "macOS gets Metal from PyPI's torch; the CUDA wheel is for Windows and Linux"
            );
        }
        if std::env::var_os("FLACIFY_TORCH_INDEX").is_none() {
            std::env::set_var("FLACIFY_TORCH_INDEX", setup::CUDA_TORCH_INDEX);
        }
    }
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

fn probe(path: &Path, rtf: f64) -> Result<()> {
    let loaded = render::load_in_memory(path)?;
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

fn enhance(file: &Path, out: Option<PathBuf>, splice: bool) -> Result<()> {
    let paths = setup::Paths::discover()?;
    if !paths.is_ready() {
        anyhow::bail!("the model is not installed yet. Run `flacify setup` once (about 6 GB).");
    }
    let session = Session::new(&paths, splice)?;
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
        session.worker(),
        render::Store {
            cache: session.cache(),
            model_id: session.model_id(),
        },
        splice,
        &AtomicBool::new(false),
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

fn play(file: &Path, smoke: bool, splice: bool) -> Result<()> {
    let paths = setup::Paths::discover()?;
    let session = Session::new(&paths, splice)?;
    let started = Instant::now();

    // A full-band file and a cache hit never need the model, so they play on a machine
    // without the 6 GB install. Only a file that has to render asks for setup.
    let mut playback = match session.open(file) {
        Ok(playback) => playback,
        Err(error) if needs_setup(&error) => {
            if smoke {
                anyhow::bail!("{error}. A headless run cannot install it.");
            }
            return ui::run(first_run::FirstRun::new(paths, file.to_path_buf(), splice))
                .map_err(|error| anyhow::anyhow!("{error}"));
        }
        Err(error) => return Err(error),
    };

    if smoke {
        let result = report(&playback, started);
        playback.cancel();
        result
    } else {
        ui::run(ui::PlayerApp::new(session, playback)).map_err(|error| anyhow::anyhow!("{error}"))
    }
}

/// Report what a headless run saw.
///
/// Playback starts once the frontier is two seconds ahead, so the first note is already
/// the generated signal and the frontier stays ahead from there (it runs at about 1.28x
/// realtime once the model is loaded). A patience budget catches the slow machine that
/// never gets there: it starts on the original and the automatic promotion does the rest.
fn report(playback: &Playback, started: Instant) -> Result<()> {
    let engine = &playback.engine;
    let info = &playback.info;
    let rate = engine.device_rate as usize;

    let prebuffer = 2 * rate;
    let patience = 30.0;
    while started.elapsed().as_secs_f64() < patience {
        let ready = match &playback.frontier {
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
        let ready = match &playback.frontier {
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
    if let Some(failure) = playback.status.as_ref().and_then(|status| status.failure()) {
        println!("render    failed: {failure}");
    }
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

/// Counts allocations on the calling thread. Exists so a test can pin the audio
/// callback's no-allocation guarantee instead of restating it in a comment.
#[cfg(test)]
mod alloc_counter {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        /// `Some(bytes)` while a measurement is armed on this thread.
        static ALLOCATED: Cell<Option<usize>> = const { Cell::new(None) };
    }

    pub struct Counting;

    // SAFETY: every method forwards to `System`, which is the allocator these tests would
    // otherwise use; the counter is bookkeeping that never touches the returned pointer.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            add(layout.size());
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            // A realloc moves the allocation, so count the new bytes rather than the
            // difference: only "did it allocate anything" is asked of this number.
            add(new_size);
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    fn add(bytes: usize) {
        // `try_with`: an allocation during thread teardown must not panic in the allocator.
        let _ = ALLOCATED.try_with(|count| {
            if let Some(total) = count.get() {
                count.set(Some(total + bytes));
            }
        });
    }

    /// The bytes `body` allocated on this thread, with its result.
    pub fn measure<T>(body: impl FnOnce() -> T) -> (T, usize) {
        ALLOCATED.with(|total| total.set(Some(0)));
        let value = body();
        let bytes = ALLOCATED.with(|total| total.replace(None)).unwrap_or(0);
        (value, bytes)
    }
}

#[cfg(test)]
#[global_allocator]
static ALLOCATOR: alloc_counter::Counting = alloc_counter::Counting;
