//! `flacify` command line: what the app will do once it has a model and a device.
//!
//! `probe` is real today and is the one that matters first: it answers what Flacify would
//! do with a file on this machine, before anything is downloaded or enhanced.

mod wav;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use flacify_core::dsp::cutoff::analyse;
use flacify_core::{plan, Probe};

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
    /// Report the detected band limit and the playback verdict for a file.
    Probe {
        /// A WAV file at 48 kHz. Other formats arrive with the decoder.
        file: std::path::PathBuf,
        /// Measured sustained cost, in seconds of compute per second of audio.
        /// Defaults to a comfortable desktop GPU until the probe exists.
        #[arg(long, default_value_t = 0.3)]
        rtf: f64,
    },
    /// Decode, enhance and write the FLAC cache entry. Needs the model (E2).
    Enhance {
        file: std::path::PathBuf,
        #[arg(long)]
        out: Option<std::path::PathBuf>,
    },
    /// Play a file with the A/B toggle. Needs the model and the audio device (E3).
    Play { file: std::path::PathBuf },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();

    match Cli::parse().command {
        Command::Probe { file, rtf } => probe(&file, rtf),
        Command::Enhance { .. } => anyhow::bail!(
            "enhance needs the ONNX model from the export spike (E1) and the runtime (E2)"
        ),
        Command::Play { .. } => anyhow::bail!(
            "play needs the model (E2) and the audio device (E3); use `flacify probe` for now"
        ),
    }
}

fn probe(path: &std::path::Path, rtf: f64) -> Result<()> {
    let audio = wav::read(path).with_context(|| format!("reading {}", path.display()))?;
    if audio.sample_rate != flacify_core::SAMPLE_RATE {
        anyhow::bail!(
            "{} is {} Hz; resampling on load arrives with the decoder",
            path.display(),
            audio.sample_rate
        );
    }

    let left = &audio.channels[0];
    let analysis = analyse(left)?;
    let seconds = left.len() as f64 / f64::from(audio.sample_rate);
    let probe = Probe::from_rtf(rtf);
    let plan = plan(&probe, 5.12, Some(seconds));

    println!("file      {}", path.display());
    println!(
        "format    {} Hz, {} ch, {} frames ({seconds:.2} s)",
        audio.sample_rate,
        audio.channels.len(),
        left.len()
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
        (f64::from(flacify_core::SAMPLE_RATE) / 2.0 - f64::from(analysis.cutoff_hz)) / 1000.0
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
