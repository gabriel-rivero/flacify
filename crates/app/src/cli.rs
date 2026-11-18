//! The command line surface: one subcommand per thing the app does.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "flacify",
    version,
    about = "It's not a FLAC, but it's close enough."
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Subcommand)]
pub(crate) enum Command {
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
        /// The default is the number the app measured end to end on an M5 Max
        /// (0.75-0.79 while playing live, 2026-11-18), not the sidecar's 0.54.
        #[arg(long, default_value_t = 0.78)]
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
        /// Play through the live path: bounded memory, the preset's window, and the
        /// latency the buffer policy states. Needs a 48 kHz stereo output.
        #[arg(long)]
        live: bool,
        /// Model window in granules: 1 (0.64 s), 2, 4 or 8 (5.12 s). Only used with
        /// --live; the default is the trained window.
        #[arg(long, default_value = "8", value_name = "1|2|4|8")]
        window: String,
        /// Sustained compute ratio the live buffer is sized from, in seconds per second
        /// of audio. 0.78 is the end-to-end figure the live path measured on the M5
        /// (0.75-0.79 over ten minutes); the status line prints the measured one.
        #[arg(long, default_value_t = 0.78)]
        rtf: f64,
        /// Seconds a --smoke run listens before it reports.
        #[arg(long, default_value_t = 4.0)]
        watch: f64,
        /// Start at this many seconds instead of the beginning. The smoke report says how
        /// long the seek took and what it starved, which is the acceptance run's check.
        #[arg(long)]
        seek: Option<f64>,
    },
    /// List the capture and playback devices, with the rate and channels each defaults to.
    Devices,
    /// Render every audio file under a folder, in order, skipping what is already done.
    Batch {
        /// Folder to walk. Every file symphonia can probe is a candidate.
        dir: PathBuf,
        /// List what would render and why it is skipped, without touching the model.
        #[arg(long)]
        dry_run: bool,
    },
    /// Blind A/B trials between the original and the generated high band, with a report.
    Ab {
        file: PathBuf,
        /// Number of trials. Each one asks which of the two excerpts is the generated one.
        #[arg(long, default_value_t = 20)]
        trials: u32,
        /// Replay seed. The same seed asks the same order with the same excerpts.
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Where to write the report; defaults next to the file as `<name>.flacify-ab.md`.
        #[arg(long)]
        export: Option<PathBuf>,
    },
    /// Show the render cache's size, or evict entries until it fits the cap.
    Cache {
        /// Evict least-recently-used entries until the cache is under the cap.
        #[arg(long)]
        evict: bool,
        /// Cap in gigabytes. Defaults to the configured 20.
        #[arg(long)]
        cap_gb: Option<f64>,
    },
    /// Write a text file with versions, paths and install state for a bug report.
    Diagnose {
        /// Defaults to flacify-diagnose.txt in the current directory.
        #[arg(long)]
        out: Option<PathBuf>,
    },
}
