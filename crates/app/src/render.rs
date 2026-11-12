//! One track through the model, start to finish.
//!
//! This is the only place the pipeline meets the Python sidecar, and the only place
//! that knows about the model's window. Everything it produces is planar stereo at
//! 48 kHz, exactly as long as the input: the pipeline's job is that invariant, and the
//! `debug_assert` at the end is the cheap way to notice if it ever stops holding.

use std::path::Path;

use anyhow::{Context, Result};
use flacify_core::{Block, Pipeline, SAMPLE_RATE};
use flacify_infer::python::{PythonWorker, WorkerConfig};
use flacify_infer::Engine;

use crate::decode;

/// The model's trained window: 5.12 s of 48 kHz audio.
pub const WINDOW: usize = 245_760;
/// One second, the reference implementation's default overlap.
pub const OVERLAP: usize = 48_000;
/// Frames handed to the pipeline per push. Small enough to report progress, large
/// enough that the per-call overhead does not matter (the model dominates by 1000x).
const CHUNK: usize = 8 * (WINDOW - OVERLAP);

/// A decoded track: planar stereo at 48 kHz.
pub struct Track {
    pub planar: Vec<f32>,
    pub frames: usize,
}

/// Interleaved stereo, the layout playback wants.
pub fn interleave_stereo(planar: &[f32], frames: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames * 2);
    for frame in 0..frames.min(planar.len() / 2) {
        out.push(planar[frame]);
        out.push(planar[frames + frame]);
    }
    out
}

impl Track {
    pub fn channel(&self, index: usize) -> &[f32] {
        &self.planar[index * self.frames..(index + 1) * self.frames]
    }

    /// Interleaved stereo, the layout playback wants.
    pub fn interleaved(&self) -> Vec<f32> {
        interleave_stereo(&self.planar, self.frames)
    }
}

/// What the file was, before it became 48 kHz stereo.
pub struct Source {
    pub sample_rate: u32,
    pub channels: usize,
}

pub struct Loaded {
    pub track: Track,
    pub analysis: flacify_core::dsp::cutoff::Analysis,
    pub source: Source,
}

impl Loaded {
    /// "44100 Hz x 2 ch -> 48000 Hz x 2 ch", or just the one format when they match.
    pub fn format_label(&self) -> String {
        if self.source.sample_rate == SAMPLE_RATE && self.source.channels == 2 {
            format!("{} Hz, 2 ch", self.source.sample_rate)
        } else {
            format!(
                "{} Hz, {} ch -> {} Hz, 2 ch",
                self.source.sample_rate, self.source.channels, SAMPLE_RATE
            )
        }
    }
}

/// Decode any supported file to 48 kHz stereo, then report what the spectrum looks like.
pub fn load(path: &Path) -> Result<Loaded> {
    let decoded = decode::decode(path)?;
    let source = Source {
        sample_rate: decoded.sample_rate,
        channels: decoded.channels,
    };
    let planar = decode::to_stereo_48k(&decoded)?;
    let frames = planar.len() / 2;
    let track = Track { planar, frames };
    let analysis = flacify_core::dsp::cutoff::analyse(track.channel(0))
        .with_context(|| format!("analysing {}", path.display()))?;
    Ok(Loaded {
        track,
        analysis,
        source,
    })
}

/// A seed that survives re-renders, cache keys and bug reports (ADR-006).
pub fn seed_for(track: &Track) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"flacify-pipeline-1");
    for sample in &track.planar {
        hasher.update(&sample.to_bits().to_le_bytes());
    }
    u64::from_le_bytes(
        hasher.finalize().as_bytes()[..8]
            .try_into()
            .expect("8 bytes"),
    )
}

/// A rendered track plus what the limiter had to do to it.
pub struct Render {
    pub planar: Vec<f32>,
    pub limit: crate::limit::LimitReport,
}

/// Render a whole track. `progress` is called with the number of input frames covered
/// each time a chunk comes back from the model, so a UI can show real movement.
pub fn enhance(
    track: &Track,
    cutoff_hz: Option<u32>,
    worker: &WorkerConfig,
    seed: u64,
    progress: &mut dyn FnMut(usize),
) -> Result<Render> {
    let model = PythonWorker::spawn(worker).context("starting the model worker")?;
    let engine = Engine::new(model, 2, WINDOW, cutoff_hz)?;
    let mut pipeline = Pipeline::new(engine, 2, WINDOW, OVERLAP, seed)?;

    let mut left = Vec::with_capacity(track.frames);
    let mut right = Vec::with_capacity(track.frames);
    let mut emitted = 0usize;

    let mut start = 0;
    while start < track.frames {
        let end = (start + CHUNK).min(track.frames);
        let mut interleaved = Vec::with_capacity((end - start) * 2);
        interleaved.extend_from_slice(&track.channel(0)[start..end]);
        interleaved.extend_from_slice(&track.channel(1)[start..end]);
        let block = Block::from_planar(2, interleaved)?;
        for out in pipeline.push(&block)? {
            left.extend_from_slice(out.channel(0));
            right.extend_from_slice(out.channel(1));
            emitted += out.frames();
        }
        progress(end);
        start = end;
    }
    for out in pipeline.finish()? {
        left.extend_from_slice(out.channel(0));
        right.extend_from_slice(out.channel(1));
        emitted += out.frames();
    }

    let mut planar = left;
    planar.extend_from_slice(&right);
    if emitted != track.frames || planar.len() != track.planar.len() {
        anyhow::bail!("pipeline returned {emitted} frames for {} in", track.frames);
    }
    // The model overshoots; nothing downstream can recover a clipped sample.
    let limit = crate::limit::limit_in_place(&mut planar);
    Ok(Render { planar, limit })
}

/// `flacify enhance` in one call: decode, decide, render, write.
pub fn enhance_file(
    input: &Path,
    output: &Path,
    worker: &WorkerConfig,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<()> {
    let Loaded {
        track,
        analysis,
        source,
    } = load(input)?;
    eprintln!(
        "source {} Hz, {} ch -> {} Hz stereo ({} frames)",
        source.sample_rate, source.channels, SAMPLE_RATE, track.frames
    );
    if analysis.looks_full_band() {
        eprintln!(
            "already full-band ({:.4}% of the spectrum above 21 kHz); copying it through",
            analysis.high_band_ratio * 100.0
        );
        let planar: Vec<f32> = track
            .channel(0)
            .iter()
            .chain(track.channel(1))
            .copied()
            .collect();
        crate::wav::write(output, SAMPLE_RATE, 2, &planar)?;
        progress(track.frames, track.frames);
        return Ok(());
    }

    eprintln!(
        "cutoff {} Hz; the top {:.1} kHz will be inferred",
        analysis.cutoff_hz,
        (f64::from(SAMPLE_RATE) / 2.0 - f64::from(analysis.cutoff_hz)) / 1000.0
    );
    let frames = track.frames;
    let rendered = enhance(
        &track,
        Some(analysis.cutoff_hz),
        worker,
        seed_for(&track),
        &mut |done| progress(done, frames),
    )?;
    if rendered.limit.limited_samples > 0 {
        eprintln!(
            "peak {:.2} -> {:.2} ({:+.1} dB) across {} samples",
            rendered.limit.peak_before,
            rendered.limit.peak_after,
            rendered.limit.reduction_db(),
            rendered.limit.limited_samples
        );
    }
    crate::wav::write(output, SAMPLE_RATE, 2, &rendered.planar)?;
    Ok(())
}
