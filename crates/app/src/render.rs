//! One track through the model, start to finish.
//!
//! This is the only place the pipeline meets the Python sidecar, and the only place
//! that knows about the model's window. Everything it produces is planar stereo at
//! 48 kHz, exactly as long as the input: the pipeline's job is that invariant, and the
//! `debug_assert` at the end is the cheap way to notice if it ever stops holding.

use std::path::Path;

use anyhow::{Context, Result};
use flacify_core::{Block, Pipeline, Preprocessor, SAMPLE_RATE};
use flacify_infer::python::{PythonWorker, WorkerConfig};
use flacify_infer::Engine;

use crate::decode;

/// The model's trained window: 5.12 s of 48 kHz audio.
pub const WINDOW: usize = 245_760;
/// One second, the reference implementation's default overlap.
pub const OVERLAP: usize = 48_000;
/// Frames handed to the pipeline per push.
///
/// One `step` per push: that is the amount of input that produces exactly one emitted
/// window, so the frontier gains a second of audio every time the model finishes one.
/// Pushing the whole track at once (which is what this used to do) means the frontier
/// stays empty until the render is over, and progressive playback cannot work at all.
const CHUNK: usize = WINDOW - OVERLAP;

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

/// A hash of the decoded samples, for cache keys. The pipeline version, window, overlap
/// and model go into the key separately, so this is only about the audio.
pub fn sample_hash(track: &Track) -> String {
    let mut hasher = blake3::Hasher::new();
    for sample in &track.planar {
        hasher.update(&sample.to_bits().to_le_bytes());
    }
    hasher.finalize().to_hex()[..32].to_string()
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
    /// Which accelerator the sidecar chose: `cuda`, `mps` or `cpu`.
    pub device: String,
    /// Loudness gain applied to the generated side, in dB, and the two measurements it
    /// came from. Zero when they were already equal (or could not be measured).
    pub gain_db: f64,
    pub lufs_input: f64,
    pub lufs_output: f64,
}

/// Render a whole track. `progress` is called with the number of input frames covered
/// each time a chunk comes back from the model, so a UI can show real movement.
pub fn enhance(
    track: &Track,
    cutoff_hz: Option<u32>,
    worker: &WorkerConfig,
    seed: u64,
    splice_low_band: bool,
    progress: &mut dyn FnMut(usize),
) -> Result<Render> {
    let mut planar = Vec::with_capacity(track.frames * 2);
    let streamed = enhance_streaming(
        track,
        cutoff_hz,
        worker,
        seed,
        splice_low_band,
        progress,
        |chunk| {
            planar.extend_from_slice(chunk);
            Ok(())
        },
    )?;
    Ok(Render {
        planar,
        limit: streamed.limit,
        device: streamed.device,
        gain_db: streamed.gain_db,
        lufs_input: streamed.lufs_input,
        lufs_output: streamed.lufs_output,
    })
}

/// The pipeline, emitting audio as it arrives instead of at the end.
///
/// `on_audio` receives planar stereo (the limiter has already run, so what it gets is
/// final) and may be called many times. The limiter is a per-sample function, so
/// limiting each block gives exactly the same samples as limiting the whole track
/// afterwards — which is what lets the frontier publish before the render finishes.
/// What `enhance_streaming` produced besides the audio.
pub struct Streamed {
    pub limit: crate::limit::LimitReport,
    pub device: String,
    pub gain_db: f64,
    pub lufs_input: f64,
    pub lufs_output: f64,
}

/// The most a loudness match may move the generated side. Beyond this something is wrong
/// with the measurement and a big gain would sound worse than the step it fixes.
const MAX_GAIN_DB: f64 = 6.0;

pub fn enhance_streaming(
    track: &Track,
    cutoff_hz: Option<u32>,
    worker: &WorkerConfig,
    seed: u64,
    splice_low_band: bool,
    progress: &mut dyn FnMut(usize),
    mut on_audio: impl FnMut(&[f32]) -> Result<()>,
) -> Result<Streamed> {
    let model = PythonWorker::spawn(worker).context("starting the model worker")?;
    let device = model.hello().device.clone();
    let engine = Engine::new(model, 2, WINDOW, cutoff_hz)?;
    let mut pipeline = Pipeline::new(engine, 2, WINDOW, OVERLAP, seed)?;

    let mut report = crate::limit::LimitReport::default();
    let mut emitted = 0usize;
    let mut carry_left: Vec<f32> = Vec::new();
    let mut carry_right: Vec<f32> = Vec::new();

    // Loudness is matched from the first window, then held for the whole track: a
    // per-window gain would pump, and the whole point is that the switch does not move
    // the level. The first window is the only stretch where input and output are the
    // same length and both available before anything is published.
    let head = track.frames.min(WINDOW);
    let lufs_input =
        crate::loudness::integrated(&[&track.channel(0)[..head], &track.channel(1)[..head]]);
    let mut gain: Option<f64> = None;
    let mut lufs_output = f64::NEG_INFINITY;

    // The splice needs the input that produced this window, which is the same frame range
    // of the track: the pipeline keeps output aligned with input, sample for sample.
    let splicer = match (splice_low_band, cutoff_hz) {
        (true, Some(cutoff)) => Some(Preprocessor::new(cutoff)?),
        _ => None,
    };

    let mut emit = |out: &Block, report: &mut crate::limit::LimitReport| -> Result<()> {
        carry_left.clear();
        carry_right.clear();
        carry_left.extend_from_slice(out.channel(0));
        carry_right.extend_from_slice(out.channel(1));
        let frames = out.frames();

        let mut planar = Vec::with_capacity(frames * 2);
        if let Some(preprocessor) = &splicer {
            let offset = emitted;
            for channel in 0..2 {
                let generated = if channel == 0 {
                    &carry_left
                } else {
                    &carry_right
                };
                let input = &track.channel(channel)[offset..offset + frames];
                planar.extend(crate::splice::channel(generated, input, preprocessor)?);
            }
        } else {
            planar.extend_from_slice(&carry_left);
            planar.extend_from_slice(&carry_right);
        }

        // First block: measure it against the input it came from, then fix the gain for
        // the rest of the track.
        let applied = match gain {
            Some(value) => value,
            None => {
                let (left, right) = planar.split_at(frames);
                lufs_output = crate::loudness::integrated(&[left, right]);
                let db = crate::loudness::gain_db(lufs_input, lufs_output)
                    .clamp(-MAX_GAIN_DB, MAX_GAIN_DB);
                let factor = 10f64.powf(db / 20.0);
                gain = Some(factor);
                factor
            }
        };
        if applied != 1.0 {
            let factor = applied as f32;
            for sample in planar.iter_mut() {
                *sample *= factor;
            }
        }

        let chunk = crate::limit::limit_in_place(&mut planar);
        report.peak_before = report.peak_before.max(chunk.peak_before);
        report.peak_after = report.peak_after.max(chunk.peak_after);
        report.limited_samples += chunk.limited_samples;
        on_audio(&planar)?;
        emitted += frames;
        Ok(())
    };

    let mut start = 0;
    while start < track.frames {
        let end = (start + CHUNK).min(track.frames);
        let mut interleaved = Vec::with_capacity((end - start) * 2);
        interleaved.extend_from_slice(&track.channel(0)[start..end]);
        interleaved.extend_from_slice(&track.channel(1)[start..end]);
        let block = Block::from_planar(2, interleaved)?;
        for out in pipeline.push(&block)? {
            emit(&out, &mut report)?;
        }
        progress(end);
        start = end;
    }
    for out in pipeline.finish()? {
        emit(&out, &mut report)?;
    }

    if emitted != track.frames {
        anyhow::bail!("pipeline returned {emitted} frames for {} in", track.frames);
    }
    let gain_db = gain.map_or(0.0, |factor| 20.0 * f64::from(factor as f32).log10());
    Ok(Streamed {
        limit: report,
        device,
        gain_db,
        lufs_input,
        lufs_output,
    })
}

/// Render into the playback frontier, publishing each finished second.
///
/// Returns the full render as well, because the cache wants the whole track; the caller
/// decides when to store it.
pub fn render_into_frontier(
    track: &Track,
    cutoff_hz: Option<u32>,
    worker: &WorkerConfig,
    seed: u64,
    splice_low_band: bool,
    frontier: &crate::frontier::Frontier,
    progress: &mut dyn FnMut(usize),
) -> Result<Render> {
    let mut planar = Vec::with_capacity(track.frames * 2);
    let mut interleaved = Vec::with_capacity(crate::frontier::CHUNK_FRAMES * 2);
    let mut published = 0usize;

    let streamed = enhance_streaming(
        track,
        cutoff_hz,
        worker,
        seed,
        splice_low_band,
        progress,
        |chunk| {
            planar.extend_from_slice(chunk);
            let frames = chunk.len() / 2;
            for frame in 0..frames {
                interleaved.push(chunk[frame]);
                interleaved.push(chunk[frames + frame]);
            }
            while interleaved.len() >= crate::frontier::CHUNK_FRAMES * 2 {
                let chunk: Vec<f32> = interleaved
                    .drain(..crate::frontier::CHUNK_FRAMES * 2)
                    .collect();
                frontier.publish(published, chunk);
                published += 1;
            }
            Ok(())
        },
    )?;

    if !interleaved.is_empty() {
        frontier.publish(published, interleaved);
    }
    Ok(Render {
        planar,
        limit: streamed.limit,
        device: streamed.device,
        gain_db: streamed.gain_db,
        lufs_input: streamed.lufs_input,
        lufs_output: streamed.lufs_output,
    })
}

/// Render the track, or take it from the cache when this exact render already exists.
///
/// Returns the audio and whether it came from disk, so the UI can say "cached" instead
/// of pretending it did 95 s of work.
pub fn enhance_or_cached(
    track: &Track,
    cutoff_hz: Option<u32>,
    worker: &WorkerConfig,
    cache: &crate::cache::Cache,
    model_id: &str,
    splice_low_band: bool,
    progress: &mut dyn FnMut(usize),
) -> Result<(Render, bool)> {
    let key = crate::cache::key(
        &sample_hash(track),
        WINDOW,
        OVERLAP,
        model_id,
        splice_low_band,
    );
    if let Some(cached) = cache.load(&key)? {
        progress(track.frames);
        let render = Render {
            planar: cached.planar,
            limit: crate::limit::LimitReport {
                peak_before: cached.meta.peak_before,
                peak_after: cached.meta.peak_after,
                limited_samples: cached.meta.limited_samples,
            },
            device: cached.meta.device,
            gain_db: cached.meta.gain_db,
            lufs_input: 0.0,
            lufs_output: 0.0,
        };
        return Ok((render, true));
    }

    let render = enhance(
        track,
        cutoff_hz,
        worker,
        seed_for(track),
        splice_low_band,
        progress,
    )?;
    if let Some(cutoff) = cutoff_hz {
        cache.store(&key, &render, cutoff)?;
    }
    Ok((render, false))
}

/// `flacify enhance` in one call: decode, decide, render, write.
pub fn enhance_file(
    input: &Path,
    output: &Path,
    worker: &WorkerConfig,
    cache: &crate::cache::Cache,
    model_id: &str,
    splice_low_band: bool,
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
    let (rendered, from_cache) = enhance_or_cached(
        &track,
        Some(analysis.cutoff_hz),
        worker,
        cache,
        model_id,
        splice_low_band,
        &mut |done| progress(done, frames),
    )?;
    if from_cache {
        eprintln!("cache hit: {frames} frames, nothing to render");
    }
    eprintln!("model on {}", rendered.device);
    if rendered.gain_db.abs() > 0.05 {
        eprintln!(
            "loudness  {:+.2} dB applied (input {:.1} LUFS, generated {:.1} LUFS before matching)",
            rendered.gain_db, rendered.lufs_input, rendered.lufs_output
        );
    }
    if splice_low_band {
        eprintln!("low band  from the input (experimental splice)");
    }
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
