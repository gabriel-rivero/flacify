//! One track through the model, start to finish.
//!
//! This is the only place the pipeline meets the Python sidecar, and the only place
//! that knows about the model's window. Everything it produces is planar stereo at
//! 48 kHz, exactly as long as the input: the pipeline's job is that invariant, and the
//! `debug_assert` at the end is the cheap way to notice if it ever stops holding.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use flacify_core::{Block, Pipeline, Preprocessor, SAMPLE_RATE};
use flacify_infer::{Engine, WorkerSession};

use crate::decode;

/// The render was superseded by another file. Not a failure: the window moved on.
#[derive(Debug)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("render cancelled")
    }
}

impl std::error::Error for Cancelled {}

/// Whether an error means the render was cancelled rather than broken.
pub fn is_cancelled(error: &anyhow::Error) -> bool {
    error.downcast_ref::<Cancelled>().is_some()
}

/// One window of check: cancelled renders stop before the next expensive call.
fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        return Err(anyhow::Error::new(Cancelled));
    }
    Ok(())
}

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
    /// blake3 of the file's bytes, the identity part of a cache key. Computed in [`load`]
    /// so nothing downstream reopens the file.
    pub source_hash: String,
    /// Per-second RMS of the 48 kHz source, for the waveform.
    #[allow(dead_code)] // Carried for the waveform; the window does not read it yet.
    pub peaks: Vec<crate::source::SecondPeaks>,
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

    /// The one place a track becomes a [`Loaded`], wherever the audio came from.
    fn from_track(
        track: Track,
        source: Source,
        source_hash: String,
        peaks: Vec<crate::source::SecondPeaks>,
        path: &Path,
    ) -> Result<Self> {
        let analysis = flacify_core::dsp::cutoff::analyse(track.channel(0))
            .with_context(|| format!("analysing {}", path.display()))?;
        Ok(Self {
            track,
            analysis,
            source,
            source_hash,
            peaks,
        })
    }
}

/// Decode any supported file to 48 kHz stereo, keep that source on disk, and report what
/// the spectrum looks like.
///
/// `cache_root` is the render cache's directory; the source tier lives in its `source/`.
pub fn load(path: &Path, cache_root: &Path) -> Result<Loaded> {
    let source_hash = crate::cache::source_hash(path)?;
    let frames_per_segment = crate::cache::frames_per_segment(WINDOW, OVERLAP)?;
    let tier = crate::source::Tier::new(cache_root, frames_per_segment)?;
    let entry = tier.build_or_reuse(path, &source_hash)?;
    let source = Source {
        sample_rate: entry.meta().native_sample_rate,
        channels: entry.meta().native_channels,
    };
    let planar = entry.read_all()?;
    let frames = planar.len() / 2;
    let track = Track { planar, frames };
    Loaded::from_track(track, source, source_hash, entry.meta().peaks.clone(), path)
}

/// Decode any supported file to 48 kHz stereo in memory and write nothing.
///
/// [`load`] materialises the source tier; a diagnosis must not spend half a gigabyte of
/// disk (or fail on a read-only cache) to print a verdict, so `probe` takes this path.
pub fn load_in_memory(path: &Path) -> Result<Loaded> {
    let source_hash = crate::cache::source_hash(path)?;
    let decoded = decode::decode(path)?;
    let source = Source {
        sample_rate: decoded.sample_rate,
        channels: decoded.channels,
    };
    let planar = decode::to_stereo_48k(&decoded)?;
    let frames = planar.len() / 2;
    let peaks = crate::source::second_peaks(&planar, frames);
    let track = Track { planar, frames };
    Loaded::from_track(track, source, source_hash, peaks, path)
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

/// Everything one render needs besides the audio: the model, the band, the seed and
/// whether to keep the input's low band.
pub struct Job {
    pub session: Arc<WorkerSession>,
    pub cutoff_hz: Option<u32>,
    pub seed: u64,
    pub splice_low_band: bool,
}

/// Where a finished render goes, and what identifies it in the cache.
pub struct Store<'a> {
    pub cache: &'a crate::cache::Cache,
    pub model_id: &'a str,
}

/// Collect the per-block `[left, right]` chunks the pipeline emits into one channel-major
/// buffer of the shape `Track::planar` uses.
///
/// Each chunk is planar *within* a block, so appending them whole interleaves the channels
/// every `step` samples. The cache, the WAV writer and `interleave_stereo` all expect the
/// whole track channel-major; this is the one place that joins them.
#[derive(Default)]
struct ChannelMajor {
    left: Vec<f32>,
    right: Vec<f32>,
}

impl ChannelMajor {
    fn with_capacity(frames: usize) -> Self {
        Self {
            left: Vec::with_capacity(frames),
            right: Vec::with_capacity(frames),
        }
    }

    fn push(&mut self, chunk: &[f32]) {
        let frames = chunk.len() / 2;
        self.left.extend_from_slice(&chunk[..frames]);
        self.right.extend_from_slice(&chunk[frames..]);
    }

    fn finish(self) -> Vec<f32> {
        let mut planar = self.left;
        planar.extend_from_slice(&self.right);
        planar
    }
}

/// Render a whole track. `progress` is called with the number of input frames covered
/// each time a chunk comes back from the model, so a UI can show real movement.
pub fn enhance(
    track: &Track,
    job: Job,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(usize),
) -> Result<Render> {
    let mut buffer = ChannelMajor::with_capacity(track.frames);
    let streamed = enhance_streaming(track, job, cancel, progress, |chunk| {
        buffer.push(chunk);
        Ok(())
    })?;
    Ok(Render {
        planar: buffer.finish(),
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
    job: Job,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(usize),
    mut on_audio: impl FnMut(&[f32]) -> Result<()>,
) -> Result<Streamed> {
    let model = job.session.model().context("starting the model worker")?;
    let hello = model.hello();
    // The worker's own "ready on cuda" line goes to its stderr, which is captured and
    // only surfaced when an inference call fails; say it out loud here instead, where
    // RUST_LOG can reach it. On a resident worker this repeats per file, which is the
    // price of not threading a "did it just start" flag through the session.
    tracing::info!(
        device = %hello.device,
        torch = %hello.torch,
        load_secs = hello.load_secs,
        "model ready"
    );
    let device = hello.device.clone();
    let engine = Engine::new(model, 2, WINDOW, job.cutoff_hz)?;
    let mut pipeline = Pipeline::new(engine, 2, WINDOW, OVERLAP, job.seed)?;

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
    let splicer = match (job.splice_low_band, job.cutoff_hz) {
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
        check_cancel(cancel)?;
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
    check_cancel(cancel)?;
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
    job: Job,
    frontier: &crate::frontier::Frontier,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(usize),
) -> Result<Render> {
    let mut buffer = ChannelMajor::with_capacity(track.frames);
    let mut interleaved = Vec::with_capacity(crate::frontier::CHUNK_FRAMES * 2);
    let mut published = 0usize;

    let streamed = enhance_streaming(track, job, cancel, progress, |chunk| {
        buffer.push(chunk);
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
    })?;

    if !interleaved.is_empty() {
        frontier.publish(published, interleaved);
    }
    Ok(Render {
        planar: buffer.finish(),
        limit: streamed.limit,
        device: streamed.device,
        gain_db: streamed.gain_db,
        lufs_input: streamed.lufs_input,
        lufs_output: streamed.lufs_output,
    })
}

/// Render the track, or take it from the cache when this exact render already exists.
///
/// `source_hash` is [`crate::cache::source_hash`] of the file `track` came from. Returns
/// the audio and whether it came from disk, so the UI can say "cached" instead of
/// pretending it did 95 s of work.
pub fn enhance_or_cached(
    track: &Track,
    source_hash: &str,
    job: Job,
    store: Store,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(usize),
) -> Result<(Render, bool)> {
    let fields = crate::cache::Fields {
        source_hash: source_hash.to_string(),
        window: WINDOW,
        overlap: OVERLAP,
        model_id: store.model_id.to_string(),
        splice: job.splice_low_band,
    };
    let key = fields.key();
    if let Some(cached) = store.cache.load(&key)? {
        if cached.meta.source_hash.is_empty() {
            let _ = store.cache.annotate(&key, &fields);
        }
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

    let cutoff_hz = job.cutoff_hz;
    let render = enhance(track, job, cancel, progress)?;
    if let Some(cutoff) = cutoff_hz {
        store.cache.store(&key, &render, cutoff, &fields)?;
    }
    Ok((render, false))
}

/// `flacify enhance` in one call: decode, decide, render, write.
pub fn enhance_file(
    input: &Path,
    output: &Path,
    session: Arc<WorkerSession>,
    store: Store,
    splice_low_band: bool,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<()> {
    let Loaded {
        track,
        analysis,
        source,
        source_hash,
        peaks: _,
    } = load(input, store.cache.root())?;
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
        write_output(output, &planar)?;
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
        &source_hash,
        Job {
            session,
            cutoff_hz: Some(analysis.cutoff_hz),
            seed: seed_for(&track),
            splice_low_band,
        },
        store,
        cancel,
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
    write_output(output, &rendered.planar)?;
    Ok(())
}

/// Write a rendered track, choosing the container from the output name.
///
/// `.flac` writes 24-bit FLAC; every other name keeps the float WAV `enhance` has always
/// produced.
fn write_output(output: &Path, planar: &[f32]) -> Result<()> {
    if is_flac_output(output) {
        crate::flac::write(output, SAMPLE_RATE, 2, planar)
    } else {
        crate::wav::write(output, SAMPLE_RATE, 2, planar)
    }
}

/// Whether the output name asks for FLAC. The extension is matched case-insensitively.
fn is_flac_output(output: &Path) -> bool {
    output
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("flac"))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;

    static NEXT_OUTPUT: AtomicU32 = AtomicU32::new(0);

    /// A unique output path in the temp directory, deleted when the test ends.
    struct TempOutput {
        path: std::path::PathBuf,
    }

    impl TempOutput {
        fn new(name: &str) -> Self {
            let unique = NEXT_OUTPUT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "flacify-render-{}-{unique}-{name}",
                std::process::id()
            ));
            Self { path }
        }
    }

    impl Drop for TempOutput {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    /// A unique cache root in the temp directory, deleted when the test ends.
    struct TempRoot {
        path: std::path::PathBuf,
    }

    impl TempRoot {
        fn new(name: &str) -> Self {
            let unique = NEXT_OUTPUT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "flacify-render-root-{}-{unique}-{name}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).expect("creating the test cache root");
            Self { path }
        }
    }

    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    /// One block of a constant signal: enough audio for a well-formed stream, cheap to
    /// encode.
    fn one_block() -> Vec<f32> {
        vec![0.25; 2 * crate::flac::BLOCK]
    }

    #[test]
    fn given_output_names_when_classified_then_only_a_flac_extension_selects_flac() {
        assert!(is_flac_output(Path::new("out.flac")));
        assert!(is_flac_output(Path::new("OUT.FLAC")));
        assert!(!is_flac_output(Path::new("out.wav")));
        assert!(!is_flac_output(Path::new("out.flac.wav")));
        assert!(!is_flac_output(Path::new("out")));
    }

    #[test]
    fn given_a_flac_output_path_when_written_then_the_file_is_a_flac_stream() {
        let output = TempOutput::new("container.flac");
        write_output(&output.path, &one_block()).expect("writing a flac");
        let bytes = std::fs::read(&output.path).expect("reading the flac");
        assert_eq!(&bytes[..4], b"fLaC");
    }

    #[test]
    fn given_a_wav_output_path_when_written_then_the_file_is_a_wave_container() {
        let output = TempOutput::new("container.wav");
        write_output(&output.path, &one_block()).expect("writing a wav");
        let bytes = std::fs::read(&output.path).expect("reading the wav");
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
    }

    #[test]
    fn given_a_set_token_when_checked_then_the_render_stops_as_cancelled() {
        let cancel = AtomicBool::new(true);
        let error = check_cancel(&cancel).unwrap_err();
        assert!(is_cancelled(&error), "{error:#}");
        assert_eq!(error.to_string(), "render cancelled");
    }

    #[test]
    fn given_an_unset_token_when_checked_then_it_is_not_cancelled() {
        let cancel = AtomicBool::new(false);
        assert!(check_cancel(&cancel).is_ok());
    }

    #[test]
    fn given_another_error_when_asked_then_it_is_not_a_cancellation() {
        let error = anyhow::anyhow!("the worker exited");
        assert!(!is_cancelled(&error));
    }

    #[test]
    fn given_per_block_chunks_when_collected_then_each_channel_is_contiguous() {
        // The pipeline emits `[left, right]` per block. Appending those whole would put
        // the right channel's first block at sample `step`; the cache would then read the
        // right channel from the middle of the track.
        let mut collected = ChannelMajor::with_capacity(5);
        collected.push(&[1.0, 2.0, 10.0, 20.0]);
        collected.push(&[3.0, 4.0, 30.0, 40.0]);
        collected.push(&[5.0, 50.0]);
        assert_eq!(
            collected.finish(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        );
    }

    #[test]
    fn given_a_collected_channel_major_buffer_when_interleaved_then_the_first_frames_pair_up() {
        // The cache-hit path is exactly this: collect channel-major, then interleave. Frame
        // k must be (left[k], right[k]), not (left[k], right of some later block).
        let mut collected = ChannelMajor::with_capacity(4);
        collected.push(&[1.0, 2.0, 10.0, 20.0]);
        collected.push(&[3.0, 4.0, 30.0, 40.0]);
        let planar = collected.finish();
        assert_eq!(
            interleave_stereo(&planar, 4),
            vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]
        );
    }

    /// A 48 kHz stereo WAV with a 100 ms tone on each channel: long enough for the cutoff
    /// analysis, small enough to write per test.
    fn write_source(name: &str) -> TempOutput {
        let output = TempOutput::new(name);
        let frames = 4_800;
        let mut planar = Vec::with_capacity(frames * 2);
        for channel in 0..2 {
            for frame in 0..frames {
                planar.push(0.2 * ((frame as f32 * 0.05).sin() + channel as f32 * 0.25));
            }
        }
        crate::wav::write(&output.path, SAMPLE_RATE, 2, &planar).expect("writing the wav");
        output
    }

    #[test]
    fn given_a_file_when_loaded_then_the_source_hash_is_the_cache_hash_of_that_path() {
        let source = write_source("source-hash.wav");
        let root = TempRoot::new("source-hash");
        let loaded = load(&source.path, &root.path).expect("loading the file");

        assert_eq!(loaded.source_hash.len(), 32);
        assert_eq!(
            loaded.source_hash,
            crate::cache::source_hash(&source.path).expect("hashing the file")
        );
    }

    #[test]
    fn given_a_file_when_loaded_then_the_source_tier_holds_it_and_a_second_load_reuses_it() {
        let source = write_source("load-source.wav");
        let root = TempRoot::new("load-source");
        let loaded = load(&source.path, &root.path).expect("loading the file");

        let decoded = crate::decode::decode(&source.path).expect("decoding the source");
        assert_eq!(loaded.track.frames, 4_800);
        assert_eq!(loaded.source.sample_rate, decoded.sample_rate);
        assert_eq!(loaded.source.channels, decoded.channels);
        assert_eq!(loaded.peaks.len(), 1, "100 ms is one waveform second");

        let entry = root.path.join("source").join(&loaded.source_hash);
        assert!(entry.join("source.json").is_file());
        assert!(entry.join("src-000000.flac").is_file());
        let manifest = entry.join("source.json");
        let written = std::fs::metadata(&manifest)
            .expect("measuring the manifest")
            .modified()
            .expect("a modification time");

        let again = load(&source.path, &root.path).expect("loading again");
        assert_eq!(again.track.planar, loaded.track.planar);
        assert_eq!(
            std::fs::metadata(&manifest)
                .expect("measuring the manifest")
                .modified()
                .expect("a modification time"),
            written,
            "a second load reuses the entry instead of rewriting it"
        );
    }

    #[test]
    fn given_a_file_when_loaded_in_memory_then_it_matches_a_one_shot_decode() {
        // The property that matters here is structural: `load_in_memory` takes no cache
        // root, so it cannot write an entry. This test pins the audio and the peaks.
        let source = write_source("probe.wav");

        let loaded = load_in_memory(&source.path).expect("loading in memory");
        let decoded = crate::decode::decode(&source.path).expect("decoding the source");
        let expected = crate::decode::to_stereo_48k(&decoded).expect("resampling the source");

        assert_eq!(loaded.track.frames, 4_800);
        let bits = |samples: &[f32]| {
            samples
                .iter()
                .map(|sample| sample.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&loaded.track.planar), bits(&expected));
        assert_eq!(
            loaded.peaks,
            crate::source::second_peaks(&expected, 4_800),
            "an in-memory load still reports the waveform"
        );
        assert_eq!(loaded.peaks.len(), 1);
    }

    /// A fixed signal with no platform-dependent rounding: a 4 096-frame ramp and its
    /// negation.
    fn fixed_signal() -> Track {
        let frames = 4_096;
        let mut planar = Vec::with_capacity(frames * 2);
        for frame in 0..frames {
            let value = (frame as f32 - 2_048.0) / 4_096.0;
            planar.push(value);
            planar.push(-value);
        }
        Track { planar, frames }
    }

    #[test]
    fn given_a_fixed_signal_when_seeded_then_the_seed_is_the_pinned_constant() {
        // ADR-006: the seed is a compatibility contract, so a change here re-renders every
        // track with different noise instead of reusing the same one.
        assert_eq!(seed_for(&fixed_signal()), 8_901_150_270_194_604_865);
    }
}
