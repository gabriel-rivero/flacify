//! One track through the model, start to finish.
//!
//! This is the only place the pipeline meets the Python sidecar, and the only place
//! that knows about the model's window. Everything it produces is planar stereo at
//! 48 kHz, exactly as long as the input: the pipeline's job is that invariant, and the
//! length check at the end is the cheap way to notice if it ever stops holding.
//!
//! Two ways to drive it: [`enhance_streaming`] keeps the whole track and writes every
//! emitted block to its caller, and [`render_segments`] reads the track one source segment
//! at a time and writes finished cache segments, so a render holds a segment rather than a
//! track and a kill leaves a resumable entry.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use flacify_core::{Block, Pipeline, Preprocessor, Tail, SAMPLE_RATE};
use flacify_infer::{Engine, ResidentModel, WorkerSession};

use crate::cache;
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
/// Frames handed to the pipeline per push: one step, `window - overlap`.
///
/// That is the amount of input that produces exactly one emitted window, so the frontier
/// gains a second of audio every time the model finishes one, and it is the step
/// `Pipeline::start_at` positions a range on. Pushing the whole track at once (which is
/// what this used to do) means the frontier stays empty until the render is over, and
/// progressive playback cannot work at all.
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

/// Where a render reads its input, a range at a time: a whole track in memory, or the
/// source tier's segments on disk.
pub trait Frames {
    /// Frames in the whole track.
    fn frames(&self) -> usize;

    /// Planar 48 kHz stereo for `[start, start + frames)`.
    fn read_frames(&self, start: usize, frames: usize) -> Result<Vec<f32>>;
}

impl Frames for Track {
    fn frames(&self) -> usize {
        self.frames
    }

    fn read_frames(&self, start: usize, frames: usize) -> Result<Vec<f32>> {
        anyhow::ensure!(
            start
                .checked_add(frames)
                .is_some_and(|end| end <= self.frames),
            "range {start} + {frames} frames is past the {} of the track",
            self.frames
        );
        let mut planar = Vec::with_capacity(frames * 2);
        planar.extend_from_slice(&self.planar[start..start + frames]);
        let right = self.frames + start;
        planar.extend_from_slice(&self.planar[right..right + frames]);
        Ok(planar)
    }
}

impl Frames for crate::source::Entry {
    fn frames(&self) -> usize {
        self.meta().frames
    }

    fn read_frames(&self, start: usize, frames: usize) -> Result<Vec<f32>> {
        self.read_range(start, frames)
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
    /// The source tier entry a segmented render reads its ranges from. Only [`load`] has
    /// one: [`load_in_memory`] writes nothing and never renders.
    pub source_entry: Option<crate::source::Entry>,
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
        source_entry: Option<crate::source::Entry>,
        source: Source,
        source_hash: String,
        peaks: Vec<crate::source::SecondPeaks>,
        path: &Path,
    ) -> Result<Self> {
        let analysis = analysis_of(track.channel(0))
            .with_context(|| format!("analysing {}", path.display()))?;
        Ok(Self {
            track,
            analysis,
            source,
            source_entry,
            source_hash,
            peaks,
        })
    }
}

/// Frames the analysis reads per push: one second.
const ANALYSIS_CHUNK: usize = SAMPLE_RATE as usize;

/// The band limit of one channel, pushed a second at a time.
///
/// The chunked analyser is the one a range reader can feed without holding the channel, and
/// its verdict is pinned to [`analyse`](flacify_core::dsp::cutoff::analyse) by the equality
/// tests.
fn analysis_of(channel: &[f32]) -> Result<flacify_core::dsp::cutoff::Analysis> {
    let mut analyser = flacify_core::dsp::cutoff::chunked();
    for chunk in channel.chunks(ANALYSIS_CHUNK) {
        analyser.push(chunk)?;
    }
    Ok(analyser.finish()?)
}

/// The band limit of a range source, pushed a second at a time: what a complete cache entry
/// needs before it can say whether it would render anything, without holding the track.
///
/// The chunks are [`ANALYSIS_CHUNK`]-frame slices of the left channel, the same ones
/// [`analysis_of`] pushes, so a range source gives the verdict a whole track does.
pub fn analyse_streaming(input: &dyn Frames) -> Result<flacify_core::dsp::cutoff::Analysis> {
    let frames = input.frames();
    let mut analyser = flacify_core::dsp::cutoff::chunked();
    let mut start = 0;
    while start < frames {
        let count = ANALYSIS_CHUNK.min(frames - start);
        let planar = input.read_frames(start, count)?;
        anyhow::ensure!(
            planar.len() == count * 2,
            "range {start} + {count} frames came back as {} samples",
            planar.len()
        );
        analyser.push(&planar[..count])?;
        start += count;
    }
    Ok(analyser.finish()?)
}

/// Decode any supported file to 48 kHz stereo, keep that source on disk, and report what
/// the spectrum looks like.
///
/// `cache_root` is the render cache's directory; the source tier lives in its `source/`.
pub fn load(path: &Path, cache_root: &Path) -> Result<Loaded> {
    let source_hash = crate::cache::source_hash(path)?;
    load_hashed(path, cache_root, source_hash)
}

/// [`load`] with the file's hash already computed, so a caller that needed it for a cache
/// lookup does not read the whole file twice.
pub fn load_hashed(path: &Path, cache_root: &Path, source_hash: String) -> Result<Loaded> {
    let frames_per_segment = crate::cache::frames_per_segment(WINDOW, OVERLAP)?;
    let tier = crate::source::Tier::new(cache_root, frames_per_segment)?;
    let entry = tier.build_or_reuse(path, &source_hash)?;
    let source = Source {
        sample_rate: entry.meta().native_sample_rate,
        channels: entry.meta().native_channels,
    };
    let planar = entry.read_all()?;
    let frames = planar.len() / 2;
    let peaks = entry.meta().peaks.clone();
    let track = Track { planar, frames };
    Loaded::from_track(track, Some(entry), source, source_hash, peaks, path)
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
    Loaded::from_track(track, None, source, source_hash, peaks, path)
}

/// Frames of source per read when hashing a seed: one second.
const SEED_RANGE: usize = SAMPLE_RATE as usize;

/// The ADR-006 seed's byte layout, fed in sample order.
///
/// The value is a compatibility contract, so the tag and the per-sample encoding live
/// here and nowhere else.
struct SeedHasher {
    hasher: blake3::Hasher,
}

impl SeedHasher {
    fn new() -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"flacify-pipeline-1");
        Self { hasher }
    }

    fn push(&mut self, samples: &[f32]) {
        for sample in samples {
            self.hasher.update(&sample.to_bits().to_le_bytes());
        }
    }

    fn finish(self) -> u64 {
        u64::from_le_bytes(
            self.hasher.finalize().as_bytes()[..8]
                .try_into()
                .expect("8 bytes"),
        )
    }
}

/// A seed that survives re-renders, cache keys and bug reports (ADR-006).
pub fn seed_for(track: &Track) -> u64 {
    let mut hasher = SeedHasher::new();
    hasher.push(&track.planar);
    hasher.finish()
}

/// The seed [`seed_for`] returns, hashed from [`Frames`] ranges so no whole track is
/// held.
///
/// A range is planar stereo while the seed is the whole track's left channel followed by
/// its right, so each channel is hashed in its own pass over the input.
pub fn seed_streaming(input: &dyn Frames) -> Result<u64> {
    let mut hasher = SeedHasher::new();
    for channel in 0..2 {
        hash_channel(input, channel, &mut hasher)?;
    }
    Ok(hasher.finish())
}

/// Adds one channel of `input` to `hasher`, [`SEED_RANGE`] frames at a time.
fn hash_channel(input: &dyn Frames, channel: usize, hasher: &mut SeedHasher) -> Result<()> {
    let frames = input.frames();
    let mut start = 0;
    while start < frames {
        let count = SEED_RANGE.min(frames - start);
        let planar = input.read_frames(start, count)?;
        anyhow::ensure!(
            planar.len() == count * 2,
            "range {start} + {count} frames came back as {} samples",
            planar.len()
        );
        let offset = channel * count;
        hasher.push(&planar[offset..offset + count]);
        start += count;
    }
    Ok(())
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
    let streamed = enhance_streaming(track, job, cancel, progress, |_, chunk| {
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

/// What a render produced besides the audio.
#[derive(Debug)]
pub struct Streamed {
    pub limit: crate::limit::LimitReport,
    pub device: String,
    /// Loudness gain applied to the generated side, in dB, as the block processing fixed
    /// it: the value the samples were scaled by, so a resume reproduces the same factor.
    pub gain_db: f64,
    /// Integrated loudness of the input's first window and of the first generated block
    /// before the gain. `NEG_INFINITY` when this process did not measure them, which is a
    /// resumed render taking its gain from the entry.
    pub lufs_input: f64,
    pub lufs_output: f64,
}

/// The most a loudness match may move the generated side. Beyond this something is wrong
/// with the measurement and a big gain would sound worse than the step it fixes.
const MAX_GAIN_DB: f64 = 6.0;

/// The pipeline, emitting audio as it arrives instead of at the end.
///
/// `on_audio` receives the block's absolute start frame and its planar stereo (the limiter
/// has already run, so what it gets is final) and may be called many times. The limiter is
/// a per-sample function, so limiting each block gives exactly the same samples as
/// limiting the whole track afterwards, which is what lets the frontier publish before
/// the render finishes.
pub fn enhance_streaming(
    track: &Track,
    job: Job,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(usize),
    mut on_audio: impl FnMut(usize, &[f32]) -> Result<()>,
) -> Result<Streamed> {
    let model = RenderModel::start(&job)?;
    let mut processor = BlockProcessor::new(track, &job, cache::EntryProgress::default())?;
    let mut pipeline = model.pipeline()?;

    let mut fed = 0;
    let mut at = 0;
    while fed < track.frames {
        check_cancel(cancel)?;
        let end = (fed + CHUNK).min(track.frames);
        let block = read_block(track, fed, end)?;
        for out in pipeline.push(&block)? {
            check_cancel(cancel)?;
            let chunk = processor.process(&out, at)?;
            on_audio(at, &chunk)?;
            at += out.frames();
        }
        progress(end);
        fed = end;
    }
    check_cancel(cancel)?;
    for out in pipeline.finish()? {
        check_cancel(cancel)?;
        let chunk = processor.process(&out, at)?;
        on_audio(at, &chunk)?;
        at += out.frames();
    }
    anyhow::ensure!(
        at == track.frames,
        "the pipeline returned {at} frames for {} in",
        track.frames
    );
    Ok(processor.into_streamed(model.device))
}

/// The model one render drives, and how to build a pipeline range from it.
struct RenderModel {
    model: ResidentModel,
    device: String,
    cutoff_hz: Option<u32>,
    seed: u64,
}

impl RenderModel {
    fn start(job: &Job) -> Result<Self> {
        let model = Arc::clone(&job.session)
            .model()
            .context("starting the model worker")?;
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
        Ok(Self {
            device: hello.device.clone(),
            model,
            cutoff_hz: job.cutoff_hz,
            seed: job.seed,
        })
    }

    /// A pipeline for one range: a pipeline owns its upscaler, so every range builds one.
    fn pipeline(&self) -> Result<Pipeline<Engine<ResidentModel>>> {
        let engine = Engine::new(self.model.clone(), 2, WINDOW, self.cutoff_hz)?;
        Ok(Pipeline::new(engine, 2, WINDOW, OVERLAP, self.seed)?)
    }
}

/// The per-block work both render paths share: splice, the track's loudness gain, the
/// limiter and what they report. `at` is the block's absolute frame in the track, which
/// the splice reads and which only the track's first block may be 0.
struct BlockProcessor<'a> {
    input: &'a dyn Frames,
    splicer: Option<Preprocessor>,
    gain_db: Option<f64>,
    report: crate::limit::LimitReport,
    lufs_input: f64,
    lufs_output: f64,
}

impl<'a> BlockProcessor<'a> {
    fn new(input: &'a dyn Frames, job: &Job, recorded: cache::EntryProgress) -> Result<Self> {
        // The splice needs the input that produced this window, which is the same frame
        // range of the track: the pipeline keeps output aligned with input, sample for
        // sample.
        let splicer = match (job.splice_low_band, job.cutoff_hz) {
            (true, Some(cutoff)) => Some(Preprocessor::new(cutoff)?),
            _ => None,
        };
        // Loudness is matched from the first window, then held for the whole track: a
        // per-window gain would pump, and the whole point is that the switch does not move
        // the level. A resumed render already has the gain, so it does not re-measure.
        let (lufs_input, lufs_output) = match recorded.gain_db {
            Some(_) => (f64::NEG_INFINITY, f64::NEG_INFINITY),
            None => (first_window_lufs(input)?, f64::NEG_INFINITY),
        };
        Ok(Self {
            input,
            splicer,
            gain_db: recorded.gain_db,
            report: recorded.limit,
            lufs_input,
            lufs_output,
        })
    }

    /// Finished planar stereo for one emitted block, limited. The first block of the track
    /// also fixes the gain, which every later block reuses.
    fn process(&mut self, out: &Block, at: usize) -> Result<Vec<f32>> {
        let frames = out.frames();
        let mut planar = Vec::with_capacity(frames * 2);
        match &self.splicer {
            Some(preprocessor) => {
                let input = self.input.read_frames(at, frames)?;
                anyhow::ensure!(
                    input.len() == frames * 2,
                    "the splice input for {at}..{} came back as {} samples",
                    at + frames,
                    input.len()
                );
                let (left, right) = input.split_at(frames);
                planar.extend(crate::splice::channel(out.channel(0), left, preprocessor)?);
                planar.extend(crate::splice::channel(out.channel(1), right, preprocessor)?);
            }
            None => {
                planar.extend_from_slice(out.channel(0));
                planar.extend_from_slice(out.channel(1));
            }
        }

        let db = match self.gain_db {
            Some(db) => db,
            None => {
                anyhow::ensure!(
                    at == 0,
                    "the loudness gain is fixed by the first block of the track"
                );
                let (left, right) = planar.split_at(frames);
                self.lufs_output = crate::loudness::integrated(&[left, right]);
                let db = crate::loudness::gain_db(self.lufs_input, self.lufs_output)
                    .clamp(-MAX_GAIN_DB, MAX_GAIN_DB);
                self.gain_db = Some(db);
                db
            }
        };
        let factor = gain_factor(db);
        if factor != 1.0 {
            for sample in &mut planar {
                *sample *= factor;
            }
        }

        let limited = crate::limit::limit_in_place(&mut planar);
        self.report.peak_before = self.report.peak_before.max(limited.peak_before);
        self.report.peak_after = self.report.peak_after.max(limited.peak_after);
        self.report.limited_samples += limited.limited_samples;
        Ok(planar)
    }

    fn progress(&self) -> cache::EntryProgress {
        cache::EntryProgress {
            gain_db: self.gain_db,
            limit: self.report,
        }
    }

    fn into_streamed(self, device: String) -> Streamed {
        Streamed {
            limit: self.report,
            device,
            gain_db: self.gain_db.unwrap_or(0.0),
            lufs_input: self.lufs_input,
            lufs_output: self.lufs_output,
        }
    }
}

/// The f32 the samples are scaled by for a gain in dB.
///
/// The dB is what the entry records; `powf` is deterministic, so replaying it gives back
/// the same f32 and a resumed render applies exactly the gain the first one did.
fn gain_factor(db: f64) -> f32 {
    10f64.powf(db / 20.0) as f32
}

/// Integrated loudness of the input's first window: the level the generated side is
/// matched to for the whole track.
fn first_window_lufs(input: &dyn Frames) -> Result<f64> {
    let head = input.frames().min(WINDOW);
    let planar = input.read_frames(0, head)?;
    anyhow::ensure!(
        planar.len() == head * 2,
        "the first window came back as {} samples for {head} frames",
        planar.len()
    );
    let (left, right) = planar.split_at(head);
    Ok(crate::loudness::integrated(&[left, right]))
}

/// One push of the pipeline: planar stereo `[start, end)` as a [`Block`].
fn read_block(input: &dyn Frames, start: usize, end: usize) -> Result<Block> {
    let frames = end - start;
    let planar = input.read_frames(start, frames)?;
    anyhow::ensure!(
        planar.len() == frames * 2,
        "range {start}..{end} came back as {} samples",
        planar.len()
    );
    Ok(Block::from_planar(2, planar)?)
}

/// One pipeline push out of a planar range already read from the tier.
fn planar_block(planar: &[f32], offset: usize, frames: usize) -> Result<Block> {
    let total = planar.len() / 2;
    let mut data = Vec::with_capacity(frames * 2);
    data.extend_from_slice(&planar[offset..offset + frames]);
    data.extend_from_slice(&planar[total + offset..total + offset + frames]);
    Ok(Block::from_planar(2, data)?)
}

/// Render the track into `writer` as cache segments, one source segment in memory at a
/// time, and complete the entry.
///
/// Segments the entry already records are skipped and never rewritten. A render that is
/// cancelled returns [`Cancelled`] and leaves the segments it wrote, which a later call
/// resumes.
pub fn render_segments(
    input: &dyn Frames,
    job: Job,
    writer: &mut cache::EntryWriter,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(usize),
    mut on_audio: impl FnMut(usize, &[f32]) -> Result<()>,
) -> Result<Streamed> {
    let frames = input.frames();
    anyhow::ensure!(frames > 0, "a cache entry needs at least one frame");
    let frames_per_segment = cache::frames_per_segment(WINDOW, OVERLAP)?;
    let segment_count = frames.div_ceil(frames_per_segment);
    let recorded = writer.progress();
    anyhow::ensure!(
        recorded.gain_db.is_some() || !writer.has_segment(0),
        "the cache entry records segment 0 without the loudness gain it fixed"
    );
    let model = RenderModel::start(&job)?;
    let mut processor = BlockProcessor::new(input, &job, recorded)?;

    let mut index = 0;
    let mut tail: Option<Tail> = None;
    while index < segment_count {
        check_cancel(cancel)?;
        if writer.has_segment(index) {
            // Its audio and its outgoing tail are both on disk or recoverable; a render
            // that has to continue after it reconstructs the tail then.
            tail = None;
            index += 1;
            continue;
        }
        let range_start = index * frames_per_segment;

        // One segment per range, so the output crosses a range boundary at every segment
        // and a resumed render can pick up at any of them. The exception is the track's
        // last segment when it is shorter than the lookahead a range needs: its final
        // window is short in a whole-track render, so it has to be rendered with the
        // range before it.
        let mut range_end = index + 1;
        let mut is_last_range = range_end == segment_count;
        if !is_last_range && range_end * frames_per_segment + OVERLAP > frames {
            anyhow::ensure!(
                !writer.has_segment(range_end),
                "the cache entry records segment {range_end}, which the track cannot hold"
            );
            range_end += 1;
            anyhow::ensure!(
                range_end == segment_count,
                "segment {index} needs a lookahead past the track before its last segment"
            );
            is_last_range = true;
        }

        if index > 0 && tail.is_none() {
            tail = Some(crossfade_tail(&model, input, (range_start / CHUNK) as u64)?);
        }
        let feed_end = if is_last_range {
            frames
        } else {
            range_end * frames_per_segment + OVERLAP
        };
        let mut pipeline = model.pipeline()?;
        pipeline.start_at((range_start / CHUNK) as u64, tail.take())?;
        let mut emission = Emission::new(&mut processor, index, frames_per_segment, frames);

        let mut fed = range_start;
        let mut at = range_start;
        while fed < feed_end {
            check_cancel(cancel)?;
            // One source segment per read, so the tier decodes each of its segments once,
            // and the pipeline is fed a step at a time so cancellation stays per window.
            let read_end = (fed + frames_per_segment).min(feed_end);
            let planar = input.read_frames(fed, read_end - fed)?;
            anyhow::ensure!(
                planar.len() == (read_end - fed) * 2,
                "range {fed}..{read_end} came back as {} samples",
                planar.len()
            );
            let read_frames = read_end - fed;
            let mut offset = 0;
            while offset < read_frames {
                check_cancel(cancel)?;
                let take = CHUNK.min(read_frames - offset);
                let block = planar_block(&planar, offset, take)?;
                for out in pipeline.push(&block)? {
                    check_cancel(cancel)?;
                    emission.emit(writer, &out, at, &mut on_audio)?;
                    at += out.frames();
                }
                offset += take;
                progress(fed + offset);
            }
            fed = read_end;
        }

        check_cancel(cancel)?;
        if is_last_range {
            for out in pipeline.finish()? {
                check_cancel(cancel)?;
                emission.emit(writer, &out, at, &mut on_audio)?;
                at += out.frames();
            }
        } else {
            tail = Some(pipeline.end_range()?);
        }
        emission.finish()?;
        index = range_end;
    }

    let progress = processor.progress();
    writer.finish(&cache::Report {
        limit: progress.limit,
        device: model.device.clone(),
        gain_db: progress.gain_db.unwrap_or(0.0),
    })?;
    Ok(processor.into_streamed(model.device))
}

/// The crossfade tail a segment on disk handed to the segment after it, recovered by
/// rendering that segment's last window.
///
/// The tail is the part of a window's output that no crossfade touches, so one model call
/// reproduces it exactly without re-rendering the recorded segment.
fn crossfade_tail(model: &RenderModel, input: &dyn Frames, window: u64) -> Result<Tail> {
    let before = window
        .checked_sub(1)
        .context("a range at window 0 has no tail to recover")?;
    let start = usize::try_from(before).context("a window index past the frame count")? * CHUNK;
    let mut pipeline = model.pipeline()?;
    pipeline.start_at(before, None)?;
    let block = read_block(input, start, start + WINDOW)?;
    // What this emits belongs to the recorded segment; only the held tail crosses.
    let _ = pipeline.push(&block)?;
    Ok(pipeline.end_range()?)
}

/// Collects the blocks of a segmented render into cache segments, which are written as
/// soon as they fill.
struct SegmentBuffer {
    /// The segment being filled, which is also the next one to write.
    index: usize,
    frames_per_segment: usize,
    /// Frames in the whole track, so the last segment can be short.
    frames: usize,
    left: Vec<f32>,
    right: Vec<f32>,
}

impl SegmentBuffer {
    fn new(first_index: usize, frames_per_segment: usize, frames: usize) -> Self {
        Self {
            index: first_index,
            frames_per_segment,
            frames,
            left: Vec::with_capacity(frames_per_segment),
            right: Vec::with_capacity(frames_per_segment),
        }
    }

    /// Append one planar block `[left, right]`, returning the segments it completes. The
    /// last window of a track emits more than one step, so a block can cross a segment
    /// boundary and even finish the short last segment as well.
    fn push(&mut self, chunk: &[f32]) -> Result<Vec<(usize, Vec<f32>)>> {
        anyhow::ensure!(
            chunk.len() % 2 == 0,
            "a stereo block of {} samples is not whole frames",
            chunk.len()
        );
        let frames = chunk.len() / 2;
        let mut finished = Vec::new();
        let mut offset = 0;
        while offset < frames {
            let expected =
                cache::expected_segment_frames(self.frames, self.frames_per_segment, self.index)
                    .with_context(|| format!("the entry has no segment {}", self.index))?;
            let take = (expected - self.left.len()).min(frames - offset);
            self.left.extend_from_slice(&chunk[offset..offset + take]);
            self.right
                .extend_from_slice(&chunk[frames + offset..frames + offset + take]);
            offset += take;
            if self.left.len() == expected {
                finished.push(self.take());
            }
        }
        Ok(finished)
    }

    fn take(&mut self) -> (usize, Vec<f32>) {
        let mut planar = Vec::with_capacity(self.left.len() * 2);
        planar.extend_from_slice(&self.left);
        planar.extend_from_slice(&self.right);
        let index = self.index;
        self.index += 1;
        self.left.clear();
        self.right.clear();
        (index, planar)
    }

    fn is_empty(&self) -> bool {
        self.left.is_empty()
    }

    fn index(&self) -> usize {
        self.index
    }
}

/// One range of a segmented render: the segment being filled and the per-block work it
/// shares with the whole-track path.
struct Emission<'a, 'b> {
    processor: &'a mut BlockProcessor<'b>,
    buffer: SegmentBuffer,
}

impl<'a, 'b> Emission<'a, 'b> {
    fn new(
        processor: &'a mut BlockProcessor<'b>,
        first_index: usize,
        frames_per_segment: usize,
        frames: usize,
    ) -> Self {
        Self {
            processor,
            buffer: SegmentBuffer::new(first_index, frames_per_segment, frames),
        }
    }

    /// Process one emitted block, publish it, and write the segment it completes.
    fn emit(
        &mut self,
        writer: &mut cache::EntryWriter,
        out: &Block,
        at: usize,
        on_audio: &mut dyn FnMut(usize, &[f32]) -> Result<()>,
    ) -> Result<()> {
        let chunk = self.processor.process(out, at)?;
        if writer.progress().gain_db.is_none() {
            // As soon as the first block fixes the gain, before any segment can land: a
            // resumed render cannot recover it from the audio, so a kill at any later
            // point has to find it in the manifest.
            writer.record_progress(&self.processor.progress())?;
        }
        on_audio(at, &chunk)?;
        let finished = self.buffer.push(&chunk)?;
        if finished.is_empty() {
            return Ok(());
        }
        for (index, planar) in &finished {
            writer.write_segment(*index, planar)?;
        }
        writer.record_progress(&self.processor.progress())?;
        Ok(())
    }

    /// A range ends on a segment boundary, so nothing may still be buffered.
    fn finish(self) -> Result<()> {
        anyhow::ensure!(
            self.buffer.is_empty(),
            "the range stopped inside segment {}",
            self.buffer.index()
        );
        Ok(())
    }
}

/// Publishes a render into the frontier one second at a time.
///
/// The frontier's slots are fixed one-second windows, so audio that does not start on a
/// second boundary is trimmed to the next one and the player reads the original where no
/// chunk exists. That is what lets a resumed render, which starts mid-track and skips the
/// segments already on disk, publish at the absolute index.
struct FrontierPublisher<'a> {
    frontier: &'a crate::frontier::Frontier,
    /// Absolute frame where `buffer` starts.
    at: usize,
    buffer: Vec<f32>,
}

impl<'a> FrontierPublisher<'a> {
    fn new(frontier: &'a crate::frontier::Frontier) -> Self {
        Self {
            frontier,
            at: 0,
            buffer: Vec::with_capacity(crate::frontier::CHUNK_FRAMES * 2),
        }
    }

    fn frames(&self) -> usize {
        self.buffer.len() / 2
    }

    fn push(&mut self, at: usize, chunk: &[f32]) -> Result<()> {
        anyhow::ensure!(
            chunk.len() % 2 == 0,
            "a stereo chunk of {} samples is not whole frames",
            chunk.len()
        );
        if at != self.at + self.frames() {
            // The stream jumped, because a segment already on disk was skipped: what is
            // buffered belongs to a stretch that ended and this one starts afresh.
            self.buffer.clear();
            self.at = at;
        }
        let frames = chunk.len() / 2;
        for frame in 0..frames {
            self.buffer.push(chunk[frame]);
            self.buffer.push(chunk[frames + frame]);
        }
        let pad = self.at % crate::frontier::CHUNK_FRAMES;
        if pad > 0 {
            let drop = (crate::frontier::CHUNK_FRAMES - pad).min(self.frames());
            self.buffer.drain(..drop * 2);
            self.at += drop;
        }
        while self.frames() >= crate::frontier::CHUNK_FRAMES {
            let chunk: Vec<f32> = self
                .buffer
                .drain(..crate::frontier::CHUNK_FRAMES * 2)
                .collect();
            self.frontier
                .publish(self.at / crate::frontier::CHUNK_FRAMES, chunk);
            self.at += crate::frontier::CHUNK_FRAMES;
        }
        Ok(())
    }

    /// Publish the last, short chunk when it ends the track; anywhere else a short chunk
    /// would move the frontier's read pointer past frames it does not hold.
    fn finish(self) {
        if !self.buffer.is_empty() && self.at + self.frames() == self.frontier.frames() {
            self.frontier
                .publish(self.at / crate::frontier::CHUNK_FRAMES, self.buffer);
        }
    }
}

/// Render the track into `writer` as segments, publishing each finished second to the
/// frontier as it lands.
pub fn render_segments_into_frontier(
    input: &dyn Frames,
    job: Job,
    writer: &mut cache::EntryWriter,
    frontier: &crate::frontier::Frontier,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(usize),
) -> Result<Streamed> {
    let mut publisher = FrontierPublisher::new(frontier);
    let streamed = render_segments(input, job, writer, cancel, progress, |at, chunk| {
        publisher.push(at, chunk)
    })?;
    publisher.finish();
    Ok(streamed)
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
            gain_db: cached.meta.gain_db.unwrap_or(0.0),
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
        source_entry: _,
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
            seed: seed_streaming(&track)?,
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
        assert_eq!(
            loaded.analysis,
            flacify_core::dsp::cutoff::analyse(loaded.track.channel(0))
                .expect("analysing the whole channel"),
            "the load path's verdict is the whole-channel one"
        );

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
        assert_eq!(
            seed_streaming(&fixed_signal()).expect("the streamed seed"),
            8_901_150_270_194_604_865,
            "the range-read seed must be the same value"
        );
    }

    #[test]
    fn given_a_track_when_analysed_a_range_at_a_time_then_the_verdict_is_the_whole_channel() {
        // Two analysis chunks and a partial third, so the carry crosses a chunk boundary.
        let track = synthetic_track(ANALYSIS_CHUNK * 2 + 12_345);
        assert_eq!(
            analyse_streaming(&track).expect("streamed"),
            analysis_of(track.channel(0)).expect("whole"),
            "a range-fed analysis must not decide differently from a whole-channel one"
        );
    }

    #[test]
    fn given_a_track_and_its_source_entry_when_the_seed_is_streamed_then_it_is_seed_for() {
        // 20 000-frame segments and more than one read range, so the hash crosses segment
        // borders and a range border: the sample order is what the value pins.
        const SEGMENT: usize = 20_000;
        let root = TempRoot::new("seed-streaming");
        let frames = SEED_RANGE + 12_345;
        let original = synthetic_track(frames);
        let source = TempOutput::new("seed-streaming.wav");
        crate::wav::write(&source.path, SAMPLE_RATE, 2, &original.planar).expect("writing the wav");
        let tier = crate::source::Tier::new(&root.path, SEGMENT).expect("a tier");
        let entry = tier
            .build_or_reuse(&source.path, "seed-streaming")
            .expect("building the source entry");

        // The track `load` holds: the entry's own samples, after the 24-bit round trip.
        let planar = entry.read_all().expect("reading the entry");
        let stored = Track {
            planar,
            frames: entry.meta().frames,
        };
        assert!(stored.frames > SEED_RANGE, "the hash needs two ranges");
        assert!(
            stored.frames > 2 * SEGMENT,
            "and the ranges must cross more than one segment"
        );

        assert_eq!(
            seed_streaming(&entry).expect("streaming the entry"),
            seed_for(&stored),
            "a streamed seed is the whole-track seed"
        );
        assert_eq!(
            seed_streaming(&stored).expect("streaming the track"),
            seed_for(&stored)
        );
    }

    /// What `enhance` and the segmented renderer get from the fake worker: it echoes the
    /// window halved, so the loudness match has a real, nonzero gain to carry.
    const FAKE_GAIN: f32 = 0.5;

    /// The cutoff the drift tests render with, recorded in the entry either way.
    const CUTOFF: u32 = 12_000;

    /// The fake worker `cargo` builds beside this test binary.
    ///
    /// `CARGO_BIN_EXE_` is only set for the package that declares a binary, and the fake
    /// lives in `flacify-infer`, so an app test looks for it above its own executable.
    /// The depth varies by runner (a plain test binary sits in `deps/`, one built through
    /// a nested directory sits deeper), so every ancestor is checked rather than one
    /// assumed parent. `cargo test --workspace --all-targets` builds it.
    fn fake_worker() -> std::path::PathBuf {
        let exe = std::env::current_exe().expect("the test binary's path");
        let name = if cfg!(windows) {
            "fake-flashsr-worker.exe"
        } else {
            "fake-flashsr-worker"
        };
        for dir in exe.ancestors().skip(1) {
            for candidate in [dir.join(name), dir.join("deps").join(name)] {
                if candidate.is_file() {
                    return candidate;
                }
            }
        }
        panic!(
            "{} was not found above {}; run `cargo test --workspace --all-targets`",
            name,
            exe.display()
        );
    }

    /// A worker session on the fake sidecar, which needs no Python and no weights.
    fn fake_session() -> Arc<WorkerSession> {
        use flacify_infer::python::{
            WorkerConfig, DEFAULT_EXCHANGE_TIMEOUT, DEFAULT_HANDSHAKE_TIMEOUT,
        };

        let config = WorkerConfig {
            program: fake_worker(),
            // The fake ignores the script and the paths `spawn` passes.
            worker_script: std::path::PathBuf::from("tools/worker.py"),
            model_dir: std::path::PathBuf::from("unused"),
            flashsr_dir: None,
            device: "fake".to_string(),
            timeout: DEFAULT_EXCHANGE_TIMEOUT,
            handshake_timeout: DEFAULT_HANDSHAKE_TIMEOUT,
        };
        WorkerSession::with_args(config, vec!["--gain".to_string(), FAKE_GAIN.to_string()])
    }

    fn fake_job(session: Arc<WorkerSession>) -> Job {
        Job {
            session,
            cutoff_hz: Some(CUTOFF),
            seed: 0x51ED_2701,
            splice_low_band: false,
        }
    }

    /// A deterministic stereo track with peaks near full scale, so the gain and the
    /// limiter both do something and no two windows are alike.
    fn synthetic_track(frames: usize) -> Track {
        let mut planar = vec![0.0; frames * 2];
        for frame in 0..frames {
            let t = frame as f32 / SAMPLE_RATE as f32;
            planar[frame] = 0.78 * (std::f32::consts::TAU * 220.0 * t).sin()
                + 0.16 * (std::f32::consts::TAU * 3_100.0 * t).sin();
            planar[frames + frame] = 0.78 * (std::f32::consts::TAU * 223.0 * t).sin()
                - 0.16 * (std::f32::consts::TAU * 4_700.0 * t).sin();
        }
        Track { planar, frames }
    }

    fn entry_fields() -> cache::Fields {
        cache::Fields {
            source_hash: "synthetic".to_string(),
            window: WINDOW,
            overlap: OVERLAP,
            model_id: "fake".to_string(),
            splice: false,
        }
    }

    fn entry_spec(frames: usize, fields: &cache::Fields) -> cache::EntrySpec {
        cache::EntrySpec {
            frames,
            cutoff_hz: CUTOFF,
            fields: fields.clone(),
        }
    }

    fn segment_frames() -> usize {
        cache::frames_per_segment(WINDOW, OVERLAP).expect("a segment length")
    }

    /// A render collected channel-major, the way the whole-track path returns one.
    fn collected(
        mut render: impl FnMut(&mut dyn FnMut(usize, &[f32]) -> Result<()>) -> Result<Streamed>,
        frames: usize,
    ) -> (Streamed, Vec<f32>) {
        let mut buffer = ChannelMajor::with_capacity(frames);
        let streamed = render(&mut |_, chunk| {
            buffer.push(chunk);
            Ok(())
        })
        .expect("rendering");
        (streamed, buffer.finish())
    }

    fn modified(path: &Path) -> std::time::SystemTime {
        std::fs::metadata(path)
            .expect("measuring a file")
            .modified()
            .expect("a modification time")
    }

    #[test]
    fn given_a_track_rendered_both_ways_when_compared_then_every_sample_the_report_and_the_gain_match(
    ) {
        // Two whole segments: the exact multiple is the end-of-track case with no tail
        // left, and the boundary between them is what the tail carry crosses.
        let frames = 2 * segment_frames();
        let track = synthetic_track(frames);
        let session = fake_session();
        let cancel = AtomicBool::new(false);

        let (whole, expected) = collected(
            |on_audio| {
                enhance_streaming(
                    &track,
                    fake_job(Arc::clone(&session)),
                    &cancel,
                    &mut |_| {},
                    on_audio,
                )
            },
            frames,
        );
        assert!(
            whole.limit.limited_samples > 0,
            "the test signal must reach the limiter, or the report proves nothing"
        );
        assert!(
            whole.gain_db.abs() > 1.0,
            "the fake worker's halved echo must need a real gain, got {}",
            whole.gain_db
        );

        let root = TempRoot::new("segmented-drift");
        let cache = crate::cache::Cache::new(root.path.clone());
        let fields = entry_fields();
        let key = fields.key();
        let mut writer = cache
            .begin(&key, &entry_spec(frames, &fields))
            .expect("beginning the entry");
        let (segmented, actual) = collected(
            |on_audio| {
                render_segments(
                    &track,
                    fake_job(Arc::clone(&session)),
                    &mut writer,
                    &cancel,
                    &mut |_| {},
                    on_audio,
                )
            },
            frames,
        );

        assert_eq!(actual.len(), expected.len());
        for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "sample {index}: {got} != {want}"
            );
        }
        assert_eq!(segmented.gain_db, whole.gain_db, "the gain carried");
        assert_eq!(segmented.limit, whole.limit, "the limiter's report carried");

        // The entry is complete and `load` is exactly the segments joined back together.
        let cached = cache
            .load(&key)
            .expect("loading")
            .expect("a complete entry");
        assert_eq!(cached.meta.state, cache::State::Complete);
        assert_eq!(cached.meta.frames, frames);
        assert_eq!(cached.meta.segments.len(), 2);
        assert_eq!(cached.meta.gain_db, Some(whole.gain_db));
        let dir = cache.entry_dir(&key);
        let mut left = Vec::new();
        let mut right = Vec::new();
        for segment in &cached.meta.segments {
            let path = dir.join(format!("seg-{:06}.flac", segment.index));
            let decoded = decode::decode(&path).expect("decoding a segment");
            assert_eq!(decoded.frames, segment.frames);
            left.extend_from_slice(&decoded.planar[..decoded.frames]);
            right.extend_from_slice(&decoded.planar[decoded.frames..]);
        }
        let mut concatenated = left;
        concatenated.extend_from_slice(&right);
        assert_eq!(concatenated.len(), cached.planar.len());
        for (index, (&joined, &loaded)) in concatenated.iter().zip(&cached.planar).enumerate() {
            assert_eq!(
                joined.to_bits(),
                loaded.to_bits(),
                "sample {index}: load is not the concatenation of its segments"
            );
        }
    }

    #[test]
    fn given_a_render_cancelled_in_its_first_segment_when_the_entry_is_inspected_then_the_gain_is_recorded_and_no_segment_is(
    ) {
        let frames = segment_frames();
        let track = synthetic_track(frames);
        let root = TempRoot::new("cancel-first");
        let cache = crate::cache::Cache::new(root.path.clone());
        let fields = entry_fields();
        let key = fields.key();
        let mut writer = cache
            .begin(&key, &entry_spec(frames, &fields))
            .expect("beginning the entry");
        let cancel = AtomicBool::new(false);

        let error = render_segments(
            &track,
            fake_job(fake_session()),
            &mut writer,
            &cancel,
            &mut |_| {},
            |at, _| {
                if at == 0 {
                    cancel.store(true, Ordering::Relaxed);
                }
                Ok(())
            },
        )
        .expect_err("the render was cancelled in its first segment");
        assert!(is_cancelled(&error), "{error:#}");
        assert!(
            writer.progress().gain_db.is_some(),
            "the gain has to be on disk before a segment can be resumed"
        );
        assert!(!writer.has_segment(0), "a cancelled block is not a segment");
        assert!(cache.load(&key).expect("no error").is_none());
    }

    #[test]
    fn given_a_partial_entry_with_segment_zero_when_resumed_then_segment_zero_is_kept_and_the_audio_is_the_fresh_render(
    ) {
        let frames = 2 * segment_frames();
        let track = synthetic_track(frames);
        let session = fake_session();
        let fields = entry_fields();
        let key = fields.key();

        // A fresh render is the oracle the resumed one has to reproduce.
        let fresh_root = TempRoot::new("resume-fresh");
        let fresh_cache = crate::cache::Cache::new(fresh_root.path.clone());
        let mut writer = fresh_cache
            .begin(&key, &entry_spec(frames, &fields))
            .expect("beginning the fresh entry");
        render_segments(
            &track,
            fake_job(Arc::clone(&session)),
            &mut writer,
            &AtomicBool::new(false),
            &mut |_| {},
            |_, _| Ok(()),
        )
        .expect("the fresh render");
        let fresh = fresh_cache
            .load(&key)
            .expect("loading")
            .expect("a complete entry");

        // Kill the render on the block that completes segment 0, so the cancel lands
        // between ranges: the segment is written, the segment after it is not started.
        let root = TempRoot::new("resume-partial");
        let cache = crate::cache::Cache::new(root.path.clone());
        let mut writer = cache
            .begin(&key, &entry_spec(frames, &fields))
            .expect("beginning the interrupted entry");
        let cancel = AtomicBool::new(false);
        let error = render_segments(
            &track,
            fake_job(Arc::clone(&session)),
            &mut writer,
            &cancel,
            &mut |_| {},
            |at, _| {
                if at + CHUNK >= segment_frames() {
                    cancel.store(true, Ordering::Relaxed);
                }
                Ok(())
            },
        )
        .expect_err("the render was cancelled after its first segment");
        assert!(is_cancelled(&error), "{error:#}");

        let dir = cache.entry_dir(&key);
        assert!(
            cache.load(&key).expect("no error").is_none(),
            "a partial entry is a miss"
        );
        assert!(
            !dir.join("seg-000001.flac").exists(),
            "the cancelled render wrote a segment it had not finished"
        );
        let recorded = dir.join("seg-000000.flac");
        assert_eq!(
            decode::decode(&recorded)
                .expect("decoding the kept segment")
                .frames,
            segment_frames(),
            "the segment the kill left behind loads on its own"
        );
        let was_modified = modified(&recorded);

        let mut writer = cache
            .begin(&key, &entry_spec(frames, &fields))
            .expect("resuming the entry");
        render_segments(
            &track,
            fake_job(session),
            &mut writer,
            &AtomicBool::new(false),
            &mut |_| {},
            |_, _| Ok(()),
        )
        .expect("the resumed render");

        assert_eq!(
            modified(&recorded),
            was_modified,
            "a resumed render rewrote the segment already on disk"
        );
        let resumed = cache
            .load(&key)
            .expect("loading")
            .expect("a complete entry");
        assert_eq!(resumed.meta.state, cache::State::Complete);
        assert_eq!(resumed.meta.segments.len(), 2);
        assert_eq!(
            std::fs::read(&recorded).expect("reading the kept segment"),
            std::fs::read(fresh_cache.entry_dir(&key).join("seg-000000.flac"))
                .expect("reading the fresh segment"),
            "segment 0 must be the one the first render wrote"
        );
        assert_eq!(resumed.planar.len(), fresh.planar.len());
        for (index, (&got, &want)) in resumed.planar.iter().zip(&fresh.planar).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "sample {index}: a resumed render is not the fresh one"
            );
        }
    }

    #[test]
    fn given_a_track_shorter_than_a_segment_when_rendered_as_segments_then_it_finishes_and_matches_the_whole_render(
    ) {
        // Three steps and a bit: one short segment, and the end-of-track window is almost
        // a whole window long.
        let frames = 3 * CHUNK + 100;
        assert!(frames < segment_frames());
        let track = synthetic_track(frames);
        let session = fake_session();
        let cancel = AtomicBool::new(false);

        let (whole, expected) = collected(
            |on_audio| {
                enhance_streaming(
                    &track,
                    fake_job(Arc::clone(&session)),
                    &cancel,
                    &mut |_| {},
                    on_audio,
                )
            },
            frames,
        );

        let root = TempRoot::new("one-segment");
        let cache = crate::cache::Cache::new(root.path.clone());
        let fields = entry_fields();
        let key = fields.key();
        let mut writer = cache
            .begin(&key, &entry_spec(frames, &fields))
            .expect("beginning the entry");
        let (segmented, actual) = collected(
            |on_audio| {
                render_segments(
                    &track,
                    fake_job(Arc::clone(&session)),
                    &mut writer,
                    &cancel,
                    &mut |_| {},
                    on_audio,
                )
            },
            frames,
        );

        assert_eq!(segmented.gain_db, whole.gain_db);
        assert_eq!(segmented.limit, whole.limit);
        assert_eq!(actual.len(), expected.len());
        for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "sample {index}: {got} != {want}"
            );
        }
        let cached = cache
            .load(&key)
            .expect("loading")
            .expect("a complete entry");
        assert_eq!(cached.meta.segments.len(), 1);
        assert_eq!(cached.meta.segments[0].frames, frames);
        assert_eq!(cached.meta.state, cache::State::Complete);
    }

    #[test]
    fn given_a_track_that_ends_inside_a_segment_when_rendered_as_segments_then_both_segments_match_the_whole_render(
    ) {
        // Seven steps and a sliver: the last segment is shorter than the lookahead a
        // mid-track range needs, so the range before it cannot be ended early and the two
        // segments render as the track's end, its last window short.
        let frames = segment_frames() + 100;
        let track = synthetic_track(frames);
        let session = fake_session();
        let cancel = AtomicBool::new(false);

        let (whole, expected) = collected(
            |on_audio| {
                enhance_streaming(
                    &track,
                    fake_job(Arc::clone(&session)),
                    &cancel,
                    &mut |_| {},
                    on_audio,
                )
            },
            frames,
        );

        let root = TempRoot::new("short-tail");
        let cache = crate::cache::Cache::new(root.path.clone());
        let fields = entry_fields();
        let key = fields.key();
        let mut writer = cache
            .begin(&key, &entry_spec(frames, &fields))
            .expect("beginning the entry");
        let (segmented, actual) = collected(
            |on_audio| {
                render_segments(
                    &track,
                    fake_job(Arc::clone(&session)),
                    &mut writer,
                    &cancel,
                    &mut |_| {},
                    on_audio,
                )
            },
            frames,
        );

        assert_eq!(segmented.gain_db, whole.gain_db);
        assert_eq!(segmented.limit, whole.limit);
        assert_eq!(actual.len(), expected.len());
        for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "sample {index}: {got} != {want}"
            );
        }

        let cached = cache
            .load(&key)
            .expect("loading")
            .expect("a complete entry");
        assert_eq!(cached.meta.segments.len(), 2);
        assert_eq!(cached.meta.segments[0].frames, segment_frames());
        assert_eq!(cached.meta.segments[1].frames, 100);
    }

    #[test]
    fn given_an_entry_that_records_segment_zero_without_the_gain_when_rendered_then_it_is_refused()
    {
        let frames = segment_frames();
        let track = synthetic_track(frames);
        let root = TempRoot::new("no-gain");
        let cache = crate::cache::Cache::new(root.path.clone());
        let fields = entry_fields();
        let key = fields.key();
        let mut writer = cache
            .begin(&key, &entry_spec(frames, &fields))
            .expect("beginning the entry");
        // What a build from before the gain was persisted would leave behind.
        let planar = track.read_frames(0, frames).expect("the segment's samples");
        writer.write_segment(0, &planar).expect("writing segment 0");
        writer.abandon();

        let mut writer = cache
            .begin(&key, &entry_spec(frames, &fields))
            .expect("resuming the entry");
        let error = render_segments(
            &track,
            fake_job(fake_session()),
            &mut writer,
            &AtomicBool::new(false),
            &mut |_| {},
            |_, _| Ok(()),
        )
        .expect_err("an entry with a segment and no gain cannot be resumed");
        assert!(
            format!("{error:#}").contains("without the loudness gain"),
            "{error:#}"
        );
    }

    /// Planar stereo whose samples are their own absolute frame index, so a frontier test
    /// can say exactly which frames it reads.
    fn ramped_planar(start: usize, frames: usize) -> Vec<f32> {
        let mut planar = Vec::with_capacity(frames * 2);
        for frame in start..start + frames {
            planar.push(frame as f32);
        }
        for frame in start..start + frames {
            planar.push(frame as f32);
        }
        planar
    }

    #[test]
    fn given_a_stream_that_starts_inside_a_second_when_published_then_it_lands_on_the_frontier_grid(
    ) {
        use crate::frontier::{Frontier, CHUNK_FRAMES};

        let frontier = Frontier::new(64 * CHUNK_FRAMES);
        let mut from_zero = FrontierPublisher::new(&frontier);
        from_zero
            .push(0, &ramped_planar(0, CHUNK_FRAMES))
            .expect("publishing the first second");
        assert_eq!(frontier.frame(0), Some((0.0, 0.0)));

        // A resumed render starts 8 000 frames into a second: the partial second has no
        // slot, so it is dropped and the next whole second is the first to land.
        let start = 1_400_000;
        let mut resumed = FrontierPublisher::new(&frontier);
        resumed
            .push(start, &ramped_planar(start, CHUNK_FRAMES))
            .expect("publishing the partial second");
        assert!(
            frontier.frame(29 * CHUNK_FRAMES).is_none(),
            "a partial second must not move the read pointer"
        );
        let second = start + CHUNK_FRAMES;
        resumed
            .push(second, &ramped_planar(second, CHUNK_FRAMES))
            .expect("publishing the next second");
        assert_eq!(
            frontier.frame(30 * CHUNK_FRAMES),
            Some((1_440_000.0, 1_440_000.0)),
            "the first whole second after the jump goes to its own index"
        );
        assert!(
            frontier.frame(28 * CHUNK_FRAMES).is_none(),
            "no second before the jump was filled in"
        );

        // What is left is not a whole second and does not end the track.
        resumed.finish();
        assert!(frontier.frame(31 * CHUNK_FRAMES).is_none());
    }
}
