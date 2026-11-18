//! Live playback: a feed goes in, the A/B mix comes out, and memory stays bounded.
//!
//! One pump thread reads a [`Feed`] and pushes the dry side into a ring holding
//! `buffer_frames + window_frames`. When it has pushed a block it asks the [`WetProducer`]
//! for the wet side of the frames behind that block, which is where the model needs a
//! window of lookahead; the wet side goes into a ring holding `buffer_frames`. The pump
//! waits for room instead of dropping audio, so resident audio is
//! `(2 × buffer + window + block) × channels × 4` bytes for the whole stream.
//!
//! [`Live::fill`] is the consumer, on the audio thread: it reads both rings at the
//! playhead, mixes wet over dry through the mixer's ramp, plays the dry side wherever the
//! wet side has not caught up, and counts those frames. No allocation, no lock, no
//! blocking, no logging.
//!
//! One producer call renders a window and hands over one block of it, and the pump checks
//! the wet ring's room after that call rather than before it, so the compute overlaps the
//! playhead's drain. The dry ring therefore carries one block over the alignment delay:
//! the block being rendered is already in it when the playhead reaches it.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use flacify_core::dsp::resample::{ChunkedResampler, Resampler};
use flacify_core::policy::LivePolicy;
use flacify_core::{Block, Pipeline, Upscaler, SAMPLE_RATE};

use crate::decode::{self, Chunk};
use crate::player::{mix_stretch, Ramp};

/// Why the A/B switch is unavailable while the buffer is behind, in the words the status
/// line prints.
pub const STARVED_REASON: &str = "the live buffer has not caught up with the playhead";

/// How long the pump waits for the buffer to fill before the device is allowed to start.
pub const PREBUFFER_TIMEOUT: Duration = Duration::from_secs(120);

/// Live audio joins no cache and is never reproduced (D5), so the seed is a constant;
/// every window still draws its own from the pipeline's lattice.
pub const SEED: u64 = 0;

/// The live path is 48 kHz stereo end to end, like the model.
const CHANNELS: usize = 2;

/// The overlap as a fraction of the model window. One fifth is the trained window's own
/// 1 s in 5.12 s, so every preset crossfades over the same share of its context and the
/// compute spent per played second stays flat across the four (D19).
const OVERLAP_DIVISOR: usize = 5;

/// Blocks the pump must have timed before the measured ratio is worth printing.
const MIN_TIMED_BLOCKS: usize = 3;
/// Timings kept for the rolling ratio: at least a second of audio at every preset.
const TIMED_BLOCKS: usize = 8;

/// The longest the pump sleeps before checking whether the mixer has gone away.
const MAX_WAIT: Duration = Duration::from_millis(50);
/// The shortest, so a full ring does not spin.
const MIN_WAIT: Duration = Duration::from_millis(1);

/// Where a live stream's audio comes from: the decoder for a file, the capture stream for
/// the sink. Implement this and [`Live::start`] plays it.
pub trait Feed: Send {
    /// Interleaved channels the feed writes.
    fn channels(&self) -> usize;

    /// Rate the feed writes at. The live path only runs at [`SAMPLE_RATE`].
    fn sample_rate(&self) -> u32;

    /// Block until at least one frame is available. Returns interleaved frames written, 0 at end.
    fn read(&mut self, out: &mut [f32]) -> anyhow::Result<usize>;
}

/// Turns the dry side of one block into its wet side.
///
/// `dry` holds the block followed by the window of lookahead the model needs after it;
/// `wet` is exactly the block's frames, interleaved like `dry`. Filling `wet` completely is
/// the contract: a producer that cannot keep up slows the pump, which is what the credit
/// wait is for.
pub type WetProducer = Box<dyn FnMut(&[f32], &mut [f32]) -> Result<()> + Send>;

/// What one fill read from the rings.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fill {
    /// Frames written into the buffer.
    pub frames: usize,
    /// Of those, frames that played the dry side because the wet side did not cover them.
    pub starved_frames: usize,
}

/// The counters the live path publishes. The status line and the panel read them without
/// a lock; the audio thread writes the starvation counters and the producer thread the
/// timings.
#[derive(Debug, Default)]
pub struct Stats {
    starved_frames: AtomicUsize,
    dropouts: AtomicUsize,
    is_wet_covered: AtomicBool,
    is_starving: AtomicBool,
    blocks: AtomicUsize,
    /// Nanoseconds of compute per block, and the frames that block carried, oldest first
    /// in the ring.
    block_nanos: [AtomicU64; TIMED_BLOCKS],
    carried_frames: [AtomicU64; TIMED_BLOCKS],
}

impl Stats {
    /// Frames that played the dry side because the wet side had not caught up.
    pub fn starved_frames(&self) -> usize {
        self.starved_frames.load(Ordering::Relaxed)
    }

    /// Starved stretches, not frames: one increment per dropout.
    pub fn dropouts(&self) -> usize {
        self.dropouts.load(Ordering::Relaxed)
    }

    /// Whether the wet side covers the next callback.
    pub fn is_wet_covered(&self) -> bool {
        self.is_wet_covered.load(Ordering::Relaxed)
    }

    /// The mixer publishes the coverage it is about to play from.
    pub(crate) fn set_wet_covered(&self, is_covered: bool) {
        self.is_wet_covered.store(is_covered, Ordering::Relaxed);
    }

    /// One fill's starvation, counted as frames and as at most one dropout.
    fn record(&self, starved_frames: usize, is_full: bool) {
        self.starved_frames
            .fetch_add(starved_frames, Ordering::Relaxed);
        if starved_frames > 0 {
            if !self.is_starving.swap(true, Ordering::Relaxed) {
                self.dropouts.fetch_add(1, Ordering::Relaxed);
            }
        } else if is_full {
            self.is_starving.store(false, Ordering::Relaxed);
        }
    }

    /// One block's compute time and how much audio it carried, published as it finishes.
    fn record_block(&self, elapsed: Duration, frames: usize) {
        let slot = self.blocks.fetch_add(1, Ordering::Relaxed) % TIMED_BLOCKS;
        let nanos = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        self.block_nanos[slot].store(nanos, Ordering::Relaxed);
        self.carried_frames[slot].store(frames as u64, Ordering::Relaxed);
    }

    /// Blocks the pump has produced, which is how many timings exist.
    pub fn blocks_rendered(&self) -> usize {
        self.blocks.load(Ordering::Relaxed)
    }

    /// Seconds of compute per second of audio the path played, over the last blocks, or
    /// `None` until [`MIN_TIMED_BLOCKS`] have finished.
    ///
    /// Per played second, not per second the model saw: a call renders a window and hands
    /// over one block of it, so this is the ratio a listener's clock sees and the one that
    /// has to stay under 1.
    pub fn measured_rtf(&self) -> Option<f64> {
        let blocks = self.blocks_rendered();
        if blocks < MIN_TIMED_BLOCKS {
            return None;
        }
        let kept = blocks.min(TIMED_BLOCKS);
        let nanos: u128 = (0..kept)
            .map(|slot| u128::from(self.block_nanos[slot].load(Ordering::Relaxed)))
            .sum();
        let frames: u128 = (0..kept)
            .map(|slot| u128::from(self.carried_frames[slot].load(Ordering::Relaxed)))
            .sum();
        let audio_secs = frames as f64 / f64::from(SAMPLE_RATE);
        Some(nanos as f64 / 1e9 / audio_secs)
    }
}

/// One bounded ring of interleaved frames, addressed by absolute frame number.
///
/// Samples are `AtomicU32` so the producer writes and the consumer reads the same buffer
/// without a lock or `unsafe`; the frame index carries the ordering that makes it sound.
struct Ring {
    samples: Vec<AtomicU32>,
    /// Frames the ring holds.
    capacity: usize,
    channels: usize,
    /// Frames written so far, absolute. The producer is the only writer, and each write is
    /// published with `Release`.
    written: AtomicUsize,
}

impl Ring {
    fn new(capacity: usize, channels: usize) -> Self {
        Self {
            samples: (0..capacity * channels)
                .map(|_| AtomicU32::new(0))
                .collect(),
            capacity,
            channels,
            written: AtomicUsize::new(0),
        }
    }

    fn bytes(&self) -> usize {
        self.capacity * self.channels * std::mem::size_of::<f32>()
    }

    fn written(&self) -> usize {
        self.written.load(Ordering::Acquire)
    }

    /// Frames readable from `frame`: 0 once the playhead has left what the ring still holds.
    fn available(&self, frame: usize) -> usize {
        let written = self.written();
        if frame < written.saturating_sub(self.capacity) {
            return 0;
        }
        written.saturating_sub(frame)
    }

    /// One sample of a frame [`Ring::available`] reported. That load is what makes the
    /// sample visible here; this one is relaxed for the same reason.
    fn sample(&self, frame: usize, channel: usize) -> f32 {
        let index = (frame % self.capacity) * self.channels + channel;
        f32::from_bits(self.samples[index].load(Ordering::Relaxed))
    }

    /// Write `frames` frames starting at absolute frame `frame`, then publish them.
    fn write(&self, frame: usize, samples: &[f32]) {
        debug_assert_eq!(samples.len() % self.channels, 0);
        for offset in 0..samples.len() / self.channels {
            let slot = ((frame + offset) % self.capacity) * self.channels;
            for channel in 0..self.channels {
                let sample = samples[offset * self.channels + channel];
                self.samples[slot + channel].store(sample.to_bits(), Ordering::Relaxed);
            }
        }
        let frames = samples.len() / self.channels;
        self.written.store(frame + frames, Ordering::Release);
    }

    /// Wait until `end_frame` is within what the ring can hold.
    ///
    /// The playhead is the only thing that frees room, so the wait is as long as the audio
    /// in between takes to play. The pump never drops a frame to make room.
    fn wait_for_room(
        &self,
        end_frame: usize,
        playhead: &AtomicUsize,
        is_stopping: &AtomicBool,
    ) -> Room {
        loop {
            if is_stopping.load(Ordering::Acquire) {
                return Room::Stopped;
            }
            let ahead = end_frame.saturating_sub(playhead.load(Ordering::Acquire));
            if ahead <= self.capacity {
                return Room::Ready;
            }
            thread::sleep(wait_for(ahead - self.capacity));
        }
    }
}

/// Whether a wait for room ended with room to write or with the mixer going away.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Room {
    Ready,
    Stopped,
}

/// How long `frames` take to play, kept short so a stopped playhead is noticed at once.
fn wait_for(frames: usize) -> Duration {
    let secs = frames as f64 / f64::from(SAMPLE_RATE);
    Duration::from_secs_f64(secs.clamp(MIN_WAIT.as_secs_f64(), MAX_WAIT.as_secs_f64()))
}

/// The two rings and the state the pump and the playhead share.
struct Rings {
    dry: Ring,
    wet: Ring,
    channels: usize,
    window_frames: usize,
    block_frames: usize,
    /// The mixer's position, which is the only thing that frees room.
    playhead: Arc<AtomicUsize>,
    /// Set once the feed has ended, so a drained ring is the end of the stream.
    is_feed_finished: AtomicBool,
    /// Set when the last `Live` drops; the pump returns at its next check.
    is_stopping: AtomicBool,
    /// What the panel and the status line read, including the pump's own timings.
    stats: Arc<Stats>,
}

impl Rings {
    /// `block_frames` is what one [`WetProducer`] call produces, and the pump keeps the
    /// rings sized from the policy so the credit can never be exceeded.
    fn new(
        buffer_frames: usize,
        window_frames: usize,
        block_frames: usize,
        channels: usize,
        stats: Arc<Stats>,
    ) -> Self {
        Self {
            // The alignment delay, plus the block the pump is rendering: while it computes
            // the playhead keeps draining the dry side, and that block has to be readable
            // when the render ends or the pump would wait for the playhead instead.
            dry: Ring::new(buffer_frames + window_frames + block_frames, channels),
            wet: Ring::new(buffer_frames, channels),
            channels,
            window_frames,
            block_frames,
            playhead: Arc::new(AtomicUsize::new(0)),
            is_feed_finished: AtomicBool::new(false),
            is_stopping: AtomicBool::new(false),
            stats,
        }
    }

    fn is_finished(&self) -> bool {
        self.is_feed_finished.load(Ordering::Acquire)
    }

    fn stop(&self) {
        self.is_stopping.store(true, Ordering::Release);
    }
}

/// A live stream being played: the rings, and the counters the UI reads.
pub struct Live {
    rings: Arc<Rings>,
}

impl Live {
    /// Start reading `feed` and producing its wet side.
    ///
    /// Returns the consumer half and the pump. Both rings are allocated here and never
    /// grow; `policy` decides how deep they are. The wet ring is `buffer_frames`, the dry
    /// one carries the window of alignment delay on top.
    pub fn start(
        feed: Box<dyn Feed>,
        policy: &LivePolicy,
        produce: WetProducer,
    ) -> Result<(Arc<Self>, Pump)> {
        let channels = feed.channels();
        anyhow::ensure!(channels > 0, "a live feed needs at least one channel");
        anyhow::ensure!(
            feed.sample_rate() == SAMPLE_RATE,
            "the live path runs at {SAMPLE_RATE} Hz; this feed is {} Hz",
            feed.sample_rate()
        );
        anyhow::ensure!(
            policy.window_frames > 0 && policy.buffer_frames > 0,
            "a live stream needs a window and a buffer: {} frames and {} frames",
            policy.window_frames,
            policy.buffer_frames
        );
        let rings = Arc::new(Rings::new(
            policy.buffer_frames,
            policy.window_frames,
            block_frames(policy),
            channels,
            Arc::new(Stats::default()),
        ));
        let live = Arc::new(Self {
            rings: Arc::clone(&rings),
        });
        let pump = Pump::spawn(rings, feed, produce)?;
        Ok((live, pump))
    }

    /// The playhead the pump measures its credit against. The mixer's `position` is this
    /// same atomic, so a wait and a fill agree on where playback is.
    pub fn playhead(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.rings.playhead)
    }

    /// Interleaved channels the feed writes, which is what the device must take.
    pub fn channels(&self) -> usize {
        self.rings.channels
    }

    /// The counters the panel and the status line print.
    pub fn stats(&self) -> Arc<Stats> {
        Arc::clone(&self.rings.stats)
    }

    /// Bytes the rings hold, fixed for the life of the stream. The pump stages one block
    /// plus its lookahead on top of this.
    pub fn resident_bytes(&self) -> usize {
        self.rings.dry.bytes() + self.rings.wet.bytes()
    }

    /// Frames of wet audio that exist at or beyond `frame`.
    pub fn wet_ahead(&self, frame: usize) -> usize {
        self.rings.wet.available(frame)
    }

    /// Frames the rings hold at or beyond `frame`, both sides.
    pub fn occupied(&self, frame: usize) -> usize {
        self.rings.dry.available(frame) + self.rings.wet.available(frame)
    }

    /// Whether the wet side covers `frames` frames starting at `frame`.
    pub fn is_wet_covered(&self, frame: usize, frames: usize) -> bool {
        self.rings.wet.available(frame) >= frames.max(1)
    }

    /// Whether the stream has ended and the rings are drained at `frame`.
    pub fn is_finished(&self, frame: usize) -> bool {
        self.rings.is_finished() && self.rings.dry.available(frame) == 0
    }

    /// Block until `frames` frames of wet audio exist at `frame`, or the stream ends.
    ///
    /// For the prebuffer before playback starts; the callback never waits.
    pub fn wait_for_wet(&self, frame: usize, frames: usize, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            if self.is_wet_covered(frame, frames) {
                return true;
            }
            if self.is_finished(frame) || Instant::now() >= deadline {
                return false;
            }
            thread::sleep(MIN_WAIT);
        }
    }

    /// Fill `out` (interleaved) from the rings at `position`, wet over dry through the A/B
    /// ramp.
    ///
    /// Called from the audio callback: no allocation, no lock, no blocking, no logging.
    pub fn fill(&self, ramp: &mut Ramp, target: f32, position: usize, out: &mut [f32]) -> Fill {
        let channels = self.rings.channels;
        let frames = out.len() / channels;
        let dry_frames = self.rings.dry.available(position).min(frames);
        if dry_frames == 0 {
            return Fill::default();
        }
        let wet_frames = self.rings.wet.available(position).min(dry_frames);
        let dry = |frame: usize, channel: usize| self.rings.dry.sample(frame, channel);
        mix_stretch(
            ramp,
            out,
            0,
            position,
            wet_frames,
            channels,
            target,
            dry,
            |frame, channel| self.rings.wet.sample(frame, channel),
        );
        // Everything past the wet side plays dry, which is the passthrough the buffer is
        // there for, and is counted as a dropout.
        mix_stretch(
            ramp,
            out,
            wet_frames,
            position + wet_frames,
            dry_frames - wet_frames,
            channels,
            target,
            dry,
            dry,
        );
        let fill = Fill {
            frames: dry_frames,
            starved_frames: dry_frames - wet_frames,
        };
        self.rings
            .stats
            .record(fill.starved_frames, dry_frames == frames);
        // The pump measures its credit against this number, so the frames this fill
        // consumed are published as it goes.
        if fill.frames > 0 {
            self.rings
                .playhead
                .store(position + fill.frames, Ordering::Release);
        }
        fill
    }
}

impl Drop for Live {
    fn drop(&mut self) {
        self.rings.stop();
    }
}

/// The block one [`WetProducer`] call produces, which is also the pipeline's step.
///
/// A call renders a whole window and hands back `window - overlap` frames of it: the rest
/// is the crossfade the next window lands in. The step is the pump's block, so one call
/// emits exactly one block and the credit is measured in the same frames the pipeline
/// advances by.
pub fn block_frames(policy: &LivePolicy) -> usize {
    let window = policy.window_frames.max(2);
    let overlap = overlap_frames(window);
    (window - overlap).max(1)
}

/// The crossfade the live path uses: one fifth of the window, inside the pipeline's clamp.
pub fn overlap_frames(window: usize) -> usize {
    let half = window / 2;
    (window / OVERLAP_DIVISOR).min(half.saturating_sub(1))
}

/// Frames of wet audio the pump can have ready before the playhead moves.
///
/// The ring holds whole blocks, so this is the largest whole number of blocks that fits
/// the policy's buffer. It has to cover the first call's own duration: a call renders a
/// window before it can hand over its block, and the device starting mid-call is the one
/// dropout a careful start can prevent.
pub fn prebuffer_frames(policy: &LivePolicy) -> usize {
    let block = block_frames(policy);
    (policy.buffer_frames / block).max(1) * block
}

/// The producer thread. Dropping it detaches; [`Pump::join`] waits for the feed to end.
pub struct Pump {
    handle: Option<JoinHandle<Result<()>>>,
}

impl Pump {
    fn spawn(rings: Arc<Rings>, feed: Box<dyn Feed>, produce: WetProducer) -> Result<Self> {
        let handle = thread::Builder::new()
            .name("flacify-live".to_string())
            .spawn(move || {
                let result = run(&rings, feed, produce);
                // Either way the stream is over, so a drained ring is the end of playback.
                rings.is_feed_finished.store(true, Ordering::Release);
                result
            })
            .context("starting the live feed")?;
        Ok(Self {
            handle: Some(handle),
        })
    }

    /// Whether the pump has returned, whether or not its result was an error.
    pub fn is_finished(&self) -> bool {
        self.handle.as_ref().map_or(true, JoinHandle::is_finished)
    }

    /// Wait for the feed to end and report why the pump stopped.
    pub fn join(mut self) -> Result<()> {
        let Some(handle) = self.handle.take() else {
            return Ok(());
        };
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("the live feed thread panicked"))?
    }
}

/// The pump: read the feed, push dry, produce wet for the block that now has its lookahead.
fn run(rings: &Rings, mut feed: Box<dyn Feed>, mut produce: WetProducer) -> Result<()> {
    let channels = rings.channels;
    let block = rings.block_frames;
    let window = rings.window_frames;
    let slice = block + window;

    // The slice holds one block and the lookahead after it. Frames past `filled` are zero,
    // which is the padding the last slice of a stream gets.
    let mut stage = vec![0.0f32; slice * channels];
    let mut wet = vec![0.0f32; block * channels];
    let mut filled = 0usize;
    // Absolute frame of the slice's first frame, and of what each ring has published.
    let mut stage_start = 0usize;
    let mut pushed = 0usize;
    let mut produced = 0usize;
    let mut is_eof = false;

    loop {
        while !is_eof && filled < slice {
            let start = filled * channels;
            let read = feed.read(&mut stage[start..])?;
            anyhow::ensure!(
                read <= slice - filled,
                "the feed reported {read} frames into a {} frame buffer",
                slice - filled
            );
            if read == 0 {
                is_eof = true;
                break;
            }
            filled += read;
        }
        if filled == 0 {
            return Ok(());
        }

        let new_frames = stage_start + filled - pushed;
        if new_frames > 0 {
            if rings
                .dry
                .wait_for_room(pushed + new_frames, &rings.playhead, &rings.is_stopping)
                == Room::Stopped
            {
                return Ok(());
            }
            let from = (pushed - stage_start) * channels;
            rings
                .dry
                .write(pushed, &stage[from..from + new_frames * channels]);
            pushed += new_frames;
        }

        // At the end of the stream the slice shrinks, and its lookahead is the zero padding
        // so the last frames still get a wet side.
        let frames = if is_eof { filled.min(block) } else { block };
        // The room is checked after the render, not before: a call takes as long as its
        // window costs, and the playhead keeps draining through it. Waiting first would
        // make the compute and the drain take turns instead of overlapping.
        let started = Instant::now();
        produce(&stage[..filled * channels], &mut wet[..frames * channels])?;
        rings.stats.record_block(started.elapsed(), frames);
        if rings
            .wet
            .wait_for_room(produced + frames, &rings.playhead, &rings.is_stopping)
            == Room::Stopped
        {
            return Ok(());
        }
        rings.wet.write(produced, &wet[..frames * channels]);
        produced += frames;

        // Slide the stage forward: what was lookahead becomes the start of the next slice.
        stage.copy_within(frames * channels..filled * channels, 0);
        filled -= frames;
        stage_start += frames;
        stage[filled * channels..].fill(0.0);
    }
}

/// Renders the wet side of a live stream one block at a time from the dry slice the pump
/// hands over.
///
/// The pipeline is the file path's, at the preset's window: the engine preprocesses and
/// calls the model per window, and the overlap-add tail crossfades each window's last
/// frames into the next. `overlap` is `window - block`, so one push of one block emits
/// exactly one block and the pump's credit is also the pipeline's step.
pub struct Renderer<U: Upscaler> {
    pipeline: Pipeline<U>,
    /// Wet frames emitted but not yet returned, interleaved.
    pending: Vec<f32>,
    /// Frames of `pending` already returned.
    pending_start: usize,
    /// Absolute frame of the next dry push, and of the next wet frame returned.
    pushed: usize,
    produced: usize,
    is_finished: bool,
}

impl<U: Upscaler> Renderer<U> {
    /// A renderer for `window_frames` whose output arrives `block_frames` at a time.
    ///
    /// `block_frames` must be the pump's, from the same policy: a block that is not the
    /// pipeline's step would leave wet audio queued or short.
    pub fn new(upscaler: U, window_frames: usize, block_frames: usize, seed: u64) -> Result<Self> {
        anyhow::ensure!(
            block_frames > 0 && block_frames <= window_frames,
            "a live renderer needs a window and a block inside it: {window_frames} and \
             {block_frames} frames"
        );
        let overlap = window_frames - block_frames;
        let pipeline = Pipeline::new(upscaler, CHANNELS, window_frames, overlap, seed)?;
        anyhow::ensure!(
            pipeline.step() == block_frames,
            "the pipeline steps {} frames but the pump's block is {block_frames}",
            pipeline.step()
        );
        Ok(Self {
            pipeline,
            pending: Vec::with_capacity(4 * block_frames * CHANNELS),
            pending_start: 0,
            pushed: 0,
            produced: 0,
            is_finished: false,
        })
    }

    /// Fill `wet` with the block of generated audio that lines up with `dry`.
    ///
    /// `dry` starts at the frame the last call ended on and carries the lookahead after the
    /// block, exactly as [`WetProducer`] documents. Filling `wet` completely is the
    /// contract; a slice the feed has ended inside is resolved against the pipeline's tail.
    pub fn produce(&mut self, dry: &[f32], wet: &mut [f32]) -> Result<()> {
        anyhow::ensure!(
            wet.len() % CHANNELS == 0 && !wet.is_empty(),
            "a live block is a whole number of stereo frames, not {} samples",
            wet.len()
        );
        let frames = wet.len() / CHANNELS;
        let overlap = self.pipeline.overlap();
        let available_end = self.produced + dry.len() / CHANNELS;
        let target_end = (self.produced + frames + overlap).min(available_end);
        if target_end > self.pushed {
            anyhow::ensure!(
                !self.is_finished,
                "the live renderer was handed more audio after its stream ended"
            );
            let start = (self.pushed - self.produced) * CHANNELS;
            let end = (target_end - self.produced) * CHANNELS;
            self.push(&dry[start..end])?;
            self.pushed = target_end;
        }
        if self.pending_frames() < frames {
            self.finish()?;
        }
        let available = self.pending_frames();
        anyhow::ensure!(
            available >= frames,
            "the live renderer has {available} frames of wet audio for a {frames} frame block"
        );
        let start = self.pending_start * CHANNELS;
        wet.copy_from_slice(&self.pending[start..start + frames * CHANNELS]);
        self.pending_start += frames;
        if self.pending_start * CHANNELS == self.pending.len() {
            self.pending.clear();
            self.pending_start = 0;
        }
        self.produced += frames;
        Ok(())
    }

    /// Push one block of dry audio through the pipeline and queue what it emitted.
    fn push(&mut self, dry_block: &[f32]) -> Result<()> {
        let block = Block::from_interleaved(CHANNELS, dry_block)?;
        let emitted = self.pipeline.push(&block)?;
        self.queue(emitted);
        Ok(())
    }

    /// End the stream: the pipeline resolves its held tail against silence.
    fn finish(&mut self) -> Result<()> {
        if self.is_finished {
            return Ok(());
        }
        self.is_finished = true;
        let emitted = self.pipeline.finish()?;
        self.queue(emitted);
        Ok(())
    }

    /// Append the emitted blocks to the queue, dropping the part already returned.
    fn queue(&mut self, emitted: Vec<Block>) {
        if self.pending_start > 0 {
            self.pending.drain(..self.pending_start * CHANNELS);
            self.pending_start = 0;
        }
        for block in emitted {
            for frame in 0..block.frames() {
                for channel in 0..CHANNELS {
                    self.pending.push(block.channel(channel)[frame]);
                }
            }
        }
    }

    fn pending_frames(&self) -> usize {
        self.pending.len() / CHANNELS - self.pending_start
    }

    /// Frames of wet audio queued but not returned, which the renderer keeps one block wide.
    #[cfg(test)]
    fn queued_frames(&self) -> usize {
        self.pending_frames()
    }
}

/// The file the live path plays: packets decoded and resampled as the pump asks for them,
/// so the track is never held whole. Both stages are the file path's.
pub struct FileFeed {
    stream: decode::AudioStream,
    /// One pass per channel. A file already at 48 kHz gets the passthrough ratio.
    resampler: Option<[ChunkedResampler<'static>; CHANNELS]>,
    /// Interleaved frames converted but not yet handed to the pump.
    pending: Vec<f32>,
    pending_start: usize,
    is_eof: bool,
}

impl FileFeed {
    /// Decode `path` at 48 kHz stereo, a packet at a time.
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let stream = decode::AudioStream::open(path)?;
        if stream.channels() > CHANNELS {
            tracing::warn!(
                channels = stream.channels(),
                "downmix is not implemented; keeping the first two channels"
            );
        }
        let rate = stream.sample_rate() as usize;
        let left = Resampler::new(SAMPLE_RATE as usize, rate)?.into_chunked();
        let right = Resampler::new(SAMPLE_RATE as usize, rate)?.into_chunked();
        Ok(Self {
            stream,
            resampler: Some([left, right]),
            pending: Vec::new(),
            pending_start: 0,
            is_eof: false,
        })
    }

    /// Pull and convert the next packet, or resolve the resampler's tail at the end.
    fn fill(&mut self) -> Result<()> {
        let Some(chunk) = self.stream.next_chunk()? else {
            self.is_eof = true;
            if let Some([left, right]) = self.resampler.take() {
                let (left, right) = (
                    decode::to_f32(&left.finish()),
                    decode::to_f32(&right.finish()),
                );
                for frame in 0..left.len().min(right.len()) {
                    self.pending.push(left[frame]);
                    self.pending.push(right[frame]);
                }
            }
            return Ok(());
        };
        self.convert(&chunk)
    }

    /// Map one packet to 48 kHz stereo and queue it interleaved.
    fn convert(&mut self, chunk: &Chunk) -> Result<()> {
        let planes = decode::stereo_planes_of(self.stream.channels(), &chunk.planar, chunk.frames)?;
        let [left, right] = self
            .resampler
            .as_mut()
            .context("a packet arrived after the feed finished")?;
        let left_out = left.process_chunk(&decode::to_f64(planes[0]));
        let right_out = right.process_chunk(&decode::to_f64(planes[1]));
        let (left_out, right_out) = (decode::to_f32(&left_out), decode::to_f32(&right_out));
        debug_assert_eq!(left_out.len(), right_out.len(), "one pass per channel");
        for frame in 0..left_out.len().min(right_out.len()) {
            self.pending.push(left_out[frame]);
            self.pending.push(right_out[frame]);
        }
        Ok(())
    }
}

impl Feed for FileFeed {
    fn channels(&self) -> usize {
        CHANNELS
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    fn read(&mut self, out: &mut [f32]) -> Result<usize> {
        let want = out.len() / CHANNELS;
        let mut written = 0;
        while written < want {
            let available = self.pending.len() / CHANNELS - self.pending_start;
            if available > 0 {
                let count = available.min(want - written);
                let from = self.pending_start * CHANNELS;
                out[written * CHANNELS..(written + count) * CHANNELS]
                    .copy_from_slice(&self.pending[from..from + count * CHANNELS]);
                self.pending_start += count;
                written += count;
                continue;
            }
            self.pending.clear();
            self.pending_start = 0;
            if self.is_eof {
                break;
            }
            self.fill()?;
        }
        Ok(written)
    }
}

/// The band limit of a file, read through the same decode and resample the feed plays.
///
/// The whole file goes through the analyser once, in second-sized slices, so the verdict is
/// the one a whole-track read gives and the track is still never held.
pub fn analyse(path: &std::path::Path) -> Result<flacify_core::dsp::cutoff::Analysis> {
    let mut feed = FileFeed::open(path)?;
    let mut analyser = flacify_core::dsp::cutoff::chunked();
    let mut buffer = vec![0.0f32; SAMPLE_RATE as usize * CHANNELS];
    let mut left = Vec::with_capacity(SAMPLE_RATE as usize);
    loop {
        let read = feed.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        left.clear();
        left.extend(buffer[..read * CHANNELS].iter().step_by(CHANNELS).copied());
        analyser.push(&left)?;
    }
    analyser
        .finish()
        .with_context(|| format!("analysing {}", path.display()))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use flacify_core::upscaler::mocks::{Gain, Identity};
    use flacify_core::{live_policy, Preset, Probe};

    /// A policy with this machine's arithmetic out of the way: the buffer and window a test
    /// asks for, and nothing else read by the rings.
    pub(crate) fn test_policy(buffer_frames: usize, window_frames: usize) -> LivePolicy {
        let mut policy = live_policy(Preset::One, &Probe::from_rtf(0.5), 0);
        policy.buffer_frames = buffer_frames;
        policy.window_frames = window_frames;
        policy
    }

    /// The dry sample of a frame, so a dropped or reordered frame is visible in the output.
    pub(crate) fn dry_sample(frame: usize, channel: usize) -> f32 {
        frame as f32 + channel as f32 * 0.5
    }

    /// The wet sample of a frame: far enough from the dry one that a mix cannot be mistaken
    /// for the dry side.
    pub(crate) fn wet_sample(frame: usize, channel: usize) -> f32 {
        dry_sample(frame, channel) + 1_000.0
    }

    /// A feed of `frames` frames whose samples identify the frame.
    pub(crate) struct RampFeed {
        frames: usize,
        delivered: usize,
    }

    impl RampFeed {
        pub(crate) fn new(frames: usize) -> Self {
            Self {
                frames,
                delivered: 0,
            }
        }
    }

    impl Feed for RampFeed {
        fn channels(&self) -> usize {
            2
        }

        fn sample_rate(&self) -> u32 {
            SAMPLE_RATE
        }

        fn read(&mut self, out: &mut [f32]) -> Result<usize> {
            let want = (out.len() / 2).min(self.frames - self.delivered);
            for offset in 0..want {
                let frame = self.delivered + offset;
                out[offset * 2] = dry_sample(frame, 0);
                out[offset * 2 + 1] = dry_sample(frame, 1);
            }
            self.delivered += want;
            Ok(want)
        }
    }

    /// Interleaved `dry_sample` frames for `frames`.
    fn dry_frames(frames: std::ops::Range<usize>) -> Vec<f32> {
        frames
            .flat_map(|frame| [dry_sample(frame, 0), dry_sample(frame, 1)])
            .collect()
    }

    /// Interleaved `wet_sample` frames for `frames`.
    fn wet_frames(frames: std::ops::Range<usize>) -> Vec<f32> {
        frames
            .flat_map(|frame| [wet_sample(frame, 0), wet_sample(frame, 1)])
            .collect()
    }

    /// A producer with nothing to do to the audio: an instant copy, so a test measures the
    /// rings and not the model.
    fn copy_dry(dry: &[f32], wet: &mut [f32]) -> Result<()> {
        wet.copy_from_slice(&dry[..wet.len()]);
        Ok(())
    }

    /// Pull the stream into `out` until it ends, returning the frames delivered.
    fn drain(live: &Live, ramp: &mut Ramp, fill_frames: usize) -> usize {
        let mut out = vec![0.0f32; fill_frames * live.channels()];
        let mut position = 0;
        let deadline = Instant::now() + Duration::from_secs(20);
        while !live.is_finished(position) {
            assert!(Instant::now() < deadline, "the stream never ended");
            let fill = live.fill(ramp, 0.0, position, &mut out);
            if fill.frames == 0 {
                thread::yield_now();
                continue;
            }
            position += fill.frames;
        }
        position
    }

    #[test]
    fn given_a_simulated_hour_when_played_at_a_fixed_buffer_then_the_rings_never_exceed_their_bound(
    ) {
        const BUFFER: usize = 4_096;
        const WINDOW: usize = 2_048;
        const FILL: usize = 1_024;
        let frames = 60 * 60 * SAMPLE_RATE as usize;

        let (live, pump) = Live::start(
            Box::new(RampFeed::new(frames)),
            &test_policy(BUFFER, WINDOW),
            Box::new(copy_dry),
        )
        .expect("a live stream");
        let block = block_frames(&test_policy(BUFFER, WINDOW));
        assert_eq!(
            live.resident_bytes(),
            (2 * BUFFER + WINDOW + block) * 2 * 4,
            "an hour of audio costs the same bytes as a minute"
        );
        assert!(
            live.wait_for_wet(0, FILL, Duration::from_secs(5)),
            "the prebuffer must fill"
        );

        let mut ramp = Ramp::new(false, SAMPLE_RATE);
        let mut out = vec![0.0f32; FILL * 2];
        let mut position = 0;
        let mut peak = 0;
        let deadline = Instant::now() + Duration::from_secs(120);
        while !live.is_finished(position) {
            assert!(Instant::now() < deadline, "the hour never finished");
            let fill = live.fill(&mut ramp, 0.0, position, &mut out);
            if fill.frames == 0 {
                thread::yield_now();
                continue;
            }
            peak = peak.max(live.occupied(position));
            assert!(
                live.occupied(position) <= 2 * BUFFER + WINDOW + block,
                "{} frames resident at {position}",
                live.occupied(position)
            );
            position += fill.frames;
        }
        assert_eq!(position, frames, "every frame of the hour must arrive");
        assert!(peak > 0);
        println!(
            "peak {peak} frames ({:.1} KiB) resident, bound {} frames, for one hour",
            peak as f64 * 2.0 * 4.0 / 1024.0,
            2 * BUFFER + WINDOW + block
        );
        pump.join().expect("the pump");
    }

    #[test]
    fn given_a_wet_producer_that_runs_behind_when_played_then_the_feed_is_delivered_and_the_overlap_counted(
    ) {
        const BUFFER: usize = 2_048;
        const WINDOW: usize = 1_024;
        const FILL: usize = 512;
        let frames = 64 * 1_024;
        let behind = Box::new(|dry: &[f32], wet: &mut [f32]| {
            thread::sleep(Duration::from_millis(2));
            wet.copy_from_slice(&dry[..wet.len()]);
            Ok(())
        });
        let (live, pump) = Live::start(
            Box::new(RampFeed::new(frames)),
            &test_policy(BUFFER, WINDOW),
            behind,
        )
        .expect("a live stream");
        assert!(live.wait_for_wet(0, FILL, Duration::from_secs(5)));

        let mut ramp = Ramp::new(false, SAMPLE_RATE);
        let mut out = vec![0.0f32; FILL * 2];
        let mut position = 0;
        let mut starved = 0;
        let deadline = Instant::now() + Duration::from_secs(20);
        while !live.is_finished(position) {
            assert!(Instant::now() < deadline, "the stream never ended");
            let fill = live.fill(&mut ramp, 0.0, position, &mut out);
            if fill.frames == 0 {
                thread::yield_now();
                continue;
            }
            for offset in 0..fill.frames {
                let frame = position + offset;
                assert_eq!(out[offset * 2], dry_sample(frame, 0), "frame {frame}");
                assert_eq!(out[offset * 2 + 1], dry_sample(frame, 1), "frame {frame}");
            }
            starved += fill.starved_frames;
            position += fill.frames;
        }
        assert_eq!(position, frames, "a slow producer must not drop a frame");
        assert!(
            starved > 0,
            "a producer that runs behind must leave the playhead starved"
        );
        assert_eq!(starved, live.stats().starved_frames());
        assert!(live.stats().dropouts() > 0);
        pump.join().expect("the pump");
    }

    #[test]
    fn given_a_starved_playhead_when_filled_then_the_dry_samples_play_and_are_counted() {
        // Hand-driven rings: no pump, so the wet side appears at a frame this test picks.
        let rings = Rings::new(128, 32, 32, 2, Arc::new(Stats::default()));
        let live = Live {
            rings: Arc::new(rings),
        };
        live.rings.dry.write(0, &dry_frames(0..96));
        let mut ramp = Ramp::new(true, SAMPLE_RATE);
        let mut out = vec![0.0f32; 32 * 2];

        let fill = live.fill(&mut ramp, 1.0, 0, &mut out);
        assert_eq!(
            fill,
            Fill {
                frames: 32,
                starved_frames: 32
            }
        );
        assert_eq!(out, dry_frames(0..32), "the dry side, whole");
        assert_eq!(live.stats().starved_frames(), 32);
        assert_eq!(live.stats().dropouts(), 1);

        // The producer catches up: the same frames now have a wet side.
        live.rings.wet.write(32, &wet_frames(32..64));
        let fill = live.fill(&mut ramp, 1.0, 32, &mut out);
        assert_eq!(
            fill,
            Fill {
                frames: 32,
                starved_frames: 0
            }
        );
        assert_eq!(out, wet_frames(32..64), "the wet side, once it exists");
        assert_eq!(
            live.stats().dropouts(),
            1,
            "a covered fill must not add a dropout"
        );
    }

    #[test]
    fn given_the_end_of_a_feed_when_the_rings_drain_then_the_fill_reports_the_end() {
        const FILL: usize = 1_024;
        let frames = 5 * FILL;
        let (live, pump) = Live::start(
            Box::new(RampFeed::new(frames)),
            &test_policy(2_048, 1_024),
            Box::new(copy_dry),
        )
        .expect("a live stream");
        assert!(live.wait_for_wet(0, FILL, Duration::from_secs(5)));

        let mut ramp = Ramp::new(false, SAMPLE_RATE);
        let delivered = drain(&live, &mut ramp, FILL);
        assert_eq!(delivered, frames);
        assert!(live.is_finished(delivered));

        // Drained and finished: nothing repeats, nothing stalls.
        let mut out = vec![0.0f32; FILL * 2];
        assert_eq!(
            live.fill(&mut ramp, 0.0, delivered, &mut out),
            Fill::default()
        );
        assert_eq!(live.occupied(delivered), 0);
        pump.join().expect("the pump");
        assert_eq!(
            live.rings.wet.written(),
            frames,
            "the tail gets its wet side from a zero-padded lookahead"
        );
    }

    #[test]
    fn given_a_playhead_that_frees_half_a_block_per_render_when_pumped_then_it_renders_ahead_of_it()
    {
        // A render takes real time, and the playhead keeps draining through it. The room a
        // block needs is asked for after the render, so a render can start while the
        // playhead is short of it: at half a block per render, three blocks get through.
        // Asking before the render stalls after one.
        const WINDOW: usize = 1_024;
        let policy = test_policy(820, WINDOW);
        let block = block_frames(&policy);
        let rings = Arc::new(Rings::new(
            820,
            WINDOW,
            block,
            CHANNELS,
            Arc::new(Stats::default()),
        ));
        let playhead = Arc::clone(&rings.playhead);
        let calls = Arc::new(AtomicUsize::new(0));
        let renders = Arc::clone(&calls);
        let pump = Pump::spawn(
            Arc::clone(&rings),
            Box::new(RampFeed::new(20 * block)),
            Box::new(move |dry: &[f32], wet: &mut [f32]| {
                wet.copy_from_slice(&dry[..wet.len()]);
                playhead.fetch_add(block / 2, Ordering::Release);
                renders.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }),
        )
        .expect("a pump");

        let deadline = Instant::now() + Duration::from_secs(5);
        while calls.load(Ordering::Relaxed) < 3 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            calls.load(Ordering::Relaxed),
            3,
            "the pump must render ahead of its room"
        );
        rings.stop();
        pump.join().expect("the pump");
    }

    #[test]
    fn given_a_playhead_that_never_moves_when_the_ring_fills_then_the_pump_waits_for_credit() {
        // No consumer at all: the pump must stop at the bound rather than overwrite audio
        // the playhead has not read.
        let buffer = 2_048;
        let window = 1_024;
        let (live, pump) = Live::start(
            Box::new(RampFeed::new(1_000_000)),
            &test_policy(buffer, window),
            Box::new(copy_dry),
        )
        .expect("a live stream");

        thread::sleep(Duration::from_millis(100));
        let pushed = live.rings.dry.written();
        let block = block_frames(&test_policy(buffer, window));
        assert!(
            pushed <= buffer + window + block,
            "{pushed} frames pushed into a {buffer} frame buffer with {window} of lookahead"
        );
        assert!(live.rings.wet.written() <= buffer);
        assert!(
            !pump.is_finished(),
            "a blocked producer is waiting, not finished"
        );

        // The playhead moves, and so does the pump.
        live.playhead().store(4 * 1_024, Ordering::Release);
        thread::sleep(Duration::from_millis(100));
        assert!(
            live.rings.dry.written() > pushed,
            "the pump must resume once the playhead frees room"
        );
        live.rings.stop();
        pump.join().expect("the pump");
    }

    /// The end-to-end figure `--live` starts from until the app measures its own (S3).
    const LIVE_PROBE: Probe = Probe {
        rtf_p50: 0.55,
        rtf_p95: 0.70,
        rtf_sustained: 0.78,
    };

    /// A small mono-ish signal, so a tolerance is a real check rather than a rounding test.
    fn tone(frame: usize) -> f32 {
        (frame as f32 * 0.01).sin() * 0.5
    }

    /// Interleaved stereo of [`tone`] for `frames`.
    fn tone_track(frames: usize) -> Vec<f32> {
        (0..frames)
            .flat_map(|frame| [tone(frame), tone(frame + 1)])
            .collect()
    }

    /// Drive a renderer the way the pump does: each call carries one block plus the window of
    /// lookahead after it, and the last call carries what is left.
    fn render(
        dry: &[f32],
        window: usize,
        block: usize,
        renderer: &mut Renderer<impl Upscaler>,
    ) -> Vec<f32> {
        let total = dry.len() / CHANNELS;
        let mut wet = vec![0.0f32; block * CHANNELS];
        let mut out = Vec::with_capacity(dry.len());
        let mut produced = 0;
        while produced < total {
            let frames = block.min(total - produced);
            let end = (produced + block + window).min(total);
            renderer
                .produce(
                    &dry[produced * CHANNELS..end * CHANNELS],
                    &mut wet[..frames * CHANNELS],
                )
                .expect("one block");
            out.extend_from_slice(&wet[..frames * CHANNELS]);
            produced += frames;
        }
        out
    }

    #[test]
    fn given_the_four_presets_when_sized_then_the_block_keeps_the_pipeline_inside_its_clamp() {
        for preset in Preset::ALL {
            let policy = live_policy(preset, &LIVE_PROBE, 1_024);
            let block = block_frames(&policy);
            assert!(block > 0 && block <= policy.window_frames, "{block}");
            assert!(block <= policy.buffer_frames, "{block}");
            let overlap = policy.window_frames - block;
            assert_eq!(overlap, overlap_frames(policy.window_frames));
            assert!(
                overlap <= policy.window_frames / 2,
                "{} frames of overlap at {}",
                policy.window_frames,
                preset.label()
            );
            let _renderer = Renderer::new(Identity, policy.window_frames, block, SEED)
                .expect("the preset's block is the pipeline's step");
            // The prebuffer is whole blocks inside the buffer, and at least one block.
            let prebuffer = prebuffer_frames(&policy);
            assert_eq!(prebuffer % block, 0, "{prebuffer} of {block}");
            assert!(prebuffer >= block && prebuffer <= policy.buffer_frames);
            assert!(prebuffer <= policy.window_frames + block);
        }
    }

    #[test]
    fn given_a_block_that_is_not_the_pipelines_step_when_a_renderer_is_built_then_it_is_refused() {
        let longer = Renderer::new(Identity, 1_024, 2_048, SEED);
        assert!(longer.is_err(), "a block cannot be longer than the window");

        // An overlap past the half-window clamp would make the step disagree with the
        // pump's block, which queues or starves wet audio for the whole stream.
        let clamped = Renderer::new(Identity, 1_024, 100, SEED)
            .err()
            .expect("the pipeline would step by 513");
        assert!(format!("{clamped}").contains("513"), "{clamped}");
    }

    #[test]
    fn given_a_buffer_that_is_not_whole_stereo_frames_when_produced_into_then_it_is_refused() {
        let mut renderer = Renderer::new(Identity, 1_024, 820, SEED).expect("a renderer");
        let dry = tone_track(2_000);
        let error = renderer
            .produce(&dry, &mut [0.0f32; 3])
            .expect_err("three samples are not a block");
        assert!(format!("{error}").contains("stereo"), "{error}");
    }

    #[test]
    fn given_any_length_when_rendered_a_block_at_a_time_then_the_block_boundaries_lose_nothing() {
        const WINDOW: usize = 4_096;
        const BLOCK: usize = 2_049;
        for frames in [1, 100, 2_048, 2_049, 4_999, 20_001] {
            let dry = tone_track(frames);
            let mut renderer = Renderer::new(Identity, WINDOW, BLOCK, SEED).expect("a renderer");
            let out = render(&dry, WINDOW, BLOCK, &mut renderer);
            assert_eq!(out.len(), dry.len(), "{frames} frames");
            for (index, (got, want)) in out.iter().zip(dry.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-4,
                    "sample {index} of {frames}: {got} for {want}"
                );
            }
        }
    }

    #[test]
    fn given_a_long_stream_when_rendered_then_the_queued_wet_audio_stays_one_block_deep() {
        // A queue that kept every emitted block would hold the whole track, which is the
        // memory the live path exists to avoid.
        const WINDOW: usize = 4_096;
        const BLOCK: usize = 3_277;
        let dry = tone_track(200 * BLOCK);
        let mut renderer = Renderer::new(Identity, WINDOW, BLOCK, SEED).expect("a renderer");
        let mut wet = vec![0.0f32; BLOCK * CHANNELS];
        let mut peak = 0usize;
        let mut produced = 0;
        while produced < dry.len() / CHANNELS {
            let frames = BLOCK.min(dry.len() / CHANNELS - produced);
            let end = (produced + BLOCK + WINDOW).min(dry.len() / CHANNELS);
            renderer
                .produce(
                    &dry[produced * CHANNELS..end * CHANNELS],
                    &mut wet[..frames * CHANNELS],
                )
                .expect("one block");
            peak = peak.max(renderer.queued_frames());
            produced += frames;
        }
        assert!(peak <= 2 * BLOCK, "{peak} frames queued");
        assert_eq!(produced, dry.len() / CHANNELS);
    }

    #[test]
    fn given_a_gain_when_rendered_a_block_at_a_time_then_every_frame_carries_it() {
        const WINDOW: usize = 4_096;
        const BLOCK: usize = 2_049;
        let dry = tone_track(3 * BLOCK);
        let mut renderer = Renderer::new(Gain(2.0), WINDOW, BLOCK, SEED).expect("a renderer");
        let out = render(&dry, WINDOW, BLOCK, &mut renderer);
        for (index, (got, want)) in out.iter().zip(dry.iter()).enumerate() {
            assert!(
                (got - 2.0 * want).abs() < 1e-3,
                "sample {index}: {got} for {}",
                2.0 * want
            );
        }
    }

    #[test]
    fn given_a_finished_renderer_when_handed_more_audio_then_it_refuses_instead_of_misaligning() {
        const WINDOW: usize = 4_096;
        const BLOCK: usize = 2_049;
        let dry = tone_track(3 * BLOCK);
        let mut renderer = Renderer::new(Identity, WINDOW, BLOCK, SEED).expect("a renderer");
        render(&dry, WINDOW, BLOCK, &mut renderer);

        let mut wet = vec![0.0f32; BLOCK * CHANNELS];
        let more = tone_track(2 * BLOCK + WINDOW);
        let error = renderer
            .produce(&more, &mut wet)
            .expect_err("the stream already ended");
        assert!(format!("{error}").contains("ended"), "{error}");
    }

    #[test]
    fn given_two_blocks_when_measured_then_there_is_no_ratio_and_three_give_one() {
        let stats = Stats::default();
        let block = SAMPLE_RATE as usize;
        stats.record_block(Duration::from_millis(700), block);
        stats.record_block(Duration::from_millis(700), block);
        assert_eq!(stats.blocks_rendered(), 2);
        assert_eq!(stats.measured_rtf(), None, "two windows are not enough");

        stats.record_block(Duration::from_millis(700), block);
        let rtf = stats.measured_rtf().expect("three blocks are enough");
        assert!((rtf - 0.7).abs() < 1e-6, "{rtf}");
    }

    #[test]
    fn given_more_blocks_than_timings_kept_when_measured_then_it_rolls_over_the_last_eight() {
        let stats = Stats::default();
        let block = SAMPLE_RATE as usize;
        for _ in 0..TIMED_BLOCKS {
            stats.record_block(Duration::from_millis(100), block);
        }
        // Replace the oldest timing and the ratio follows the ring, not the whole run.
        stats.record_block(Duration::from_millis(900), block);
        let rtf = stats.measured_rtf().expect("a ratio");
        let expected = (7.0 * 0.1 + 0.9) / TIMED_BLOCKS as f64;
        assert!((rtf - expected).abs() < 1e-6, "{rtf} for {expected}");
    }

    #[test]
    fn given_a_real_renderer_when_the_pump_drives_it_then_every_frame_arrives_aligned() {
        const WINDOW: usize = 4_096;
        let policy = test_policy(2 * WINDOW, WINDOW);
        let block = block_frames(&policy);
        let frames = 20 * block;
        let mut renderer = Renderer::new(Identity, WINDOW, block, SEED).expect("a renderer");
        let (live, pump) = Live::start(
            Box::new(RampFeed::new(frames)),
            &policy,
            Box::new(move |dry, wet| renderer.produce(dry, wet)),
        )
        .expect("a live stream");
        assert!(
            live.wait_for_wet(0, prebuffer_frames(&policy), Duration::from_secs(10)),
            "the prebuffer fills"
        );

        let mut ramp = Ramp::new(true, SAMPLE_RATE);
        let mut out = vec![0.0f32; 1_024 * CHANNELS];
        let mut position = 0;
        let deadline = Instant::now() + Duration::from_secs(30);
        while !live.is_finished(position) {
            assert!(Instant::now() < deadline, "the stream never ended");
            let fill = live.fill(&mut ramp, 1.0, position, &mut out);
            if fill.frames == 0 {
                thread::yield_now();
                continue;
            }
            for offset in 0..fill.frames {
                let frame = position + offset;
                let want = dry_sample(frame, 0);
                let got = out[offset * CHANNELS];
                assert!(
                    (got - want).abs() <= 1e-5 * want.abs() + 1e-3,
                    "frame {frame}: {got} for {want}"
                );
            }
            position += fill.frames;
        }
        assert_eq!(position, frames, "every frame must play");
        assert!(
            live.stats().blocks_rendered() >= 3,
            "{} windows",
            live.stats().blocks_rendered()
        );
        pump.join().expect("the pump");
    }

    #[test]
    fn given_a_48_khz_file_when_read_through_the_feed_then_every_sample_arrives_unchanged() {
        let root = crate::live::tests::temp_root("feed-48k");
        let path = root.join("source.wav");
        let frames = 12_345;
        let left: Vec<f32> = (0..frames).map(tone).collect();
        let right: Vec<f32> = (0..frames).map(|frame| tone(frame + 7)).collect();
        let planar: Vec<f32> = left.iter().chain(right.iter()).copied().collect();
        crate::wav::write(&path, SAMPLE_RATE, 2, &planar).expect("writing the wav");
        let expected: Vec<f32> = (0..frames)
            .flat_map(|frame| [left[frame], right[frame]])
            .collect();

        let mut feed = FileFeed::open(&path).expect("a feed");
        let mut buffer = vec![0.0f32; 1_000 * CHANNELS];
        let mut out = Vec::new();
        loop {
            let read = feed.read(&mut buffer).expect("reading the feed");
            if read == 0 {
                break;
            }
            out.extend_from_slice(&buffer[..read * CHANNELS]);
        }
        assert_eq!(out.len(), expected.len(), "every frame arrives once");
        for (index, (got, want)) in out.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got.to_bits(), want.to_bits(), "sample {index}");
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_44_khz_file_when_read_through_the_feed_then_it_matches_the_whole_file_path() {
        let root = crate::live::tests::temp_root("feed-44k");
        let path = root.join("source.wav");
        let frames = 20_000;
        let left: Vec<f32> = (0..frames).map(tone).collect();
        let planar: Vec<f32> = left.iter().chain(left.iter()).copied().collect();
        crate::wav::write(&path, 44_100, 2, &planar).expect("writing the wav");

        let decoded = decode::decode(&path).expect("decoding the wav");
        let whole = decode::to_stereo_48k(&decoded).expect("the whole-file mapping");
        let frames = whole.len() / 2;
        let expected: Vec<f32> = (0..frames)
            .flat_map(|frame| [whole[frame], whole[frames + frame]])
            .collect();

        let mut feed = FileFeed::open(&path).expect("a feed");
        let mut buffer = vec![0.0f32; 1_003 * CHANNELS];
        let mut streamed = Vec::new();
        loop {
            let read = feed.read(&mut buffer).expect("reading the feed");
            if read == 0 {
                break;
            }
            streamed.extend_from_slice(&buffer[..read * CHANNELS]);
        }
        assert_eq!(
            streamed.len(),
            expected.len(),
            "the chunked pass is the whole pass"
        );
        for (index, (got, want)) in streamed.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got.to_bits(), want.to_bits(), "sample {index}");
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_file_when_analysed_for_live_then_the_cutoff_is_the_whole_file_verdict() {
        let root = crate::live::tests::temp_root("analyse");
        let path = root.join("source.wav");
        let frames = 3 * 48_000;
        let samples: Vec<f32> = (0..frames)
            .map(|frame| {
                let time = frame as f32 / 48_000.0;
                (std::f32::consts::TAU * 440.0 * time).sin() * 0.3
                    + (std::f32::consts::TAU * 12_000.0 * time).sin() * 0.2
            })
            .collect();
        let planar: Vec<f32> = samples.iter().chain(samples.iter()).copied().collect();
        crate::wav::write(&path, 48_000, 2, &planar).expect("writing the wav");

        let whole = crate::render::load_in_memory(&path).expect("the whole-file analysis");
        let live = analyse(&path).expect("the live analysis");
        assert_eq!(live.cutoff_hz, whole.analysis.cutoff_hz);
        assert_eq!(live.high_band_ratio, whole.analysis.high_band_ratio);
        std::fs::remove_dir_all(root).ok();
    }

    /// A unique data directory in the temp directory, deleted by the test that made it.
    fn temp_root(name: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!(
            "flacify-live-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("creating the test directory");
        root
    }
}
