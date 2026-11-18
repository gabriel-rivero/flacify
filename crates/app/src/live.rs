//! Live playback: a feed goes in, the A/B mix comes out, and memory stays bounded.
//!
//! One pump thread reads a [`Feed`] and pushes the dry side into a ring holding
//! `buffer_frames + window_frames`. When it has pushed a block it asks the [`WetProducer`]
//! for the wet side of the frames behind that block, which is where the model needs a
//! window of lookahead; the wet side goes into a ring holding `buffer_frames`. The pump
//! waits for room instead of dropping audio, so resident audio is
//! `(2 × buffer + window) × channels × 4` bytes for the whole stream.
//!
//! [`Live::fill`] is the consumer, on the audio thread: it reads both rings at the
//! playhead, mixes wet over dry through the mixer's ramp, plays the dry side wherever the
//! wet side has not caught up, and counts those frames. No allocation, no lock, no
//! blocking, no logging.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use flacify_core::policy::LivePolicy;
use flacify_core::SAMPLE_RATE;

use crate::player::{mix_stretch, Ramp};

/// Why the A/B switch is unavailable while the buffer is behind, in the words the status
/// line prints.
pub const STARVED_REASON: &str = "the live buffer has not caught up with the playhead";

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

/// The counters the live path publishes. The status line and `diagnose` read them without
/// a lock, and the producer thread never touches them.
#[derive(Debug, Default)]
pub struct Stats {
    starved_frames: AtomicUsize,
    dropouts: AtomicUsize,
    is_wet_covered: AtomicBool,
    is_starving: AtomicBool,
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

    fn capacity(&self) -> usize {
        self.capacity
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
}

impl Rings {
    /// `block_frames` is what one [`WetProducer`] call produces, and the pump keeps the
    /// rings sized from the policy so the credit can never be exceeded.
    fn new(
        buffer_frames: usize,
        window_frames: usize,
        block_frames: usize,
        channels: usize,
    ) -> Self {
        Self {
            dry: Ring::new(buffer_frames + window_frames, channels),
            wet: Ring::new(buffer_frames, channels),
            channels,
            window_frames,
            block_frames,
            playhead: Arc::new(AtomicUsize::new(0)),
            is_feed_finished: AtomicBool::new(false),
            is_stopping: AtomicBool::new(false),
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
    stats: Arc<Stats>,
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
        ));
        let live = Arc::new(Self {
            stats: Arc::new(Stats::default()),
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

    /// The counters the status line and `diagnose` print.
    pub fn stats(&self) -> Arc<Stats> {
        Arc::clone(&self.stats)
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
        self.stats.record(fill.starved_frames, dry_frames == frames);
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

/// The block one [`WetProducer`] call produces: half the buffer, capped at the model
/// window, so the next block has room while the playhead is still inside this one.
fn block_frames(policy: &LivePolicy) -> usize {
    (policy.buffer_frames / 2).clamp(1, policy.window_frames)
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
        if rings
            .wet
            .wait_for_room(produced + frames, &rings.playhead, &rings.is_stopping)
            == Room::Stopped
        {
            return Ok(());
        }
        produce(&stage[..filled * channels], &mut wet[..frames * channels])?;
        rings.wet.write(produced, &wet[..frames * channels]);
        produced += frames;

        // Slide the stage forward: what was lookahead becomes the start of the next slice.
        stage.copy_within(frames * channels..filled * channels, 0);
        filled -= frames;
        stage_start += frames;
        stage[filled * channels..].fill(0.0);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
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
        assert_eq!(
            live.resident_bytes(),
            (2 * BUFFER + WINDOW) * 2 * 4,
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
                live.occupied(position) <= 2 * BUFFER + WINDOW,
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
            2 * BUFFER + WINDOW
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
        let rings = Rings::new(128, 32, 32, 2);
        let live = Live {
            stats: Arc::new(Stats::default()),
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
        assert!(
            pushed <= buffer + window,
            "{pushed} frames pushed into a {buffer} frame buffer"
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
}
