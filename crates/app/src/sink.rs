//! The sink's feed: one capture device into the live rings, corrected to the output
//! device's clock.
//!
//! The two devices run on different crystals, so the ring between them slowly fills or
//! empties. The capture callback only writes into a preallocated ring: no allocation, no
//! lock, no wait. The feed's thread reads it through a cubic interpolator whose ratio is a
//! proportional controller on the ring's fill. When the correction saturates and the fill
//! reaches a ring edge, the read resynchronises with a short crossfade and counts it:
//! a deliberate resync beats a click or a latency that grows until the ring overflows.
//!
//! Capture is not resampled. A device that cannot give 48 kHz stereo is refused by
//! [`Input::open_named`] with its own format, because the drift trim is a rate correction
//! and a converter in front of it would be two resamplers for one signal.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use flacify_core::dsp::cutoff;
use flacify_core::SAMPLE_RATE;

use crate::device::Input;
use crate::live::{self, Feed};

/// The live path is 48 kHz stereo end to end, and so is the ring.
const CHANNELS: usize = 2;

/// Frames the capture ring holds: 400 ms, twice the fill the loop steers to.
pub const RING_FRAMES: usize = 19_200;

/// The fill the loop steers to: half the ring, so a resync has equal room both ways.
pub const TARGET_FILL_FRAMES: usize = RING_FRAMES / 2;

/// The crossfade a resync splices the jump over: 10 ms, long enough to hide the step and
/// short enough that the ring does not have to hold much for it.
pub const RESYNC_CROSSFADE_FRAMES: usize = SAMPLE_RATE as usize / 100;

/// Frames one read asks for: 50 ms, which is also the control step.
pub const READ_FRAMES: usize = SAMPLE_RATE as usize / 20;

/// The largest rate correction the loop applies, in parts per million.
pub const MAX_CORRECTION_PPM: f64 = 1_000.0;

/// Gain of the proportional controller: ppm of correction per frame of fill error.
///
/// The plant is first order: the fill moves by (drift minus correction) frames per frame,
/// so a proportional law settles without overshoot. This gain gives a time constant near
/// 10 s and holds 50 ppm of drift at 25 frames of error, well inside the ring.
pub const FILL_GAIN_PPM_PER_FRAME: f64 = 2.0;

/// A ring this shallow is a stream that stopped, not a clock to correct: repeating it would
/// loop the same few milliseconds forever instead of letting the playhead starve.
const MIN_REPEAT_FRAMES: usize = 2 * RESYNC_CROSSFADE_FRAMES;

/// The shortest the feed's thread sleeps while it waits for the capture callback.
const MIN_WAIT: Duration = Duration::from_millis(1);

/// Seconds of capture measured before the sink starts: the preprocessor is built for the
/// band the source actually has, because the sink has no file to analyse.
pub const MEASURE_SECS: usize = 1;

/// Frames of consecutive digital silence that mean the capture is not carrying a signal.
///
/// macOS never refuses the device: a process without Microphone permission opens the input,
/// the stream delivers its frames, and every sample is zero. Five seconds of that is the
/// whole signature, so the sink treats it as its own state rather than as quiet audio.
pub const SILENT_FRAMES: usize = 5 * SAMPLE_RATE as usize;

/// A block whose samples are all below this is digital silence: far below any converter's
/// noise floor, and nowhere near a signal at any listening level.
const SILENT_PEAK: f32 = 1e-6;

/// What a silent capture most likely means, in the words the report, the error and the
/// panel all print. One string, so the three cannot drift apart.
pub const SILENT_HINT: &str = "digital silence: either nothing is playing into the capture \
     device, or macOS denied Microphone permission to the app that launched this (System \
     Settings → Privacy & Security → Microphone)";

/// Whether the capture is carrying a signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureState {
    /// Frames are arriving and they carry audio.
    Signal,
    /// Frames are arriving and the last [`SILENT_FRAMES`] of them are digital silence.
    Silent,
}

/// The largest absolute sample of a block, which is what tells digital silence apart.
fn peak_of(block: &[f32]) -> f32 {
    block
        .iter()
        .fold(0.0f32, |peak, sample| peak.max(sample.abs()))
}

/// The largest correction the loop applied, in thousandths of a ppm, for the peak.
fn milli_ppm(ppm: f64) -> usize {
    (ppm.abs() * 1_000.0) as usize
}

/// What the sink has had to do to keep the two clocks together. The status line and the
/// panel read these without a lock; the feed's thread is the only writer.
#[derive(Debug, Default)]
pub struct SinkStats {
    resyncs: AtomicUsize,
    dropped_frames: AtomicUsize,
    /// The correction applied by the last control step, as `f64` bits.
    correction_ppm: AtomicU64,
    /// The largest correction magnitude so far, in thousandths of a ppm.
    peak_milli_ppm: AtomicUsize,
    fill_frames: AtomicUsize,
    /// Frames of consecutive digital silence, capped at [`SILENT_FRAMES`].
    silent_run: AtomicUsize,
    /// Whether any block has carried signal since the feed opened.
    has_signal: AtomicBool,
}

impl SinkStats {
    /// Resyncs: drop or repeat blocks spliced in because the correction ran out of range.
    pub fn resyncs(&self) -> usize {
        self.resyncs.load(Ordering::Relaxed)
    }

    /// Frames the capture callback could not write because the read had not freed room.
    pub fn dropped_frames(&self) -> usize {
        self.dropped_frames.load(Ordering::Relaxed)
    }

    /// The correction the loop is applying now, in ppm: positive means the capture clock is
    /// ahead, so the read consumes its frames faster than one per output frame.
    pub fn correction_ppm(&self) -> f64 {
        f64::from_bits(self.correction_ppm.load(Ordering::Relaxed))
    }

    /// The largest correction magnitude applied since the feed started.
    pub fn peak_correction_ppm(&self) -> f64 {
        self.peak_milli_ppm.load(Ordering::Relaxed) as f64 / 1_000.0
    }

    /// Frames the ring held at the last control step.
    pub fn fill_frames(&self) -> usize {
        self.fill_frames.load(Ordering::Relaxed)
    }

    /// Frames of consecutive digital silence the capture has delivered, capped at
    /// [`SILENT_FRAMES`].
    pub fn silent_frames(&self) -> usize {
        self.silent_run.load(Ordering::Relaxed)
    }

    /// Whether the capture is carrying a signal: the state every other number here is worth
    /// nothing without.
    pub fn capture_state(&self) -> CaptureState {
        if self.silent_frames() >= SILENT_FRAMES {
            CaptureState::Silent
        } else {
            CaptureState::Signal
        }
    }

    /// Whether any block has carried signal since the feed opened. A run that never did is
    /// a refused microphone, not a quiet passage.
    pub fn has_carried_signal(&self) -> bool {
        self.has_signal.load(Ordering::Relaxed)
    }

    fn record_correction(&self, ppm: f64, fill: usize) {
        self.correction_ppm.store(ppm.to_bits(), Ordering::Relaxed);
        self.fill_frames.store(fill, Ordering::Relaxed);
        self.peak_milli_ppm
            .fetch_max(milli_ppm(ppm), Ordering::Relaxed);
    }

    fn record_resync(&self) {
        self.resyncs.fetch_add(1, Ordering::Relaxed);
    }

    fn record_dropped(&self, frames: usize) {
        self.dropped_frames.fetch_add(frames, Ordering::Relaxed);
    }

    /// One capture block's peak: a silent block extends the run, anything else ends it.
    fn record_peak(&self, peak: f32, frames: usize) {
        if peak >= SILENT_PEAK {
            self.has_signal.store(true, Ordering::Relaxed);
            self.silent_run.store(0, Ordering::Relaxed);
            return;
        }
        // The capture callback is the only writer, so a load and a store is enough.
        let run = self.silent_run.load(Ordering::Relaxed);
        self.silent_run
            .store((run + frames).min(SILENT_FRAMES), Ordering::Relaxed);
    }
}

/// The rate correction a fill error asks for, in ppm, clamped to the loop's limit.
///
/// Positive means the ring is too full, so the read consumes captured frames faster than
/// one per output frame.
pub fn correction_for(fill: usize, target: usize) -> f64 {
    let error = fill as f64 - target as f64;
    (error * FILL_GAIN_PPM_PER_FRAME).clamp(-MAX_CORRECTION_PPM, MAX_CORRECTION_PPM)
}

/// What a saturated loop at a ring edge must do to stay bounded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resync {
    /// The capture clock is ahead: skip `frames` of captured audio.
    Skip(usize),
    /// The output clock is ahead: play `frames` of captured audio again.
    Repeat(usize),
}

/// The resync a saturated loop at an edge asks for, or `None` while the fill is inside.
///
/// The edges are a quarter ring from either end, and the jump is a quarter ring, so either
/// answer puts the fill back on its target: `RING_FRAMES / 4` is 100 ms. A repeat also needs a writer that is still
/// moving: repeating a ring whose capture has stopped would loop a few milliseconds
/// forever instead of letting the playhead starve, which is the caller's to count.
pub fn resync_for(
    fill: usize,
    capacity: usize,
    ppm: f64,
    is_writer_moving: bool,
) -> Option<Resync> {
    if ppm >= MAX_CORRECTION_PPM && fill * 4 >= capacity * 3 {
        return Some(Resync::Skip(capacity / 4));
    }
    let is_draining = ppm <= -MAX_CORRECTION_PPM && fill * 4 <= capacity;
    if !is_draining || !is_writer_moving || fill < MIN_REPEAT_FRAMES {
        return None;
    }
    Some(Resync::Repeat((capacity / 4).min(fill)))
}

/// The fill a read at `position` sees: captured frames it has not consumed yet.
fn fill_of(written: usize, position: f64) -> usize {
    if position >= written as f64 {
        return 0;
    }
    written - position.floor().max(0.0) as usize
}

/// How many of the `want` frames at `position` the ring can produce.
///
/// The cubic read needs one tap before the frame and two after it, so the furthest read is
/// three frames past the last frame it can produce.
fn frames_ready(position: f64, written: usize, ratio: f64, want: usize) -> usize {
    let Some(last) = written.checked_sub(3) else {
        return 0;
    };
    let last = last as f64;
    if position > last || ratio <= 0.0 {
        return 0;
    }
    let span = ((last - position) / ratio).floor().max(0.0) as usize;
    want.min(span + 1)
}

/// The uniform Catmull-Rom spline through four samples, at `t` in `0..=1` from `b` to `c`.
///
/// It reproduces a straight line exactly, which is what keeps a ppm trim inaudible: the
/// read position moves by a thousandth of a frame per frame, and a plain linear
/// interpolator's passband loss would cost more than the correction gains.
pub fn catmull_rom(a: f32, b: f32, c: f32, d: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * b)
        + (-a + c) * t
        + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2
        + (-a + 3.0 * b - 3.0 * c + d) * t3)
}

/// One frame of `ring` read at a fractional position.
fn sample_at(ring: &live::Ring, position: f64, channel: usize) -> f32 {
    let base = position.floor().max(0.0);
    let t = (position - base) as f32;
    let frame = base as usize;
    catmull_rom(
        ring.sample(frame.saturating_sub(1), channel),
        ring.sample(frame, channel),
        ring.sample(frame + 1, channel),
        ring.sample(frame + 2, channel),
        t,
    )
}

/// The crossfade a resync is splicing: from where the read was, to where it jumped.
#[derive(Clone, Copy, Debug)]
struct Crossfade {
    /// Absolute frame the fade started reading at. Advances with the read position.
    from: f64,
    /// Frames of the fade already played.
    played: usize,
}

/// The fractional read the feed performs over the capture ring.
///
/// Separate from [`CaptureFeed`] so the loop can be simulated without a device: the tests
/// drive [`Reader::read`] and a writer by frame counts, not by wall clock.
pub(crate) struct Reader {
    ring: Arc<live::Ring>,
    /// The read position the capture callback measures its room against. Published after
    /// every control step, so a callback that sees a stale value sees less room, never more.
    head: Arc<AtomicUsize>,
    stats: Arc<SinkStats>,
    /// Absolute capture frame the next output frame comes from, fractionally.
    position: f64,
    /// Captured frames consumed per output frame: the controller's output.
    ratio: f64,
    crossfade: Option<Crossfade>,
    /// False until the first read has pre-rolled to the target fill.
    is_started: bool,
    /// The last published capture count, so a stalled writer is told from a slow clock.
    last_written: usize,
    /// Frames the ring holds, and the fill the loop steers to.
    capacity: usize,
    target: usize,
}

impl Reader {
    /// A reader over `ring`, with the head and counters its writer shares.
    pub(crate) fn new(
        ring: Arc<live::Ring>,
        head: Arc<AtomicUsize>,
        stats: Arc<SinkStats>,
    ) -> Self {
        let capacity = ring.capacity();
        Self {
            ring,
            head,
            stats,
            position: 0.0,
            ratio: 1.0,
            crossfade: None,
            is_started: false,
            last_written: 0,
            capacity,
            target: capacity / 2,
        }
    }

    /// Fill `out` with the next frames that are readable now, and return how many.
    ///
    /// Zero means the ring does not hold a complete read yet: the caller waits, because a
    /// caller that guessed would be inventing silence the capture never had.
    pub(crate) fn read(&mut self, out: &mut [f32]) -> usize {
        let want = (out.len() / CHANNELS).min(READ_FRAMES);
        if want == 0 {
            return 0;
        }
        let written = self.ring.written();
        if !self.is_started {
            // Pre-roll: the first read starts a target's worth behind the writer, so the
            // loop opens at its set point instead of crawling there at a thousandth of a
            // frame per frame.
            if written < self.target {
                return 0;
            }
            self.position = (written - self.target) as f64;
            self.is_started = true;
            self.last_written = written;
        }

        let fill = fill_of(written, self.position);
        let ppm = correction_for(fill, self.target);
        self.ratio = 1.0 + ppm / 1_000_000.0;
        self.stats.record_correction(ppm, fill);
        if self.crossfade.is_none() {
            let is_writer_moving = written > self.last_written;
            if let Some(resync) = resync_for(fill, self.capacity, ppm, is_writer_moving) {
                self.resync(resync);
            }
        }
        self.last_written = written;

        // A repeat reads from `position` while the fade still reads from `from`, and the
        // fade's side is the further of the two after a repeat.
        let furthest = match self.crossfade {
            Some(fade) => self.position.max(fade.from),
            None => self.position,
        };
        let frames = frames_ready(furthest, written, self.ratio, want);
        if frames == 0 {
            return 0;
        }
        self.emit(out, frames);
        self.publish_head();
        frames
    }

    /// Jump the read position by one resync's worth and start the fade across the step.
    fn resync(&mut self, resync: Resync) {
        let from = self.position;
        self.position = match resync {
            Resync::Skip(frames) => from + frames as f64,
            Resync::Repeat(frames) => from - frames as f64,
        };
        self.crossfade = Some(Crossfade { from, played: 0 });
        self.stats.record_resync();
        // A repeat moves the head back over audio the reader still wants, so it is
        // published before any of it is re-read.
        self.publish_head();
    }

    fn publish_head(&self) {
        self.head
            .store(self.position.floor().max(0.0) as usize, Ordering::Release);
    }

    /// Write `frames` output frames, blending across a resync if one is running.
    fn emit(&mut self, out: &mut [f32], frames: usize) {
        let mut fade = self.crossfade;
        let mut position = self.position;
        for index in 0..frames {
            let base = index * CHANNELS;
            match fade {
                Some(mut active) => {
                    let weight = active.played as f32 / RESYNC_CROSSFADE_FRAMES as f32;
                    for channel in 0..CHANNELS {
                        let from = sample_at(&self.ring, active.from, channel);
                        let to = sample_at(&self.ring, position, channel);
                        out[base + channel] = from + (to - from) * weight;
                    }
                    active.from += self.ratio;
                    active.played += 1;
                    fade = if active.played >= RESYNC_CROSSFADE_FRAMES {
                        None
                    } else {
                        Some(active)
                    };
                }
                None => {
                    for channel in 0..CHANNELS {
                        out[base + channel] = sample_at(&self.ring, position, channel);
                    }
                }
            }
            position += self.ratio;
        }
        self.position = position;
        self.crossfade = fade;
    }
}

/// The sink's [`Feed`]: a capture device whose blocks land in a ring the read trims to the
/// output clock.
pub(crate) struct CaptureFeed {
    reader: Reader,
    /// Kept for its lifetime: dropping it stops capture.
    stream: Option<cpal::Stream>,
    is_stopping: Arc<AtomicBool>,
    has_failed: Arc<AtomicBool>,
    name: String,
    stats: Arc<SinkStats>,
}

impl CaptureFeed {
    /// Open `name` for capture and start the stream.
    ///
    /// The caller resolves the name; a name that resolves to no device is an error here
    /// too, because recording the built-in microphone because a loopback was misspelled is
    /// not the stream the user asked for.
    pub(crate) fn open(name: &str) -> Result<Self> {
        let input = Input::open_named(Some(name))?;
        let ring = Arc::new(live::Ring::new(RING_FRAMES, CHANNELS));
        let head = Arc::new(AtomicUsize::new(0));
        let stats = Arc::new(SinkStats::default());
        let is_stopping = Arc::new(AtomicBool::new(false));
        let has_failed = Arc::new(AtomicBool::new(false));

        let writer = Arc::clone(&ring);
        let writer_head = Arc::clone(&head);
        let writer_stats = Arc::clone(&stats);
        let failed = Arc::clone(&has_failed);
        let stream = input
            .build_input_stream(
                move |block: &[f32]| write_capture(&writer, &writer_head, &writer_stats, block),
                move || failed.store(true, Ordering::Release),
            )
            .with_context(|| format!("capturing from \"{}\"", input.name()))?;

        Ok(Self {
            reader: Reader::new(ring, head, Arc::clone(&stats)),
            stream: Some(stream),
            is_stopping,
            has_failed,
            name: input.name().to_string(),
            stats,
        })
    }

    /// A feed over `ring` with no device behind it, for the tests.
    #[cfg(test)]
    fn without_device(
        ring: Arc<live::Ring>,
        head: Arc<AtomicUsize>,
        stats: Arc<SinkStats>,
    ) -> Self {
        Self {
            reader: Reader::new(ring, head, Arc::clone(&stats)),
            stream: None,
            is_stopping: Arc::new(AtomicBool::new(false)),
            has_failed: Arc::new(AtomicBool::new(false)),
            name: "a simulated capture".to_string(),
            stats,
        }
    }

    /// The name the device reported, for the status line and the panel.
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    /// The counters the status line and the panel read.
    pub(crate) fn stats(&self) -> Arc<SinkStats> {
        Arc::clone(&self.stats)
    }

    /// Set to end the feed at its next read. Dropping the stream stops capture; this stops
    /// the thread that is waiting for it.
    pub(crate) fn stopper(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.is_stopping)
    }
}

/// The capture callback: copy what fits into the ring and count the rest.
///
/// No allocation, no lock, no wait. A block that does not fit is truncated rather than
/// overwriting audio the read has not consumed, and the frames are counted so the panel
/// says so instead of a latency that grows until the ring tears.
pub(crate) fn write_capture(
    ring: &live::Ring,
    head: &AtomicUsize,
    stats: &SinkStats,
    block: &[f32],
) {
    let frames = block.len() / CHANNELS;
    if frames == 0 {
        return;
    }
    stats.record_peak(peak_of(block), frames);
    let written = ring.written();
    let read = head.load(Ordering::Acquire);
    let room = ring.capacity().saturating_sub(written.saturating_sub(read));
    let fit = frames.min(room);
    if fit > 0 {
        ring.write(written, &block[..fit * CHANNELS]);
    }
    if fit < frames {
        stats.record_dropped(frames - fit);
    }
}

impl Feed for CaptureFeed {
    fn channels(&self) -> usize {
        CHANNELS
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    fn read(&mut self, out: &mut [f32]) -> Result<usize> {
        loop {
            if self.is_stopping.load(Ordering::Acquire) {
                return Ok(0);
            }
            let frames = self.reader.read(out);
            if frames > 0 {
                return Ok(frames);
            }
            // A capture stream that failed is over: ending the feed is the truth, and
            // waiting would hold the playhead on a device that is gone.
            if self.has_failed.load(Ordering::Acquire) {
                return Ok(0);
            }
            thread::sleep(MIN_WAIT);
        }
    }
}

impl Drop for CaptureFeed {
    fn drop(&mut self) {
        self.is_stopping.store(true, Ordering::Release);
        // Taking the stream here stops capture now rather than when the callback next runs.
        self.stream.take();
    }
}

/// The band the capture has, measured on [`MEASURE_SECS`] of it.
///
/// The sink has no file to analyse, so it measures the stream: a preprocessor built for the
/// wrong band would either keep a top octave the source never had or throw away one it did.
/// A silent second is waited on, because the user may start their player a moment after the
/// command; [`SILENT_FRAMES`] of digital silence is not quiet audio, it is a capture that is
/// not carrying a signal, and it is named rather than played. A capture that is already
/// full-band has nothing for the model to add.
pub(crate) fn measure_cutoff(feed: &mut dyn Feed, device: &str) -> Result<u32> {
    let frames = MEASURE_SECS * SAMPLE_RATE as usize;
    let mut buffer = vec![0.0f32; frames * CHANNELS];
    let mut left = vec![0.0f32; frames];
    let mut silent = 0usize;
    loop {
        let mut filled = 0;
        while filled < frames {
            let read = feed.read(&mut buffer[filled * CHANNELS..])?;
            anyhow::ensure!(
                read > 0,
                "the capture ended after {filled} frames, before a second of it could be \
                 measured"
            );
            filled += read;
        }
        for frame in 0..frames {
            left[frame] = buffer[frame * CHANNELS];
        }
        if peak_of(&left) < SILENT_PEAK {
            silent += frames;
            anyhow::ensure!(
                silent < SILENT_FRAMES,
                "the capture from \"{device}\" has delivered {} s of {SILENT_HINT}",
                silent / SAMPLE_RATE as usize
            );
            continue;
        }
        let analysis = cutoff::analyse(&left)?;
        anyhow::ensure!(
            !analysis.looks_full_band(),
            "the capture is already full-band ({:.4}% of the spectrum above 21 kHz), so the \
             sink has nothing to generate; route a lossy source to it",
            analysis.high_band_ratio * 100.0
        );
        anyhow::ensure!(
            analysis.cutoff_hz < SAMPLE_RATE / 2,
            "the capture's band limit could not be measured (it reads as {} Hz); route lossy \
             source material to it",
            analysis.cutoff_hz
        );
        return Ok(analysis.cutoff_hz);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// A capture clock to simulate against, and the state the simulation needs.
    struct Sim {
        ring: Arc<live::Ring>,
        head: Arc<AtomicUsize>,
        stats: Arc<SinkStats>,
        reader: Reader,
        /// Capture frames produced so far, fractional so a small drift accumulates.
        produced: f64,
        /// Frames written to the ring, whole.
        written: usize,
    }

    impl Sim {
        fn new() -> Self {
            let ring = Arc::new(live::Ring::new(RING_FRAMES, CHANNELS));
            let head = Arc::new(AtomicUsize::new(0));
            let stats = Arc::new(SinkStats::default());
            let reader = Reader::new(Arc::clone(&ring), Arc::clone(&head), Arc::clone(&stats));
            Self {
                ring,
                head,
                stats,
                reader,
                produced: 0.0,
                written: 0,
            }
        }

        /// Write the capture frames this many output frames are worth at `clock_ppm`.
        fn produce(&mut self, output_frames: usize, clock_ppm: f64) {
            self.produced += output_frames as f64 * (1.0 + clock_ppm / 1_000_000.0);
            let target = self.produced as usize;
            let frames = target - self.written;
            if frames == 0 {
                return;
            }
            let block = tones(self.written, frames);
            write_capture(&self.ring, &self.head, &self.stats, &block);
            self.written = target;
        }

        fn read(&mut self, out: &mut [f32]) -> usize {
            self.reader.read(out)
        }

        fn fill(&self) -> usize {
            fill_of(self.ring.written(), self.reader.position)
        }
    }

    /// What one simulated stream did, as the numbers the report quotes.
    #[derive(Debug)]
    struct Run {
        frames: usize,
        min_fill: usize,
        max_fill: usize,
        peak_ppm: f64,
        resyncs: usize,
        dropped: usize,
        /// The largest step between adjacent output samples of one channel.
        max_step: f32,
    }

    impl Run {
        fn new() -> Self {
            Self {
                frames: 0,
                min_fill: usize::MAX,
                max_fill: 0,
                peak_ppm: 0.0,
                resyncs: 0,
                dropped: 0,
                max_step: 0.0,
            }
        }
    }

    /// A 997 Hz tone: not a multiple of the resync jump, so a jump without a crossfade
    /// lands on a different phase instead of a step that happens to line up.
    fn tone(frame: usize) -> f32 {
        let phase = frame as f64 * 997.0 / f64::from(SAMPLE_RATE) * std::f64::consts::TAU;
        (phase.sin() as f32) * 0.5
    }

    /// Interleaved stereo of [`tone`] for `frames`, starting at `start`.
    fn tones(start: usize, frames: usize) -> Vec<f32> {
        (start..start + frames)
            .flat_map(|frame| [tone(frame), tone(frame)])
            .collect()
    }

    /// Play `frames` output frames against a capture clock running `clock_ppm` fast.
    fn simulate(clock_ppm: f64, frames: usize) -> Run {
        let mut sim = Sim::new();
        let mut out = vec![0.0f32; READ_FRAMES * CHANNELS];
        let mut run = Run::new();
        let mut previous: [Option<f32>; CHANNELS] = [None; CHANNELS];
        while run.frames < frames {
            let want = (frames - run.frames).min(READ_FRAMES);
            sim.produce(want, clock_ppm);
            let read = sim.read(&mut out[..want * CHANNELS]);
            if read == 0 {
                continue;
            }
            for frame in 0..read {
                for channel in 0..CHANNELS {
                    let sample = out[frame * CHANNELS + channel];
                    if let Some(previous) = previous[channel] {
                        run.max_step = run.max_step.max((sample - previous).abs());
                    }
                    previous[channel] = Some(sample);
                }
            }
            let fill = sim.fill();
            run.min_fill = run.min_fill.min(fill);
            run.max_fill = run.max_fill.max(fill);
            run.frames += read;
        }
        run.peak_ppm = sim.stats.peak_correction_ppm();
        run.resyncs = sim.stats.resyncs();
        run.dropped = sim.stats.dropped_frames();
        run
    }

    #[test]
    fn given_a_ring_with_less_than_a_target_when_read_then_nothing_is_invented() {
        let mut sim = Sim::new();
        let mut out = vec![0.0f32; READ_FRAMES * CHANNELS];
        // Half a target: the pre-roll has not filled yet, so there is nothing to play.
        sim.produce(TARGET_FILL_FRAMES / 2, 0.0);
        assert_eq!(sim.read(&mut out), 0);
        assert_eq!(out[0], 0.0, "a read that returns nothing writes nothing");

        // A target's worth, and the read starts that far behind the writer.
        sim.produce(TARGET_FILL_FRAMES / 2, 0.0);
        assert_eq!(sim.read(&mut out), READ_FRAMES);
        assert_eq!(sim.fill(), TARGET_FILL_FRAMES - READ_FRAMES);
        assert_eq!(
            out[..8],
            tones(0, 4)[..8],
            "the read starts at the oldest frame of the pre-roll"
        );
    }

    #[test]
    fn given_a_constant_when_interpolated_then_every_fraction_returns_the_constant() {
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            assert_eq!(catmull_rom(0.5, 0.5, 0.5, 0.5, t), 0.5, "at {t}");
        }
    }

    #[test]
    fn given_a_linear_ramp_when_interpolated_then_the_fraction_is_exact() {
        // A trim of a few ppm reads between samples of a signal that is locally linear; an
        // interpolator that does not reproduce a line would bend it.
        for t in [0.0, 0.1, 0.5, 0.9, 1.0] {
            let value = catmull_rom(-1.0, 0.0, 1.0, 2.0, t);
            assert!((value - t).abs() < 1e-6, "{value} at {t}");
        }
    }

    #[test]
    fn given_an_integer_position_when_sampled_then_the_ring_sample_comes_out() {
        let ring = live::Ring::new(64, CHANNELS);
        let block: Vec<f32> = (0..32)
            .flat_map(|frame| [frame as f32, frame as f32])
            .collect();
        ring.write(0, &block);
        for frame in 1..20 {
            assert_eq!(sample_at(&ring, frame as f64, 0), frame as f32);
        }
    }

    #[test]
    fn given_a_fill_at_the_target_when_corrected_then_no_rate_change_is_applied() {
        assert_eq!(correction_for(TARGET_FILL_FRAMES, TARGET_FILL_FRAMES), 0.0);
        assert_eq!(correction_for(0, 0), 0.0);
    }

    #[test]
    fn given_a_full_or_empty_ring_when_corrected_then_the_clamp_holds() {
        assert_eq!(
            correction_for(RING_FRAMES, TARGET_FILL_FRAMES),
            MAX_CORRECTION_PPM
        );
        assert_eq!(correction_for(0, TARGET_FILL_FRAMES), -MAX_CORRECTION_PPM);
    }

    #[test]
    fn given_a_saturated_loop_at_an_edge_when_asked_then_it_resyncs_the_other_way() {
        let capacity = RING_FRAMES;
        assert_eq!(
            resync_for(capacity * 3 / 4, capacity, MAX_CORRECTION_PPM, true),
            Some(Resync::Skip(capacity / 4))
        );
        assert_eq!(
            resync_for(capacity / 4, capacity, -MAX_CORRECTION_PPM, true),
            Some(Resync::Repeat(capacity / 4))
        );
        // Inside the edges, or unsaturated at an edge, the loop is doing its job.
        assert_eq!(resync_for(capacity / 2, capacity, 0.0, true), None);
        assert_eq!(resync_for(capacity * 3 / 4, capacity, 899.0, true), None);
        assert_eq!(resync_for(capacity / 4, capacity, -899.0, true), None);
    }

    #[test]
    fn given_a_stopped_capture_at_the_low_edge_when_asked_then_it_is_left_to_starve() {
        // A stopped capture is not drift: repeating it would loop a few milliseconds
        // forever instead of letting the playhead run dry, which is the caller's to count.
        assert_eq!(resync_for(0, RING_FRAMES, -MAX_CORRECTION_PPM, true), None);
        assert_eq!(
            resync_for(RING_FRAMES / 4, RING_FRAMES, -MAX_CORRECTION_PPM, false),
            None
        );
        assert_eq!(
            resync_for(
                MIN_REPEAT_FRAMES - 1,
                RING_FRAMES,
                -MAX_CORRECTION_PPM,
                true
            ),
            None
        );
        assert!(resync_for(MIN_REPEAT_FRAMES, RING_FRAMES, -MAX_CORRECTION_PPM, true).is_some());
    }

    #[test]
    fn given_a_fifty_ppm_capture_clock_when_ten_minutes_play_then_the_ring_stays_bounded() {
        let minutes = 10;
        let run = simulate(50.0, minutes * 60 * SAMPLE_RATE as usize);
        println!(
            "50 ppm: fill {}..{} frames, peak {:.1} ppm, {} resyncs, {} dropped, max step {:.4}",
            run.min_fill, run.max_fill, run.peak_ppm, run.resyncs, run.dropped, run.max_step
        );
        assert_eq!(run.frames, minutes * 60 * SAMPLE_RATE as usize);
        assert!(
            run.min_fill > RING_FRAMES / 4 && run.max_fill < RING_FRAMES * 3 / 4,
            "the fill ran {}..{} frames of a {RING_FRAMES} frame ring",
            run.min_fill,
            run.max_fill
        );
        assert_eq!(run.resyncs, 0, "50 ppm is inside the correction's range");
        assert_eq!(
            run.dropped, 0,
            "a bounded fill must not drop capture frames"
        );
        assert!(
            run.peak_ppm < 200.0,
            "the loop applied {:.1} ppm for 50 ppm of drift",
            run.peak_ppm
        );
        assert!(
            run.max_step < 0.15,
            "the output stepped {:.3}",
            run.max_step
        );
    }

    #[test]
    fn given_a_five_thousand_ppm_capture_clock_when_played_then_it_resyncs_without_a_step() {
        // Past the clamp the ring would fill in half a minute; the resync is the answer,
        // and the crossfade is what keeps it quiet.
        let run = simulate(5_000.0, 2 * 60 * SAMPLE_RATE as usize);
        println!(
            "5000 ppm: fill {}..{} frames, {:.1} ppm peak, {} resyncs, {} dropped, max step {:.4}",
            run.min_fill, run.max_fill, run.peak_ppm, run.resyncs, run.dropped, run.max_step
        );
        assert!(
            run.resyncs >= 3,
            "2 minutes at 5000 ppm outruns the clamp: {} resyncs",
            run.resyncs
        );
        assert_eq!(run.dropped, 0, "the ring must stay inside its bound");
        assert!(
            run.max_step < 0.15,
            "a resync stepped {:.3}: the crossfade is not doing its job",
            run.max_step
        );
        assert!(
            run.max_fill < RING_FRAMES,
            "the fill reached {} frames of {RING_FRAMES}",
            run.max_fill
        );
    }

    #[test]
    fn given_a_step_in_the_capture_clock_when_the_loop_settles_then_it_does_not_overshoot() {
        // A step change is the one shape a proportional loop could ring on. This one is
        // first order, so the correction climbs and stops.
        let mut sim = Sim::new();
        let mut out = vec![0.0f32; READ_FRAMES * CHANNELS];
        let mut corrections = Vec::new();
        let step = 400.0;
        let frames = 8 * 60 * SAMPLE_RATE as usize;
        let mut played = 0;
        while played < frames {
            let clock = if played < frames / 2 { 0.0 } else { step };
            sim.produce(READ_FRAMES, clock);
            let read = sim.read(&mut out);
            if read == 0 {
                continue;
            }
            played += read;
            corrections.push(sim.stats.correction_ppm());
        }
        let after: Vec<f64> = corrections
            .into_iter()
            .skip_while(|ppm| *ppm == 0.0)
            .collect();
        let highest = after.iter().copied().fold(f64::MIN, f64::max);
        assert!(
            highest <= step + 1.0,
            "the loop overshot {highest:.1} ppm of a {step} ppm step"
        );
        // The writer adds whole frames, so the fill has a one-frame sawtooth: a couple of
        // ppm of wobble is quantisation, an overshoot would be tens.
        let quantum = 2.0 * FILL_GAIN_PPM_PER_FRAME;
        assert!(
            after.windows(2).all(|pair| pair[1] >= pair[0] - quantum),
            "the correction oscillated"
        );
        let settled = *after.last().expect("a correction after the step");
        assert!(
            (settled - step).abs() < step * 0.02,
            "settled at {settled:.1} ppm for a {step} ppm step"
        );
        assert!(
            sim.stats.peak_correction_ppm() < MAX_CORRECTION_PPM,
            "the clamp was reached on a {step} ppm step"
        );
    }

    #[test]
    fn given_a_capture_that_stops_when_the_feed_is_read_then_it_ends_rather_than_pads() {
        let ring = Arc::new(live::Ring::new(RING_FRAMES, CHANNELS));
        let head = Arc::new(AtomicUsize::new(0));
        let stats = Arc::new(SinkStats::default());
        let mut feed = CaptureFeed::without_device(ring, head, stats);
        let mut out = vec![0.0f32; READ_FRAMES * CHANNELS];

        // Nothing captured yet: the read waits, so it is stopped from another thread.
        let stopper = feed.stopper();
        let released = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            stopper.store(true, Ordering::Release);
        });
        assert_eq!(feed.read(&mut out).expect("a stopped feed"), 0);
        released.join().expect("the releasing thread");
    }

    #[test]
    fn given_a_capture_that_fails_mid_stream_when_read_then_the_frames_it_had_are_played() {
        let ring = Arc::new(live::Ring::new(RING_FRAMES, CHANNELS));
        let head = Arc::new(AtomicUsize::new(0));
        let stats = Arc::new(SinkStats::default());
        let mut feed =
            CaptureFeed::without_device(Arc::clone(&ring), Arc::clone(&head), Arc::clone(&stats));
        let mut out = vec![0.0f32; READ_FRAMES * CHANNELS];
        write_capture(&ring, &head, &stats, &tones(0, TARGET_FILL_FRAMES));
        feed.has_failed.store(true, Ordering::Release);

        let mut drained = 0;
        let mut reads = 0;
        loop {
            reads += 1;
            assert!(
                reads <= 64,
                "the feed kept repeating a capture that had stopped"
            );
            let read = feed.read(&mut out).expect("a failed feed");
            if read == 0 {
                break;
            }
            drained += read;
        }
        // The trim runs at most 1000 ppm, so buffered audio plays as slightly more or fewer
        // output frames; what it must not do is invent a second of it.
        let surplus = (drained as f64 - TARGET_FILL_FRAMES as f64).abs();
        assert!(
            surplus <= TARGET_FILL_FRAMES as f64 * MAX_CORRECTION_PPM / 1e6 + 4.0,
            "{drained} frames came out of {TARGET_FILL_FRAMES} captured"
        );
    }

    /// A feed that hands out fixed seconds of a scripted signal, a slice at a time like the
    /// capture feed, so the measurement has to accumulate.
    struct Scripted {
        seconds: Vec<Vec<f32>>,
        second: usize,
        offset: usize,
    }

    impl Scripted {
        fn new(seconds: Vec<Vec<f32>>) -> Self {
            Self {
                seconds,
                second: 0,
                offset: 0,
            }
        }
    }

    impl Feed for Scripted {
        fn channels(&self) -> usize {
            CHANNELS
        }

        fn sample_rate(&self) -> u32 {
            SAMPLE_RATE
        }

        fn read(&mut self, out: &mut [f32]) -> Result<usize> {
            let Some(data) = self.seconds.get(self.second) else {
                return Ok(0);
            };
            let remaining = data.len() / CHANNELS - self.offset;
            let frames = (out.len() / CHANNELS).min(remaining).min(READ_FRAMES);
            out[..frames * CHANNELS]
                .copy_from_slice(&data[self.offset * CHANNELS..(self.offset + frames) * CHANNELS]);
            self.offset += frames;
            if self.offset * CHANNELS == data.len() {
                self.second += 1;
                self.offset = 0;
            }
            Ok(frames)
        }
    }

    /// One second of interleaved silence.
    fn silence() -> Vec<f32> {
        vec![0.0f32; SAMPLE_RATE as usize * CHANNELS]
    }

    /// One second of the sines at `frequencies`, sharing full scale between them.
    fn sines(frequencies: &[f64]) -> Vec<f32> {
        let amplitude = 1.0 / frequencies.len() as f64;
        let frames = SAMPLE_RATE as usize;
        (0..frames)
            .flat_map(|frame| {
                let sample: f64 = frequencies
                    .iter()
                    .map(|hz| {
                        (frame as f64 * hz / f64::from(SAMPLE_RATE) * std::f64::consts::TAU).sin()
                            * amplitude
                    })
                    .sum();
                [sample as f32, sample as f32]
            })
            .collect()
    }

    #[test]
    fn given_a_silent_second_then_a_band_limited_one_when_measured_then_it_waits_and_reports() {
        // The user starts their player after the command: a silent second is not a verdict.
        let mut feed = Scripted::new(vec![silence(), sines(&[1_000.0, 8_000.0, 15_000.0])]);
        let cutoff = measure_cutoff(&mut feed, "BlackHole 2ch").expect("a measured band");
        assert!(
            (10_000..SAMPLE_RATE / 2).contains(&cutoff),
            "a 15 kHz source measured as {cutoff} Hz"
        );
    }

    #[test]
    fn given_five_seconds_of_digital_silence_when_measured_then_it_names_the_microphone() {
        // What a refused TCC permission looks like: no error, no prompt, every sample zero.
        let mut feed = Scripted::new(vec![silence(); SILENT_FRAMES / SAMPLE_RATE as usize]);
        let error = measure_cutoff(&mut feed, "BlackHole 2ch")
            .expect_err("digital silence is not quiet audio");
        let message = format!("{error:#}");
        assert!(message.contains("BlackHole 2ch"), "{message}");
        assert!(message.contains("5 s of digital silence"), "{message}");
        assert!(message.contains("Microphone"), "{message}");
        assert!(message.contains("Privacy & Security"), "{message}");
    }

    #[test]
    fn given_a_capture_of_digital_silence_when_read_then_the_state_says_silent() {
        // Five seconds of zeros is the state, and one block with signal ends it: the sink
        // can never report a running capture while it is being handed silence.
        let ring = Arc::new(live::Ring::new(RING_FRAMES, CHANNELS));
        let head = Arc::new(AtomicUsize::new(0));
        let stats = SinkStats::default();
        assert_eq!(stats.capture_state(), CaptureState::Signal);
        assert!(!stats.has_carried_signal());

        let silent = vec![0.0f32; RING_FRAMES * CHANNELS];
        let mut written = 0;
        while written < SILENT_FRAMES {
            write_capture(&ring, &head, &stats, &silent);
            written += RING_FRAMES;
        }
        assert_eq!(stats.silent_frames(), SILENT_FRAMES);
        assert_eq!(stats.capture_state(), CaptureState::Silent);
        assert!(
            !stats.has_carried_signal(),
            "a silent run never carried signal"
        );

        write_capture(&ring, &head, &stats, &tones(0, 256));
        assert_eq!(stats.capture_state(), CaptureState::Signal);
        assert_eq!(stats.silent_frames(), 0);
        assert!(stats.has_carried_signal());
    }

    #[test]
    fn given_a_full_band_capture_when_measured_then_it_is_refused_with_what_it_measured() {
        let mut feed = Scripted::new(vec![sines(&[1_000.0, 8_000.0, 15_000.0, 22_000.0])]);
        let error = measure_cutoff(&mut feed, "BlackHole 2ch")
            .expect_err("a full-band capture has nothing to add");
        let message = format!("{error:#}");
        assert!(message.contains("already full-band"), "{message}");
        assert!(message.contains("21 kHz"), "{message}");
    }

    #[test]
    fn given_a_capture_that_ends_when_measured_then_it_says_how_much_it_got() {
        let mut feed = Scripted::new(vec![vec![0.1f32; 480 * CHANNELS]]);
        let error = measure_cutoff(&mut feed, "BlackHole 2ch")
            .expect_err("a capture that ends cannot be measured");
        let message = format!("{error:#}");
        assert!(message.contains("480 frames"), "{message}");
    }

    #[test]
    fn given_a_capture_that_stalls_when_played_then_the_dry_side_plays_and_is_counted() {
        // F3's policy, reused: the sink's capture is the dry side, so a capture that stops
        // leaving the rings ahead of the playhead is a starved playhead, not a padded one.
        use crate::live::tests::test_policy;
        use crate::player::Ramp;

        const BUFFER: usize = 2_048;
        const WINDOW: usize = 1_024;
        let policy = test_policy(BUFFER, WINDOW);
        let ring = Arc::new(live::Ring::new(RING_FRAMES, CHANNELS));
        let head = Arc::new(AtomicUsize::new(0));
        let stats = Arc::new(SinkStats::default());
        // Four hundred milliseconds of capture and then nothing: the pump reads what the
        // ring holds and waits on a device that has stopped.
        // A constant, so a starved frame is told from a covered one without tracking which
        // capture frame the trim happens to be reading.
        let captured = vec![0.5f32; RING_FRAMES * CHANNELS];
        write_capture(&ring, &head, &stats, &captured);
        let feed =
            CaptureFeed::without_device(Arc::clone(&ring), Arc::clone(&head), Arc::clone(&stats));
        let stopper = feed.stopper();

        let (source, pump) = live::Live::start(
            Box::new(feed),
            &policy,
            Box::new(|dry: &[f32], wet: &mut [f32]| {
                for (out, sample) in wet.iter_mut().zip(dry) {
                    *out = sample + 1_000.0;
                }
                Ok(())
            }),
        )
        .expect("a live stream");

        let mut ramp = Ramp::new(true, SAMPLE_RATE);
        let mut out = vec![0.0f32; 512 * CHANNELS];
        let mut position = 0;
        let mut starved = 0;
        let deadline = Instant::now() + Duration::from_secs(10);
        while starved == 0 && Instant::now() < deadline {
            let fill = source.fill(&mut ramp, 1.0, position, &mut out);
            if fill.frames == 0 {
                thread::yield_now();
                continue;
            }
            for offset in 0..fill.frames {
                let sample = out[offset * CHANNELS];
                let is_wet = (sample - 1_000.5).abs() < 0.01;
                let is_dry = (sample - 0.5).abs() < 0.01;
                assert!(
                    is_wet || is_dry,
                    "frame {} played {sample}: the sink padded a starved fill",
                    position + offset
                );
                starved += usize::from(is_dry);
            }
            position += fill.frames;
        }
        assert!(
            starved > 0,
            "a stalled capture must starve the playhead, not pad it with silence"
        );
        assert_eq!(starved, source.stats().starved_frames());
        assert!(source.stats().dropouts() > 0);

        stopper.store(true, Ordering::Release);
        drop(source);
        pump.join().expect("the pump");
    }

    #[test]
    fn given_a_capture_ring_with_no_room_when_written_then_the_excess_is_counted_not_forced() {
        let ring = Arc::new(live::Ring::new(64, CHANNELS));
        let head = Arc::new(AtomicUsize::new(0));
        let stats = SinkStats::default();

        // The read is at frame 0, so the ring takes 64 frames and no more.
        write_capture(&ring, &head, &stats, &tones(0, 64));
        assert_eq!(ring.written(), 64);
        assert_eq!(stats.dropped_frames(), 0);

        write_capture(&ring, &head, &stats, &tones(64, 32));
        assert_eq!(ring.written(), 64, "a full ring must not be overwritten");
        assert_eq!(stats.dropped_frames(), 32);

        // The read frees room, and the callback resumes exactly there.
        head.store(16, Ordering::Release);
        write_capture(&ring, &head, &stats, &tones(64, 8));
        assert_eq!(ring.written(), 72);
        assert_eq!(stats.dropped_frames(), 32);
        assert_eq!(ring.sample(64, 0), tone(64));
    }
}
