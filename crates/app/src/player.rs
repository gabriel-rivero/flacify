//! How the A/B switch works: one position, two signals, a short ramp.
//!
//! The two signals are the same length and the same sample rate, so "sample-locked"
//! is not an alignment problem, it is a rule: every frame is read at the same index
//! from both sides and the choice only changes how much of each is mixed in. A
//! hard switch would click whenever the waveforms differ at the cut, so the mix
//! ramps over 8 ms, which is short enough to feel instant and long enough to hide the
//! step.
//!
//! Both sides arrive as whole buffers, from [`Reader`]'s decoded segments, or from
//! [`live::Live`]'s bounded rings, and the mixing loop is the same one for all three, so a
//! cache hit and a whole-track render produce identical samples.
//!
//! Nothing here allocates, locks or logs: `fill` is called from the audio callback.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::frontier::{Frontier, CHUNK_FRAMES};
use crate::live;
use crate::segments::Reader;

/// How long the switch takes, in seconds.
const RAMP_SECS: f32 = 0.008;

/// How far ahead of the playhead the frontier must be before automatic promotion, so the
/// ramp never runs into the edge of what exists.
const PROMOTE_MARGIN: usize = 2 * 48_000;

/// The parts of playback the UI touches. Atomics, because the audio thread reads them
/// without taking a lock.
pub struct Controls {
    /// Playhead in frames. Shared with a live stream's pump, which measures its credit
    /// against the same number.
    pub position: Arc<AtomicUsize>,
    pub playing: AtomicBool,
    /// True when the user wants the generated signal.
    pub use_enhanced: AtomicBool,
    /// Frames in the track. A live stream has no known length, so this is 0 there and
    /// [`Mixer::fill`] asks the source for the end instead.
    pub frames: usize,
    /// False when there is nothing to switch to (already full-band, or not rendered).
    has_wet_side: bool,
    /// The live counters, when the source is a live stream.
    stats: Option<Arc<live::Stats>>,
    /// Set while the app wants the switch to happen on its own, once.
    pub auto_promote: AtomicBool,
    /// True after that promotion has happened, so toggling back sticks.
    pub promoted: AtomicBool,
}

impl Controls {
    pub fn is_playing(&self) -> bool {
        self.playing.load(Ordering::Relaxed)
    }

    /// Whether there is a generated side to switch to right now. A live stream's wet side
    /// only counts once it covers the playhead, so the switch follows the buffer.
    pub fn has_enhanced(&self) -> bool {
        self.has_wet_side
            && self
                .stats
                .as_ref()
                .map_or(true, |stats| stats.is_wet_covered())
    }

    /// Why the switch cannot be used, in the words the status line prints.
    #[allow(dead_code)] // The live status line prints it (U1).
    pub fn enhanced_blocked(&self) -> Option<&'static str> {
        if !self.has_wet_side {
            return Some("the source has no generated side");
        }
        match &self.stats {
            Some(stats) if !stats.is_wet_covered() => Some(live::STARVED_REASON),
            _ => None,
        }
    }

    pub fn is_enhanced(&self) -> bool {
        self.has_enhanced() && self.use_enhanced.load(Ordering::Relaxed)
    }

    pub fn position_secs(&self, sample_rate: u32) -> f64 {
        self.position.load(Ordering::Relaxed) as f64 / f64::from(sample_rate)
    }

    /// Move the playhead. Clamped so a seek past the end lands on the last frame
    /// instead of stopping playback, which is what a scrubber at 100% should do.
    ///
    /// A live stream has no length and no seekable buffer, so its playhead does not move.
    pub fn seek_frames(&self, frame: usize) {
        if self.frames == 0 {
            return;
        }
        let last = self.frames.saturating_sub(1);
        self.position.store(frame.min(last), Ordering::Relaxed);
    }

    pub fn seek_secs(&self, seconds: f64, sample_rate: u32) {
        let frame = (seconds.max(0.0) * f64::from(sample_rate)) as usize;
        self.seek_frames(frame);
    }

    pub fn position_frames(&self) -> usize {
        self.position.load(Ordering::Relaxed)
    }

    pub fn toggle(&self) {
        if self.has_enhanced() {
            let next = !self.use_enhanced.load(Ordering::Relaxed);
            self.use_enhanced.store(next, Ordering::Relaxed);
        }
    }

    pub fn set_playing(&self, playing: bool) {
        self.playing.store(playing, Ordering::Relaxed);
    }

    /// Publish what the mixer is about to play from; a no-op on the file-shaped sources.
    fn set_wet_covered(&self, is_covered: bool) {
        if let Some(stats) = &self.stats {
            stats.set_wet_covered(is_covered);
        }
    }
}

/// The A/B switch's position and how far it moves per frame.
#[derive(Clone, Copy)]
pub(crate) struct Ramp {
    /// 0.0 plays the dry side, 1.0 the wet one.
    pub(crate) value: f32,
    /// Change per frame, from [`RAMP_SECS`].
    pub(crate) step: f32,
}

impl Ramp {
    pub(crate) fn new(is_enhanced: bool, sample_rate: u32) -> Self {
        Self {
            value: if is_enhanced { 1.0 } else { 0.0 },
            step: 1.0 / (RAMP_SECS * sample_rate.max(1) as f32),
        }
    }
}

/// Cheap to clone: the source is shared and only the ramp state is copied, which is what
/// lets the device open a second stream if the first buffer size is refused.
#[derive(Clone)]
/// Where playback reads its two signals.
enum Source {
    /// Whole buffers, interleaved at the device's rate: the original and the completed
    /// render when there is one.
    Whole {
        original: Arc<[f32]>,
        enhanced: Option<Arc<[f32]>>,
    },
    /// A whole-track original with the generated side still being rendered.
    Progressive {
        original: Arc<[f32]>,
        frontier: Arc<Frontier>,
    },
    /// A complete cache entry: both sides are decoded a segment at a time. The first play
    /// of an uncached track keeps [`Source::Progressive`]; replacing it is a later change.
    Segmented(Arc<Reader>),
    /// A feed being read while it plays: bounded rings, passthrough when the wet side has
    /// not caught up, no known length.
    #[allow(dead_code)] // Built by `play --live` (U1) and the sink (K1).
    Live(Arc<live::Live>),
}

#[derive(Clone)]
pub struct Mixer {
    source: Source,
    channels: usize,
    ramp: Ramp,
    controls: Arc<Controls>,
}

impl Mixer {
    /// Buffers are interleaved at the device's channel count and sample rate.
    pub fn new(
        original: Vec<f32>,
        enhanced: Option<Vec<f32>>,
        channels: usize,
        sample_rate: u32,
    ) -> (Self, Arc<Controls>) {
        let frames = match channels {
            0 => 0,
            _ => original.len() / channels,
        };
        let has_enhanced = enhanced.is_some();
        let controls = Arc::new(Controls {
            position: Arc::new(AtomicUsize::new(0)),
            playing: AtomicBool::new(false),
            use_enhanced: AtomicBool::new(has_enhanced),
            frames,
            has_wet_side: has_enhanced,
            stats: None,
            auto_promote: AtomicBool::new(false),
            promoted: AtomicBool::new(has_enhanced),
        });
        let mixer = Self {
            source: Source::Whole {
                original: original.into(),
                enhanced: enhanced.map(Into::into),
            },
            channels,
            ramp: Ramp::new(has_enhanced, sample_rate),
            controls: Arc::clone(&controls),
        };
        (mixer, controls)
    }

    /// Play a track whose generated side is still being rendered.
    ///
    /// The original plays immediately; `frontier` grows behind it. Promotion happens
    /// once, automatically, when the frontier is comfortably ahead of the playhead.
    pub fn progressive(
        original: Vec<f32>,
        frontier: Arc<Frontier>,
        channels: usize,
        sample_rate: u32,
    ) -> (Self, Arc<Controls>) {
        let frames = match channels {
            0 => 0,
            _ => original.len() / channels,
        };
        let controls = Arc::new(Controls {
            position: Arc::new(AtomicUsize::new(0)),
            playing: AtomicBool::new(false),
            use_enhanced: AtomicBool::new(false),
            frames,
            has_wet_side: true,
            stats: None,
            auto_promote: AtomicBool::new(true),
            promoted: AtomicBool::new(false),
        });
        let mixer = Self {
            source: Source::Progressive {
                original: original.into(),
                frontier,
            },
            channels,
            ramp: Ramp::new(false, sample_rate),
            controls: Arc::clone(&controls),
        };
        (mixer, controls)
    }

    /// Play a complete cache entry from its segments. Stereo only, because a segment is
    /// never resampled or remapped on the way to the device.
    pub fn segmented(reader: Arc<Reader>, sample_rate: u32) -> (Self, Arc<Controls>) {
        let has_enhanced = reader.has_rendered();
        let controls = Arc::new(Controls {
            position: Arc::new(AtomicUsize::new(0)),
            playing: AtomicBool::new(false),
            use_enhanced: AtomicBool::new(has_enhanced),
            frames: reader.frames(),
            has_wet_side: has_enhanced,
            stats: None,
            auto_promote: AtomicBool::new(false),
            promoted: AtomicBool::new(has_enhanced),
        });
        let mixer = Self {
            source: Source::Segmented(reader),
            channels: 2,
            ramp: Ramp::new(has_enhanced, sample_rate),
            controls: Arc::clone(&controls),
        };
        (mixer, controls)
    }

    /// Play a live feed: rings instead of a buffer, and the dry side wherever the wet side
    /// has not caught up.
    ///
    /// The stream has no known length, so `controls.frames` is 0 and the end comes from the
    /// rings. The switch starts on the dry side; the user asks for the generated one.
    #[allow(dead_code)] // Called by `play --live` (U1) and the sink (K1).
    pub fn live(source: Arc<live::Live>, sample_rate: u32) -> (Self, Arc<Controls>) {
        let channels = source.channels();
        let controls = Arc::new(Controls {
            position: source.playhead(),
            playing: AtomicBool::new(false),
            use_enhanced: AtomicBool::new(false),
            frames: 0,
            has_wet_side: true,
            stats: Some(source.stats()),
            auto_promote: AtomicBool::new(false),
            promoted: AtomicBool::new(false),
        });
        let mixer = Self {
            source: Source::Live(source),
            channels,
            ramp: Ramp::new(false, sample_rate),
            controls: Arc::clone(&controls),
        };
        (mixer, controls)
    }

    /// The whole-track original, for the window to draw. `None` on the segmented path,
    /// which holds no whole-track buffer.
    pub fn display_original(&self) -> Option<Arc<[f32]>> {
        match &self.source {
            Source::Whole { original, .. } | Source::Progressive { original, .. } => {
                Some(Arc::clone(original))
            }
            Source::Segmented(_) | Source::Live(_) => None,
        }
    }

    pub fn display_generated(&self) -> Option<Arc<[f32]>> {
        match &self.source {
            Source::Whole {
                enhanced: Some(buffer),
                ..
            } => Some(Arc::clone(buffer)),
            _ => None,
        }
    }

    /// How many frames of generated audio exist at or beyond the playhead.
    pub fn ready_frames(&self) -> usize {
        match &self.source {
            Source::Progressive { frontier, .. } => frontier.ready_frames(),
            Source::Live(live) => {
                let position = self.controls.position_frames();
                position + live.wet_ahead(position)
            }
            Source::Whole { .. } | Source::Segmented(_) => self.controls.frames,
        }
    }

    /// One automatic switch, then never again: after it, the toggle is the user's.
    fn promote_if_ready(&mut self) {
        if !self.controls.auto_promote.load(Ordering::Relaxed)
            || self.controls.promoted.load(Ordering::Relaxed)
        {
            return;
        }
        let position = self.controls.position.load(Ordering::Relaxed);
        if self.ready_frames() >= position + PROMOTE_MARGIN {
            self.controls.use_enhanced.store(true, Ordering::Relaxed);
            self.controls.promoted.store(true, Ordering::Relaxed);
        }
    }

    /// Fill an interleaved buffer. Returns how many frames carried audio; the rest of
    /// the buffer is silence, which is what a stopped or finished track should be.
    pub fn fill(&mut self, out: &mut [f32]) -> usize {
        let channels = self.channels.max(1);
        let frames = out.len() / channels;
        if !self.controls.playing.load(Ordering::Relaxed) {
            out.fill(0.0);
            return 0;
        }
        self.publish_wet_coverage(frames);
        self.promote_if_ready();

        let mut ramp = self.ramp;
        let mut position = self.controls.position.load(Ordering::Relaxed);
        let mut written = 0;

        while written < frames {
            if self.is_at_end(position) {
                self.controls.playing.store(false, Ordering::Relaxed);
                out[written * channels..].fill(0.0);
                break;
            }
            let target = if self.controls.is_enhanced() {
                1.0
            } else {
                0.0
            };
            // Never mix past the end of the track; the next iteration writes silence.
            let want = frames - written;

            match &self.source {
                // A whole track: one stretch, no segment bookkeeping.
                Source::Whole {
                    original,
                    enhanced: Some(enhanced),
                } => {
                    let span = want.min(self.controls.frames - position);
                    mix_stretch(
                        &mut ramp,
                        out,
                        written,
                        position,
                        span,
                        channels,
                        target,
                        |frame, channel| original[frame * channels + channel],
                        |frame, channel| enhanced[frame * channels + channel],
                    );
                    position += span;
                    written += span;
                }
                // Nothing generated: both sides are the original, so the toggle is a no-op.
                Source::Whole {
                    original,
                    enhanced: None,
                } => {
                    let span = want.min(self.controls.frames - position);
                    mix_stretch(
                        &mut ramp,
                        out,
                        written,
                        position,
                        span,
                        channels,
                        target,
                        |frame, channel| original[frame * channels + channel],
                        |frame, channel| original[frame * channels + channel],
                    );
                    position += span;
                    written += span;
                }
                // Progressive: at most one chunk per iteration, so the slot is loaded
                // once per second of audio instead of once per sample.
                Source::Progressive { original, frontier } => {
                    let (_, guard, chunk_end) = frontier.chunk_at(position);
                    let chunk = guard.as_ref();
                    let span = (chunk_end
                        .min(position + want.min(self.controls.frames - position)))
                    .saturating_sub(position)
                    .max(1);
                    mix_stretch(
                        &mut ramp,
                        out,
                        written,
                        position,
                        span,
                        channels,
                        target,
                        |frame, channel| original[frame * channels + channel],
                        |frame, channel| match chunk {
                            // `frame` is absolute; the chunk holds one second from its
                            // own start. Chunks are interleaved stereo, and only stereo
                            // devices take this path (the engine guarantees it).
                            Some(samples) => samples[(frame % CHUNK_FRAMES) * 2 + channel],
                            None => original[frame * channels + channel],
                        },
                    );
                    position += span;
                    written += span;
                }
                // Segmented: at most one decoded segment per iteration. The worker is
                // told where the playhead went. Until it catches up, the fill holds the
                // position and writes silence: it never repeats a stale frame and never
                // waits for the decode.
                Source::Segmented(reader) => {
                    reader.request(position);
                    let guard = reader.segment_at(position);
                    let Some(segment) = guard.as_ref() else {
                        out[written * channels..].fill(0.0);
                        break;
                    };
                    let start = segment.start();
                    let offset = position - start;
                    let span = (frames - written)
                        .min(segment.frames() - offset)
                        .min(self.controls.frames - position)
                        .max(1);
                    mix_stretch(
                        &mut ramp,
                        out,
                        written,
                        position,
                        span,
                        channels,
                        target,
                        |frame, channel| segment.dry(frame - start, channel),
                        |frame, channel| segment.wet(frame - start, channel),
                    );
                    position += span;
                    written += span;
                }
                // Live: one stretch from the rings, and the playhead holds still for
                // whatever the rings do not hold yet.
                Source::Live(live) => {
                    let fill =
                        live.fill(&mut ramp, target, position, &mut out[written * channels..]);
                    position += fill.frames;
                    written += fill.frames;
                    if fill.frames < want {
                        out[written * channels..].fill(0.0);
                        break;
                    }
                }
            }
        }

        self.ramp = ramp;
        self.controls.position.store(position, Ordering::Relaxed);
        written
    }

    /// Whether playback has reached the end: a file source knows its length, and a live
    /// stream ends when the feed has finished and the rings are drained.
    fn is_at_end(&self, position: usize) -> bool {
        match &self.source {
            Source::Live(live) => live.is_finished(position),
            _ => position >= self.controls.frames,
        }
    }

    /// Publish whether a live stream's wet side covers the next callback, so
    /// [`Controls::has_enhanced`] and the ramp answer for what the rings hold. A no-op on
    /// the file-shaped sources.
    fn publish_wet_coverage(&self, frames: usize) {
        let Source::Live(live) = &self.source else {
            return;
        };
        let position = self.controls.position_frames();
        self.controls
            .set_wet_covered(live.is_wet_covered(position, frames));
    }
}

/// Mix `want` frames starting at the playhead into `out[written..]`.
///
/// Free function rather than a method: the caller holds an immutable borrow of the
/// source while mixing, and only the ramp is mutable state. Both sides are closures so
/// a whole buffer and a decoded segment fill the loop with the same arithmetic.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mix_stretch(
    ramp: &mut Ramp,
    out: &mut [f32],
    written: usize,
    position: usize,
    want: usize,
    channels: usize,
    target: f32,
    dry: impl Fn(usize, usize) -> f32,
    wet: impl Fn(usize, usize) -> f32,
) {
    for index in 0..want {
        let delta = (target - ramp.value).clamp(-ramp.step, ramp.step);
        ramp.value += delta;
        let frame = position + index;
        for channel in 0..channels {
            let dry = dry(frame, channel);
            let generated = if ramp.value > 0.0 {
                wet(frame, channel)
            } else {
                dry
            };
            out[(written + index) * channels + channel] = dry + (generated - dry) * ramp.value;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;
    use crate::live::tests::{dry_sample, test_policy, RampFeed};
    use crate::segments::{self, MemorySource, Resident};

    /// The segment length the segmented tests use: small enough to cross a boundary in a
    /// few thousand frames, so a seek lands in another segment.
    const SEGMENT: usize = 10_000;

    /// The two sides of one synthetic track, as whole interleaved buffers.
    fn whole(frames: usize) -> (Vec<f32>, Vec<f32>) {
        (
            segments::memory_interleaved(frames, 0.0, 2),
            segments::memory_interleaved(frames, 1_000.0, 2),
        )
    }

    fn bits(samples: &[f32]) -> Vec<u32> {
        samples.iter().map(|sample| sample.to_bits()).collect()
    }

    /// A mixer over segments, plus the reader behind it, at the same track as [`whole`].
    fn segmented(frames: usize) -> (Mixer, Arc<Controls>, Arc<segments::Reader>) {
        let reader = segments::Reader::open(
            Arc::new(MemorySource::new(frames, SEGMENT, 0.0)),
            Some(Arc::new(MemorySource::new(frames, SEGMENT, 1_000.0))),
        )
        .expect("a reader");
        let (mixer, controls) = Mixer::segmented(Arc::clone(&reader), 48_000);
        (mixer, controls, reader)
    }

    /// Fill once before measuring: the first guard load on a thread initialises
    /// `arc-swap`'s per-thread debt list, which is not part of the callback's steady state.
    fn warm_up(mixer: &mut Mixer, out: &mut [f32]) {
        mixer.fill(out);
    }

    /// Move the playhead and let the worker catch up, so a fill is not comparing a decoded
    /// segment with a prefetch that is still running.
    fn seek(frame: usize, controls: &Arc<Controls>, reader: &segments::Reader) {
        controls.seek_frames(frame);
        reader.request(frame);
        assert!(
            reader.wait_until_ready(frame, Duration::from_secs(5)),
            "the segment at {frame} must be decoded before the comparison"
        );
    }

    #[test]
    fn given_the_same_audio_as_a_whole_buffer_and_as_segments_when_mixed_then_the_samples_are_identical(
    ) {
        let frames = 3 * SEGMENT + 1_234;
        let (mixer_original, mixer_enhanced) = whole(frames);
        let (mut whole_mixer, whole_controls) =
            Mixer::new(mixer_original, Some(mixer_enhanced), 2, 48_000);
        let (mut segment_mixer, segment_controls, reader) = segmented(frames);
        whole_controls.set_playing(true);
        segment_controls.set_playing(true);

        // Two segments' worth from the start: the stretch crosses the boundary at 10 000,
        // which is the only place a segment-backed mixer could disagree.
        let mut whole_out = vec![0.0f32; 2 * SEGMENT * 2];
        let mut segment_out = vec![0.0f32; 2 * SEGMENT * 2];
        assert_eq!(
            whole_mixer.fill(&mut whole_out),
            segment_mixer.fill(&mut segment_out)
        );
        assert_eq!(
            bits(&whole_out),
            bits(&segment_out),
            "the first stretch crosses a segment boundary"
        );

        // The toggle and its 8 ms ramp must behave the same on both.
        whole_controls.toggle();
        segment_controls.toggle();
        seek(2 * SEGMENT, &segment_controls, &reader);
        whole_controls.seek_frames(2 * SEGMENT);
        let mut whole_back = vec![0.0f32; 5_000 * 2];
        let mut segment_back = vec![0.0f32; 5_000 * 2];
        whole_mixer.fill(&mut whole_back);
        segment_mixer.fill(&mut segment_back);
        assert_eq!(bits(&whole_back), bits(&segment_back), "the dry ramp");

        // A seek into the last segment: the next fill must read that segment's samples.
        seek(2 * SEGMENT + 17, &segment_controls, &reader);
        whole_controls.seek_frames(2 * SEGMENT + 17);
        let mut whole_seek = vec![0.0f32; 7_000 * 2];
        let mut segment_seek = vec![0.0f32; 7_000 * 2];
        whole_mixer.fill(&mut whole_seek);
        segment_mixer.fill(&mut segment_seek);
        assert_eq!(
            bits(&whole_seek),
            bits(&segment_seek),
            "a seeked fill must read the segment it landed in"
        );
        assert_eq!(
            segment_controls.position_frames(),
            whole_controls.position_frames()
        );
    }

    #[test]
    fn given_a_seek_into_another_segment_when_filled_then_the_previous_segment_is_dropped() {
        let (mut mixer, controls, reader) = segmented(8 * SEGMENT);
        controls.set_playing(true);
        let mut out = vec![0.0f32; 1_000 * 2];
        mixer.fill(&mut out);
        assert_eq!(reader.resident_indexes(), [0, 1]);

        seek(5 * SEGMENT, &controls, &reader);
        mixer.fill(&mut out);

        assert_eq!(reader.resident_indexes(), [5, 6]);
        assert!(
            reader.segment_at(0).as_ref().is_none(),
            "the segment at the old playhead must be gone"
        );
    }

    #[test]
    fn given_a_segment_with_no_rendered_side_when_filled_then_it_emits_the_original_and_allocates_nothing(
    ) {
        let frames = SEGMENT;
        let dry: Vec<f32> = {
            let mut planar = Vec::with_capacity(frames * 2);
            for channel in 0..2 {
                for frame in 0..frames {
                    planar.push(segments::memory_sample(frame, channel, 0.0));
                }
            }
            planar
        };
        let resident = Resident::new(0, 0, frames, Arc::from(dry), None);
        let reader = segments::Reader::with_residents(
            Arc::new(MemorySource::new(frames, SEGMENT, 0.0)),
            Some(Arc::new(MemorySource::new(frames, SEGMENT, 1_000.0))),
            vec![resident],
        );
        let (mut mixer, controls) = Mixer::segmented(Arc::clone(&reader), 48_000);
        controls.set_playing(true);
        // The ramp is at 1.0 because the entry has a rendered side, so a missing segment
        // is the only reason a sample would not be the rendered one.
        assert!(controls.is_enhanced());

        let mut out = vec![0.0f32; 4_000 * 2];
        let (written, allocated) = crate::alloc_counter::measure(|| mixer.fill(&mut out));
        assert_eq!(written, 4_000);
        assert_eq!(allocated, 0, "the callback must not allocate");
        for frame in 0..4_000 {
            for channel in 0..2 {
                assert_eq!(
                    out[frame * 2 + channel],
                    segments::memory_sample(frame, channel, 0.0),
                    "frame {frame} channel {channel} must be the original"
                );
            }
        }
    }

    #[test]
    fn given_a_resident_segment_when_filled_then_the_callback_allocates_nothing() {
        let (mut mixer, controls, reader) = segmented(6 * SEGMENT);
        controls.set_playing(true);
        let mut out = vec![0.0f32; 4_000 * 2];
        warm_up(&mut mixer, &mut out);
        seek(0, &controls, &reader);
        let (written, allocated) = crate::alloc_counter::measure(|| mixer.fill(&mut out));
        assert_eq!(written, 4_000);
        assert_eq!(allocated, 0, "the callback must not allocate");
    }

    #[test]
    fn given_a_playhead_further_than_the_prefetch_when_filled_then_it_holds_and_writes_silence() {
        // No worker at all: nothing is resident, which is what a seek past the prefetch
        // looks like to the callback before the worker catches up.
        let reader = segments::Reader::with_residents(
            Arc::new(MemorySource::new(8 * SEGMENT, SEGMENT, 0.0)),
            None,
            Vec::new(),
        );
        let (mut mixer, controls) = Mixer::segmented(Arc::clone(&reader), 48_000);
        controls.set_playing(true);
        assert!(
            !controls.has_enhanced(),
            "a reader with no rendered side has nothing to switch to"
        );
        controls.seek_frames(7 * SEGMENT);

        let mut out = vec![1.0f32; 1_000 * 2];
        warm_up(&mut mixer, &mut out);
        out.fill(1.0);
        let (written, allocated) = crate::alloc_counter::measure(|| mixer.fill(&mut out));
        assert_eq!(written, 0, "the fill must hold rather than block");
        assert_eq!(allocated, 0, "the callback must not allocate");
        assert!(out.iter().all(|sample| *sample == 0.0));
        assert_eq!(
            controls.position_frames(),
            7 * SEGMENT,
            "the playhead must not advance over frames it could not read"
        );
    }

    fn ramp_track(frames: usize, offset: f32) -> Vec<f32> {
        (0..frames)
            .flat_map(|i| [i as f32 + offset, i as f32 + offset])
            .collect()
    }

    #[test]
    fn given_a_stopped_track_when_filled_then_the_buffer_is_silence() {
        let (mut mixer, controls) = Mixer::new(ramp_track(64, 0.0), None, 2, 48_000);
        let mut out = vec![1.0f32; 32];
        assert_eq!(mixer.fill(&mut out), 0);
        assert!(out.iter().all(|sample| *sample == 0.0));
        assert!(!controls.is_playing());
    }

    #[test]
    fn given_playback_when_filled_then_the_position_advances_by_the_frames_written() {
        let (mut mixer, controls) = Mixer::new(ramp_track(64, 0.0), None, 2, 48_000);
        controls.set_playing(true);
        let mut out = vec![0.0f32; 20];
        assert_eq!(mixer.fill(&mut out), 10);
        assert_eq!(controls.position.load(Ordering::Relaxed), 10);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[18], 9.0);
    }

    #[test]
    fn given_a_toggle_mid_playback_when_the_ramp_finishes_then_frames_come_from_the_original_buffer(
    ) {
        let frames = 4000;
        let (mut mixer, controls) = Mixer::new(
            ramp_track(frames, 0.0),
            Some(ramp_track(frames, 1000.0)),
            2,
            48_000,
        );
        controls.set_playing(true);

        let mut first = vec![0.0f32; 1000 * 2];
        mixer.fill(&mut first);
        // Enhanced is the default when it exists: frame 999 must be 999 + 1000.
        assert!((first[999 * 2] - 1999.0).abs() < 1.0);

        controls.toggle(); // back to the original
        let mut second = vec![0.0f32; 1000 * 2];
        mixer.fill(&mut second);
        // The switch starts at frame 1000, so the first frame is still wet...
        assert!((second[0] - 2000.0).abs() < 50.0);
        // ...and by frame 1999 the ramp is done and the value is the dry buffer's.
        assert!((second[999 * 2] - 1999.0).abs() < 1.0);
    }

    #[test]
    fn given_playback_when_seeking_then_the_next_fill_reads_from_the_new_position() {
        let (mut mixer, controls) = Mixer::new(ramp_track(64, 0.0), None, 2, 48_000);
        controls.set_playing(true);
        controls.seek_frames(40);

        let mut out = vec![0.0f32; 8];
        assert_eq!(mixer.fill(&mut out), 4);
        assert_eq!(out[0], 40.0);
        assert_eq!(controls.position_frames(), 44);
    }

    #[test]
    fn given_a_position_past_the_end_when_seeking_then_it_lands_on_the_last_frame() {
        let (_mixer, controls) = Mixer::new(ramp_track(64, 0.0), None, 2, 48_000);
        controls.seek_frames(10_000);
        assert_eq!(controls.position_frames(), 63);
    }

    #[test]
    fn given_a_frontier_when_frames_pass_it_then_generated_audio_stops_at_the_frontier() {
        let frames = 2 * CHUNK_FRAMES;
        let frontier = Frontier::new(frames);
        // One second of full-scale generated audio, silence as the original.
        frontier.publish(0, vec![1.0f32; CHUNK_FRAMES * 2]);
        let (mut mixer, controls) =
            Mixer::progressive(vec![0.0f32; frames * 2], Arc::clone(&frontier), 2, 48_000);
        controls.set_playing(true);
        controls.use_enhanced.store(true, Ordering::Relaxed);
        controls.promoted.store(true, Ordering::Relaxed);

        let mut out = vec![0.0f32; CHUNK_FRAMES * 2];
        mixer.fill(&mut out);
        // The ramp takes 8 ms; by the end of the chunk the mix is fully generated.
        assert!(
            (out[CHUNK_FRAMES * 2 - 2] - 1.0).abs() < 1e-3,
            "{}",
            out[CHUNK_FRAMES * 2 - 2]
        );

        // Past the frontier there is nothing generated, so the mix is the original again.
        let mut beyond = vec![0.0f32; 4_800];
        mixer.fill(&mut beyond);
        assert!(beyond.iter().all(|sample| sample.abs() < 1e-3));
    }

    #[test]
    fn given_a_frontier_ahead_of_the_playhead_when_played_then_it_promotes_once_and_stays_put() {
        let frames = 8 * CHUNK_FRAMES;
        let frontier = Frontier::new(frames);
        for index in 0..4 {
            frontier.publish(index, vec![0.5f32; CHUNK_FRAMES * 2]);
        }
        let (mut mixer, controls) =
            Mixer::progressive(vec![0.0f32; frames * 2], Arc::clone(&frontier), 2, 48_000);
        controls.set_playing(true);

        let mut out = vec![0.0f32; 9_600];
        mixer.fill(&mut out);
        assert!(controls.is_enhanced(), "four seconds ready should promote");
        assert!(controls.promoted.load(Ordering::Relaxed));

        // The user turns it off; the automatic promotion must not fight them.
        controls.toggle();
        assert!(!controls.is_enhanced());
        let mut again = vec![0.0f32; 9_600];
        mixer.fill(&mut again);
        assert!(!controls.is_enhanced(), "promotion must happen only once");
    }

    #[test]
    fn given_a_playhead_inside_a_chunk_when_filled_then_it_reads_the_right_second() {
        // A fill that starts mid-chunk used to index the chunk with an absolute frame
        // number and run off the end. Two chunks, playhead at 10,000 frames, values
        // chosen so a wrong offset is visible rather than merely out of bounds.
        let frames = 3 * CHUNK_FRAMES;
        let frontier = Frontier::new(frames);
        let mut first = vec![0.0f32; CHUNK_FRAMES * 2];
        for frame in 0..CHUNK_FRAMES {
            first[frame * 2] = frame as f32;
            first[frame * 2 + 1] = frame as f32;
        }
        frontier.publish(0, first);
        frontier.publish(1, vec![0.0f32; CHUNK_FRAMES * 2]);

        let (mut mixer, controls) =
            Mixer::progressive(vec![0.0f32; frames * 2], Arc::clone(&frontier), 2, 48_000);
        controls.set_playing(true);
        controls.use_enhanced.store(true, Ordering::Relaxed);
        controls.promoted.store(true, Ordering::Relaxed);
        controls.seek_frames(10_000);

        let mut out = vec![0.0f32; 2_048];
        mixer.fill(&mut out);
        // The ramp is 8 ms; by frame 1000 the mix should be the chunk's own sample.
        assert!(
            (out[1_000 * 2] - (10_000.0 + 1_000.0)).abs() < 1.0,
            "{}",
            out[1_000 * 2]
        );
    }

    #[test]
    fn given_no_enhanced_buffer_when_toggled_then_nothing_changes() {
        let (mut mixer, controls) = Mixer::new(ramp_track(32, 0.0), None, 2, 48_000);
        controls.set_playing(true);
        controls.toggle();
        assert!(!controls.is_enhanced());
        let mut out = vec![0.0f32; 8];
        mixer.fill(&mut out);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn given_the_end_of_the_track_when_filled_then_playback_stops_and_the_rest_is_silence() {
        let (mut mixer, controls) = Mixer::new(ramp_track(4, 0.0), None, 2, 48_000);
        controls.set_playing(true);
        let mut out = vec![1.0f32; 40];
        let written = mixer.fill(&mut out);
        assert_eq!(written, 4);
        assert!(!controls.is_playing());
        assert!(out[8..].iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn given_the_ramp_when_it_runs_then_it_is_monotonic_and_lands_exactly_on_the_target() {
        // Dry is silence, wet is full scale, so the mixed sample *is* the ramp.
        let frames = 2000;
        let (mut mixer, controls) = Mixer::new(
            vec![0.0f32; frames * 2],
            Some(vec![1.0f32; frames * 2]),
            2,
            48_000,
        );
        controls.set_playing(true);
        controls.toggle(); // wet -> dry

        let mut out = vec![0.0f32; frames * 2];
        mixer.fill(&mut out);
        let mut previous = 1.0;
        for frame in 0..frames {
            let value = out[frame * 2];
            assert!(value <= previous + 1e-6, "ramp went backwards at {frame}");
            assert!((0.0..=1.0).contains(&value));
            previous = value;
        }
        assert!(previous.abs() < 1e-3, "ramp ended at {previous}");
    }

    /// A live mixer over a feed of `frames` frames. The wet side is the dry side plus 1 000,
    /// so a mix is visible in the output. The pump is left running: dropping the mixer stops
    /// it.
    fn live_mixer(
        buffer_frames: usize,
        window_frames: usize,
        frames: usize,
    ) -> (Mixer, Arc<Controls>, Arc<live::Live>) {
        let (live, _pump) = live::Live::start(
            Box::new(RampFeed::new(frames)),
            &test_policy(buffer_frames, window_frames),
            Box::new(|dry: &[f32], wet: &mut [f32]| {
                wet.copy_from_slice(&dry[..wet.len()]);
                for sample in wet.iter_mut() {
                    *sample += 1_000.0;
                }
                Ok(())
            }),
        )
        .expect("a live stream");
        let (mixer, controls) = Mixer::live(Arc::clone(&live), 48_000);
        (mixer, controls, live)
    }

    #[test]
    fn given_a_live_source_when_the_callback_fills_then_it_allocates_nothing() {
        let (mut mixer, controls, live) = live_mixer(4_096, 2_048, 64 * 1_024);
        assert!(
            live.wait_for_wet(0, 1_024, Duration::from_secs(5)),
            "the prebuffer must fill"
        );
        controls.set_playing(true);

        let mut out = vec![0.0f32; 1_024 * 2];
        warm_up(&mut mixer, &mut out);
        let (written, allocated) = crate::alloc_counter::measure(|| mixer.fill(&mut out));
        assert_eq!(written, 1_024);
        assert_eq!(allocated, 0, "the callback must not allocate");
    }

    #[test]
    fn given_a_live_playhead_when_the_wet_side_does_not_cover_the_callback_then_the_switch_has_a_reason(
    ) {
        // The wet ring holds 2 048 frames, so a callback of 4 096 can never be covered: the
        // switch must refuse and say why, and the dry side must play in full.
        let (mut mixer, controls, live) = live_mixer(2_048, 1_024, 64 * 1_024);
        assert!(live.wait_for_wet(0, 1_024, Duration::from_secs(5)));
        controls.set_playing(true);

        let mut out = vec![1.0f32; 4_096 * 2];
        let written = mixer.fill(&mut out);
        assert!(written > 0);
        assert!(!controls.has_enhanced());
        assert_eq!(controls.enhanced_blocked(), Some(live::STARVED_REASON));
        controls.toggle();
        assert!(
            !controls.is_enhanced(),
            "the switch must refuse while the wet side is behind"
        );
        for frame in 0..written {
            assert_eq!(out[frame * 2], dry_sample(frame, 0), "frame {frame}");
            assert_eq!(out[frame * 2 + 1], dry_sample(frame, 1), "frame {frame}");
        }
        assert!(
            out[written * 2..].iter().all(|sample| *sample == 0.0),
            "what the rings do not hold is silence, not a repeat"
        );
    }

    #[test]
    fn given_a_live_source_when_the_prebuffer_is_full_then_the_switch_plays_the_wet_side() {
        let (mut mixer, controls, live) = live_mixer(4_096, 2_048, 64 * 1_024);
        assert!(live.wait_for_wet(0, 1_024, Duration::from_secs(5)));
        controls.set_playing(true);

        let mut out = vec![0.0f32; 1_024 * 2];
        mixer.fill(&mut out);
        assert!(controls.has_enhanced(), "the buffer covers the callback");
        assert_eq!(controls.enhanced_blocked(), None);
        controls.toggle();
        assert!(controls.is_enhanced());

        let mut wet_out = vec![0.0f32; 1_024 * 2];
        mixer.fill(&mut wet_out);
        // The ramp takes 8 ms, so by the end of the buffer the sample is the wet side's.
        let frame = controls.position_frames() - 1;
        let last = (1_024 - 1) * 2;
        assert!(
            (wet_out[last] - (dry_sample(frame, 0) + 1_000.0)).abs() < 0.01,
            "{} is not the wet side at {frame}",
            wet_out[last]
        );
    }

    #[test]
    fn given_the_end_of_a_live_feed_when_filled_then_playback_stops_at_the_last_frame() {
        let frames = 8 * 1_024;
        let (mut mixer, controls, _live) = live_mixer(4_096, 2_048, frames);
        controls.set_playing(true);

        let mut out = vec![1.0f32; 1_024 * 2];
        let mut delivered = 0;
        let deadline = Instant::now() + Duration::from_secs(10);
        while controls.is_playing() && Instant::now() < deadline {
            let written = mixer.fill(&mut out);
            delivered += written;
            if written == 0 {
                std::thread::yield_now();
            }
        }
        assert_eq!(delivered, frames, "every frame of the feed must be played");
        assert!(
            !controls.is_playing(),
            "the fill must report the end rather than stall"
        );
        assert_eq!(controls.position_frames(), frames);
    }
}
