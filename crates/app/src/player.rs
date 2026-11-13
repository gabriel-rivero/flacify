//! What the A/B switch actually does: one position, two buffers, a short ramp.
//!
//! The two signals are the same length and the same sample rate, so "sample-locked"
//! is not an alignment problem, it is a rule: every frame is read at the same index
//! from both buffers and the choice only changes how much of each is mixed in. A
//! hard switch would click whenever the waveforms differ at the cut, so the mix
//! ramps over 8 ms, which is short enough to feel instant and long enough to hide the
//! step.
//!
//! Nothing here allocates, locks or logs: `fill` is called from the audio callback.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::frontier::{Frontier, CHUNK_FRAMES};

/// How long the switch takes, in seconds.
const RAMP_SECS: f32 = 0.008;

/// How far ahead of the playhead the frontier must be before automatic promotion, so the
/// ramp never runs into the edge of what exists.
const PROMOTE_MARGIN: usize = 2 * 48_000;

/// The parts of playback the UI touches. Atomics, because the audio thread reads them
/// without taking a lock.
pub struct Controls {
    /// Playhead in frames.
    pub position: AtomicUsize,
    pub playing: AtomicBool,
    /// True when the user wants the generated signal.
    pub use_enhanced: AtomicBool,
    pub frames: usize,
    /// False when there is nothing to switch to (already full-band, or not rendered).
    pub has_enhanced: bool,
    /// Set while the app wants the switch to happen on its own, once.
    pub auto_promote: AtomicBool,
    /// True after that promotion has happened, so toggling back sticks.
    pub promoted: AtomicBool,
}

impl Controls {
    pub fn is_playing(&self) -> bool {
        self.playing.load(Ordering::Relaxed)
    }

    pub fn is_enhanced(&self) -> bool {
        self.has_enhanced && self.use_enhanced.load(Ordering::Relaxed)
    }

    pub fn position_secs(&self, sample_rate: u32) -> f64 {
        self.position.load(Ordering::Relaxed) as f64 / f64::from(sample_rate)
    }

    /// Move the playhead. Clamped so a seek past the end lands on the last frame
    /// instead of stopping playback, which is what a scrubber at 100% should do.
    pub fn seek_frames(&self, frame: usize) {
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
        if self.has_enhanced {
            let next = !self.use_enhanced.load(Ordering::Relaxed);
            self.use_enhanced.store(next, Ordering::Relaxed);
        }
    }

    pub fn set_playing(&self, playing: bool) {
        self.playing.store(playing, Ordering::Relaxed);
    }
}

/// Cheap to clone: the buffers are shared and only the ramp state is copied, which is
/// what lets the device open a second stream if the first buffer size is refused.
#[derive(Clone)]
/// Where the generated signal comes from.
enum Wet {
    /// A completed track, same layout and length as the original.
    Full(Arc<[f32]>),
    /// Chunks published by the renderer as they finish; interleaved stereo at 48 kHz.
    Frontier(Arc<Frontier>),
    None,
}

#[derive(Clone)]
pub struct Mixer {
    original: Arc<[f32]>,
    wet: Wet,
    channels: usize,
    ramp: f32,
    ramp_step: f32,
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
            position: AtomicUsize::new(0),
            playing: AtomicBool::new(false),
            use_enhanced: AtomicBool::new(has_enhanced),
            frames,
            has_enhanced,
            auto_promote: AtomicBool::new(false),
            promoted: AtomicBool::new(has_enhanced),
        });
        let mixer = Self {
            original: original.into(),
            wet: match enhanced {
                Some(buffer) => Wet::Full(buffer.into()),
                None => Wet::None,
            },
            channels,
            ramp: if has_enhanced { 1.0 } else { 0.0 },
            ramp_step: 1.0 / (RAMP_SECS * sample_rate.max(1) as f32),
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
            position: AtomicUsize::new(0),
            playing: AtomicBool::new(false),
            use_enhanced: AtomicBool::new(false),
            frames,
            has_enhanced: true,
            auto_promote: AtomicBool::new(true),
            promoted: AtomicBool::new(false),
        });
        let mixer = Self {
            original: original.into(),
            wet: Wet::Frontier(frontier),
            channels,
            ramp: 0.0,
            ramp_step: 1.0 / (RAMP_SECS * sample_rate.max(1) as f32),
            controls: Arc::clone(&controls),
        };
        (mixer, controls)
    }

    /// How many frames of generated audio exist at or beyond the playhead.
    pub fn ready_frames(&self) -> usize {
        match &self.wet {
            Wet::Full(_) => self.controls.frames,
            Wet::Frontier(frontier) => frontier.ready_frames(),
            Wet::None => 0,
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
        self.promote_if_ready();

        let mut ramp = self.ramp;
        let ramp_step = self.ramp_step;
        let mut position = self.controls.position.load(Ordering::Relaxed);
        let mut written = 0;

        while written < frames {
            if position >= self.controls.frames {
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
            let want = (frames - written).min(self.controls.frames - position);

            match &self.wet {
                // A finished track: one stretch, no chunk bookkeeping.
                Wet::Full(buffer) => {
                    mix_stretch(
                        &mut ramp,
                        ramp_step,
                        &self.original,
                        out,
                        written,
                        position,
                        want,
                        channels,
                        target,
                        |frame, channel| buffer[frame * channels + channel],
                    );
                    position += want;
                    written += want;
                }
                // Progressive: at most one chunk per iteration, so the slot is loaded
                // once per second of audio instead of once per sample.
                Wet::Frontier(frontier) => {
                    let (_, guard, chunk_end) = frontier.chunk_at(position);
                    let chunk = guard.as_ref();
                    let span = (chunk_end.min(position + want))
                        .saturating_sub(position)
                        .max(1);
                    mix_stretch(
                        &mut ramp,
                        ramp_step,
                        &self.original,
                        out,
                        written,
                        position,
                        span,
                        channels,
                        target,
                        |frame, channel| match chunk {
                            // `frame` is absolute; the chunk holds one second from its
                            // own start. Chunks are interleaved stereo, and only stereo
                            // devices take this path (the engine guarantees it).
                            Some(samples) => samples[(frame % CHUNK_FRAMES) * 2 + channel],
                            None => self.original[frame * channels + channel],
                        },
                    );
                    position += span;
                    written += span;
                }
                // Nothing generated at all: still ramp, so the toggle is a no-op.
                Wet::None => {
                    mix_stretch(
                        &mut ramp,
                        ramp_step,
                        &self.original,
                        out,
                        written,
                        position,
                        want,
                        channels,
                        target,
                        |frame, channel| self.original[frame * channels + channel],
                    );
                    position += want;
                    written += want;
                }
            }
        }

        self.ramp = ramp;
        self.controls.position.store(position, Ordering::Relaxed);
        written
    }
}

/// Mix `want` frames starting at `position` into `out[written..]`.
///
/// Free function rather than a method: the caller holds an immutable borrow of the
/// generated side while mixing, and only the ramp is mutable state.
#[allow(clippy::too_many_arguments)]
fn mix_stretch(
    ramp: &mut f32,
    ramp_step: f32,
    original: &[f32],
    out: &mut [f32],
    written: usize,
    position: usize,
    want: usize,
    channels: usize,
    target: f32,
    wet: impl Fn(usize, usize) -> f32,
) {
    for index in 0..want {
        let delta = (target - *ramp).clamp(-ramp_step, ramp_step);
        *ramp += delta;
        let frame = position + index;
        let base = frame * channels;
        for channel in 0..channels {
            let dry = original[base + channel];
            let generated = if *ramp > 0.0 {
                wet(frame, channel)
            } else {
                dry
            };
            out[(written + index) * channels + channel] = dry + (generated - dry) * *ramp;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
