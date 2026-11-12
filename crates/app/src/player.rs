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

/// How long the switch takes, in seconds.
const RAMP_SECS: f32 = 0.008;

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

pub struct Mixer {
    original: Arc<[f32]>,
    enhanced: Option<Arc<[f32]>>,
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
        });
        let mixer = Self {
            original: original.into(),
            enhanced: enhanced.map(Into::into),
            channels,
            ramp: if has_enhanced { 1.0 } else { 0.0 },
            ramp_step: 1.0 / (RAMP_SECS * sample_rate.max(1) as f32),
            controls: Arc::clone(&controls),
        };
        (mixer, controls)
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

        let target = if self.controls.is_enhanced() {
            1.0
        } else {
            0.0
        };
        let mut position = self.controls.position.load(Ordering::Relaxed);
        let mut written = 0;

        for frame in 0..frames {
            if position >= self.controls.frames {
                self.controls.playing.store(false, Ordering::Relaxed);
                out[frame * channels..].fill(0.0);
                break;
            }
            // Ramp toward the chosen source, one step per frame.
            let delta = (target - self.ramp).clamp(-self.ramp_step, self.ramp_step);
            self.ramp += delta;

            let base = position * channels;
            let enhanced = self.enhanced.as_deref();
            for channel in 0..channels {
                let dry = self.original[base + channel];
                let wet = match enhanced {
                    Some(buffer) if self.ramp > 0.0 => buffer[base + channel],
                    _ => dry,
                };
                out[frame * channels + channel] = dry + (wet - dry) * self.ramp;
            }
            position += 1;
            written += 1;
        }

        self.controls.position.store(position, Ordering::Relaxed);
        written
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
