//! Keep a rendered track inside full scale.
//!
//! FlashSR overshoots. On a real 192 kbps MP3 the rendered signal peaked at 1.36, which
//! is 2.7 dB past full scale: a float WAV stores it happily, but every device clips it,
//! and clipped samples are audible as crackle on exactly the transients the model is
//! supposed to be good at.
//!
//! Scaling the whole track down would fix the peaks and break the A/B comparison, since
//! the two sides would then differ in level as well as in brightness. So only the peaks
//! move: samples below the threshold are untouched, and above it they are folded through
//! a soft knee that is continuous, smooth at the threshold (derivative 1) and approaches
//! unity without reaching it.

/// Where the soft knee starts. Below this, the signal is bit-identical.
pub const THRESHOLD: f32 = 0.9;

/// What the knee approaches. Not exactly 1.0: f32 rounds 1.0 - 1e-10 to 1.0, and a
/// limiter whose ceiling rounds up to full scale is not a limiter.
pub const CEILING: f32 = 0.999;

/// How far the limiting moved things, so the UI can say so instead of hiding it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LimitReport {
    pub peak_before: f32,
    pub peak_after: f32,
    pub limited_samples: usize,
}

impl LimitReport {
    /// Gain reduction at the loudest sample, in dB. Negative means it was pulled down.
    pub fn reduction_db(&self) -> f32 {
        if self.peak_before <= 0.0 {
            return 0.0;
        }
        20.0 * (self.peak_after / self.peak_before).log10()
    }
}

/// Apply the knee in place, interleaved or planar: it is per sample either way.
pub fn limit_in_place(samples: &mut [f32]) -> LimitReport {
    let mut report = LimitReport::default();
    for sample in samples.iter_mut() {
        let magnitude = sample.abs();
        report.peak_before = report.peak_before.max(magnitude);
        if magnitude > THRESHOLD {
            report.limited_samples += 1;
            let span = CEILING - THRESHOLD;
            let knee = (magnitude - THRESHOLD) / span;
            let shaped = THRESHOLD + span * knee.tanh();
            *sample = shaped.copysign(*sample);
            report.peak_after = report.peak_after.max(shaped);
        } else {
            report.peak_after = report.peak_after.max(magnitude);
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn given_samples_below_the_threshold_when_limited_then_they_are_unchanged() {
        let mut samples = vec![0.0f32, 0.25, -0.5, 0.9];
        let before = samples.clone();
        let report = limit_in_place(&mut samples);
        assert_eq!(samples, before);
        assert_eq!(report.limited_samples, 0);
        assert!((report.peak_before - 0.9).abs() < 1e-6);
    }

    #[test]
    fn given_a_sample_past_full_scale_when_limited_then_it_lands_below_unity() {
        let mut samples = vec![1.36f32];
        let report = limit_in_place(&mut samples);
        assert!(samples[0] < 1.0, "{}", samples[0]);
        assert!(samples[0] > THRESHOLD, "{}", samples[0]);
        assert_eq!(report.limited_samples, 1);
        assert!(report.reduction_db() < 0.0);
    }

    #[test]
    fn given_increasing_inputs_when_limited_then_the_output_stays_monotonic() {
        let mut samples: Vec<f32> = (0..=200).map(|i| i as f32 / 100.0).collect();
        limit_in_place(&mut samples);
        assert!(samples.windows(2).all(|pair| pair[0] <= pair[1] + 1e-7));
        assert!(samples.iter().all(|sample| *sample < 1.0));
    }

    #[test]
    fn given_a_negative_peak_when_limited_then_the_shaping_is_symmetric() {
        let mut samples = vec![1.2f32, -1.2];
        limit_in_place(&mut samples);
        assert!((samples[0] + samples[1]).abs() < 1e-6);
    }
}
