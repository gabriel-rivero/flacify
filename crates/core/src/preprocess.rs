//! Band-limiting and zero-phase filtering, the part of the pipeline that decides what
//! the model is allowed to see.
//!
//! This is a port of two reference functions: `UtilAudioSR.find_cutoff_freq` decides the
//! cutoff, and `UtilAudioLowPassFilter.lowpass` applies an eighth-order Chebyshev
//! filter zero-phase and then round-trips the signal through a rate of twice the cutoff.
//! The round trip is what actually removes the top band, and it is not optional: the
//! model was trained on inputs that went through it.

use crate::dsp::cheby::{cheby1_lowpass, Sos};
use crate::dsp::filtfilt::sosfiltfilt;
use crate::dsp::resample::Resampler;
use crate::error::{Error, Result};
use crate::SAMPLE_RATE;

/// Filter order and passband ripple used by the reference implementation.
pub const ORDER: usize = 8;
pub const RIPPLE_DB: f64 = 0.1;

/// A filter designed for one cutoff, reusable across every window of a file.
///
/// A source that is already full-band is bypassed before it reaches this type, by the
/// caller; there is no cutoff above Nyquist to design for.
pub struct Preprocessor {
    cutoff_hz: u32,
    sample_rate: u32,
    sos: Vec<Sos>,
    down: Resampler,
    up: Resampler,
}

impl Preprocessor {
    pub fn new(cutoff_hz: u32) -> Result<Self> {
        Self::with_sample_rate(cutoff_hz, SAMPLE_RATE)
    }

    pub fn with_sample_rate(cutoff_hz: u32, sample_rate: u32) -> Result<Self> {
        let fs = f64::from(sample_rate);
        let cutoff = f64::from(cutoff_hz);
        if cutoff_hz == 0 || cutoff >= fs / 2.0 {
            return Err(Error::Filter(format!(
                "cutoff {cutoff_hz} Hz is not usable at {sample_rate} Hz; \
                 a full-band source must be bypassed, not preprocessed"
            )));
        }
        let sos = cheby1_lowpass(ORDER, RIPPLE_DB, cutoff, fs)?;

        // Upstream: fs_down = int((cutoff / (fs / 2)) * fs), then resample to it and back.
        // Keeping the expression, rather than doubling the cutoff, keeps the truncation
        // identical to the Python.
        let fs_down = ((cutoff / (fs / 2.0)) * fs) as usize;
        if fs_down < 2 || fs_down >= sample_rate as usize {
            return Err(Error::Filter(format!(
                "cutoff {cutoff_hz} Hz gives an unusable intermediate rate {fs_down}"
            )));
        }
        let down = Resampler::new(fs_down, sample_rate as usize)?;
        let up = Resampler::new(sample_rate as usize, fs_down)?;

        Ok(Self {
            cutoff_hz,
            sample_rate,
            sos,
            down,
            up,
        })
    }

    pub fn cutoff_hz(&self) -> u32 {
        self.cutoff_hz
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// The first stage: zero-phase Chebyshev lowpass, still at full resolution.
    pub fn filtered(&self, channel: &[f32]) -> Result<Vec<f64>> {
        let samples: Vec<f64> = channel.iter().map(|s| f64::from(*s)).collect();
        sosfiltfilt(&self.sos, &samples)
    }

    /// The second stage: down to twice the cutoff and back, which is stricter than the
    /// Chebyshev skirt and is the part the model actually expects.
    pub fn band_limited(&self, filtered: &[f64]) -> Vec<f64> {
        let mut out = self.up.process(&self.down.process(filtered));
        // Round-tripping a ratio that does not divide evenly leaves a sample or two of
        // slack; the reference pads with zeros or truncates.
        out.resize(filtered.len(), 0.0);
        out
    }

    /// Both stages, which is what the model sees.
    pub fn process(&self, channel: &[f32]) -> Result<Vec<f64>> {
        Ok(self.band_limited(&self.filtered(channel)?))
    }
}

impl std::fmt::Debug for Preprocessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Preprocessor")
            .field("cutoff_hz", &self.cutoff_hz)
            .field("sample_rate", &self.sample_rate)
            .field("sections", &self.sos.len())
            .field("down", &(self.down.up(), self.down.down()))
            .field("up", &(self.up.up(), self.up.down()))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn given_a_full_band_cutoff_when_built_then_it_refuses_to_filter() {
        assert!(Preprocessor::new(24_000).is_err());
        assert!(Preprocessor::new(0).is_err());
    }

    #[test]
    fn given_a_detected_cutoff_when_a_dc_signal_is_preprocessed_then_only_the_dc_droop_remains() {
        let pre = Preprocessor::new(15_328).unwrap();
        let x = vec![0.25f32; 8192];
        let out = pre.process(&x).unwrap();
        assert_eq!(out.len(), x.len());
        // Zero-phase runs the Chebyshev twice, so DC sees its 0.1 dB droop squared.
        let dc_gain = 0.988_553_1f64.powi(2);
        for sample in &out[256..out.len() - 256] {
            assert!((sample - 0.25 * dc_gain).abs() < 1e-3, "{sample}");
        }
    }

    #[test]
    fn given_a_half_nyquist_cutoff_when_built_then_the_round_trip_keeps_the_length() {
        let pre = Preprocessor::new(12_000).unwrap();
        let x: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.01).sin() * 0.4).collect();
        let out = pre.process(&x).unwrap();
        assert_eq!(out.len(), x.len());
    }
}
