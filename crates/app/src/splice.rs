//! Swap the model's low band for the input's.
//!
//! The model regenerates the whole waveform, so below the cutoff it returns its own take
//! on content the original already had: level-matched, −18.7 dB RMS away from the input
//! under 12 kHz on the tracks measured so far. The generated high band is the product;
//! the regenerated low band is collateral.
//!
//! The splice is a subtraction with the same zero-phase filter on both sides:
//!
//! ```text
//! out' = out − low(out) + low(input)
//! ```
//!
//! Both low bands come from the same Chebyshev `filtfilt`, so they lose phase in exactly
//! the same way and the difference cancels the model's low band instead of smearing it.
//! Doing it with a one-sided filter would shift the low band by the filter's group delay
//! and be audible as a smeared bass line.
//!
//! This changes what the product sounds like, so it ships behind a flag (`--splice`) until
//! someone has compared it on real music.
//!
//! **Not verified to work on real music (2026-11-13).** The unit tests pass on synthetic
//! signals, but on a rendered track the spliced output correlates with the input's low band
//! exactly as much as the unspliced one does (0.297 in 100-6,000 Hz, both fresh renders),
//! while differing from it by -14.8 dB. Something in the wiring or in the assumption that
//! output frame k corresponds to input frame k is wrong — the generated signal's best
//! alignment against the original wanders (3, -53 and +429 samples at 0.5, 2 and 5 s), so
//! the splice's premise of identical alignment may simply not hold. Diagnose before
//! enabling.

use anyhow::Result;
use flacify_core::Preprocessor;

/// One channel of the splice. `input` and `output` must be the same length.
pub fn channel(output: &[f32], input: &[f32], preprocessor: &Preprocessor) -> Result<Vec<f32>> {
    if output.len() != input.len() {
        anyhow::bail!(
            "splice needs matching lengths, got {} and {}",
            output.len(),
            input.len()
        );
    }
    let low_output = preprocessor.filtered(output)?;
    let low_input = preprocessor.filtered(input)?;

    Ok(output
        .iter()
        .zip(low_input.iter().zip(low_output.iter()))
        .map(|(sample, (keep, drop))| (f64::from(*sample) + (keep - drop)) as f32)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn amplitude_at(samples: &[f32], hz: f32) -> f64 {
        // Projection onto the sine basis: two passes, because the phase is unknown.
        let mut real = 0.0;
        let mut imag = 0.0;
        for (index, sample) in samples.iter().enumerate() {
            let phase = std::f64::consts::TAU * f64::from(hz) * index as f64 / 48_000.0;
            real += f64::from(*sample) * phase.cos();
            imag += f64::from(*sample) * phase.sin();
        }
        2.0 * (real * real + imag * imag).sqrt() / samples.len() as f64
    }

    fn tone(hz: f32, amplitude: f32, frames: usize) -> Vec<f32> {
        (0..frames)
            .map(|index| amplitude * (std::f32::consts::TAU * hz * index as f32 / 48_000.0).sin())
            .collect()
    }

    #[test]
    fn given_output_with_a_different_low_band_when_spliced_then_the_input_low_band_returns() {
        let frames = 48_000;
        let input: Vec<f32> = (0..frames)
            .map(|_| 0.0)
            .zip(tone(1_000.0, 0.30, frames))
            .map(|(_, value)| value)
            .collect();
        let output: Vec<f32> = tone(1_000.0, 0.10, frames)
            .iter()
            .zip(tone(18_000.0, 0.20, frames))
            .map(|(low, high)| low + high)
            .collect();

        let pre = Preprocessor::new(14_000).expect("a filter");
        let spliced = channel(&output, &input, &pre).expect("a splice");

        let low = amplitude_at(&spliced, 1_000.0);
        let high = amplitude_at(&spliced, 18_000.0);
        assert!((low - 0.30).abs() < 0.02, "1 kHz came out at {low}");
        assert!((high - 0.20).abs() < 0.02, "18 kHz came out at {high}");
    }

    #[test]
    fn given_matching_low_bands_when_spliced_then_the_signal_is_unchanged() {
        let frames = 48_000;
        let signal = tone(1_000.0, 0.25, frames);
        let pre = Preprocessor::new(14_000).expect("a filter");
        let spliced = channel(&signal, &signal, &pre).expect("a splice");
        let worst = spliced
            .iter()
            .zip(signal.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Two identical filtfilt passes cancel exactly, up to f64 rounding.
        assert!(worst < 1e-6, "worst difference {worst}");
    }

    #[test]
    fn given_mismatched_lengths_when_spliced_then_it_refuses() {
        let pre = Preprocessor::new(14_000).expect("a filter");
        assert!(channel(&[0.0; 10], &[0.0; 9], &pre).is_err());
    }
}
