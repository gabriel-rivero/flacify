//! What the switch is actually doing: two spectra of the same moment.
//!
//! The picture the product needs is not a waveform, it is the top octaves. This takes a
//! 4096-sample window around the playhead from both signals, averages the channels, and
//! returns dB per bin — the original in one curve, the generated one in another. Above
//! the cutoff they diverge; that is the whole story, visible in one glance.

use realfft::RealFftPlanner;

/// Samples per analysis window: 85 ms at 48 kHz, which resolves 11.7 Hz — enough to see
/// the cutoff and the shape of the generated band.
pub const WINDOW: usize = 4_096;

/// Halves of a Hann window, so a mute toggle does not click.
fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|index| {
            let phase = std::f32::consts::TAU * index as f32 / n as f32;
            0.5 - 0.5 * phase.cos()
        })
        .collect()
}

/// dB per bin for `n / 2 + 1` bins, from a mono mix of `centre` ± `WINDOW / 2`.
///
/// Returns `None` when the buffer is too short to hold a window at that position.
pub fn analyse(interleaved: &[f32], channels: usize, centre: usize) -> Option<Vec<f32>> {
    if channels == 0 {
        return None;
    }
    let frames = interleaved.len() / channels;
    if frames < WINDOW {
        return None;
    }
    let start = centre.saturating_sub(WINDOW / 2).min(frames - WINDOW);

    let window = hann(WINDOW);
    let mut input = vec![0.0f32; WINDOW];
    for index in 0..WINDOW {
        let base = (start + index) * channels;
        let mono: f32 = (0..channels)
            .map(|channel| interleaved[base + channel])
            .sum::<f32>()
            / channels as f32;
        input[index] = mono * window[index];
    }

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(WINDOW);
    let mut spectrum = r2c.make_output_vec();
    if r2c.process(&mut input, &mut spectrum).is_err() {
        return None;
    }

    // A full-scale sine reads about 0 dB: its energy sits in one bin, spread by the
    // window, and the Hann window halves the sum.
    let scale = 2.0 / (WINDOW as f32 * 0.5);
    Some(
        spectrum
            .iter()
            .map(|bin| {
                let magnitude = (bin.re * bin.re + bin.im * bin.im).sqrt() * scale;
                20.0 * magnitude.max(1e-9).log10()
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(hz: f32, frames: usize, amplitude: f32) -> Vec<f32> {
        (0..frames)
            .flat_map(|index| {
                let value =
                    amplitude * (std::f32::consts::TAU * hz * index as f32 / 48_000.0).sin();
                [value, value]
            })
            .collect()
    }

    #[test]
    fn given_a_1khz_sine_when_analysed_then_the_peak_bin_is_at_1khz() {
        let signal = sine(1_000.0, WINDOW * 2, 0.5);
        let spectrum = analyse(&signal, 2, WINDOW).expect("a spectrum");
        let peak = spectrum
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(index, _)| index)
            .unwrap();
        let hz = peak as f32 * (48_000.0 / 2.0) / (WINDOW / 2) as f32;
        assert!((hz - 1_000.0).abs() < 25.0, "peak at {hz} Hz");
        assert_eq!(spectrum.len(), WINDOW / 2 + 1);
    }

    #[test]
    fn given_a_louder_signal_when_analysed_then_its_peak_is_higher() {
        let quiet = analyse(&sine(1_000.0, WINDOW * 2, 0.05), 2, WINDOW).expect("quiet");
        let loud = analyse(&sine(1_000.0, WINDOW * 2, 0.5), 2, WINDOW).expect("loud");
        let peak = |spectrum: &[f32]| spectrum.iter().cloned().fold(f32::MIN, f32::max);
        assert!(peak(&loud) > peak(&quiet) + 15.0);
    }

    #[test]
    fn given_a_full_scale_sine_when_analysed_then_the_peak_is_near_zero_db() {
        let spectrum = analyse(&sine(1_000.0, WINDOW * 2, 1.0), 2, WINDOW).expect("a spectrum");
        let peak = spectrum.iter().cloned().fold(f32::MIN, f32::max);
        assert!(peak > -6.0 && peak < 3.0, "peak {peak} dB");
    }

    #[test]
    fn given_a_short_buffer_when_analysed_then_there_is_no_spectrum() {
        assert!(analyse(&[0.0; 64], 2, 0).is_none());
    }
}
