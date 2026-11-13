//! BS.1770 loudness, for one purpose: make the two sides of the switch the same level.
//!
//! Measured before this existed, the generated side was **+0.57 dB** on a steady track.
//! Half a decibel is enough to make a sighted comparison dishonest: the brighter side
//! also sounds better because it is louder. Matching integrated loudness removes that.
//!
//! The implementation is the standard's own: K-weighting (a high shelf and a high-pass at
//! 48 kHz), 400 ms blocks with 75% overlap, an absolute gate at -70 LUFS and a relative
//! gate 10 LU below the ungated mean. It is not a full EBU R128 meter — no true peak, no
//! momentary/short-term windows — because nothing here needs them.

/// BS.1770 high-shelf stage, 48 kHz.
const SHELF_B: [f64; 3] = [
    1.535_124_859_586_97,
    -2.691_696_189_406_38,
    1.198_392_810_852_85,
];
const SHELF_A: [f64; 3] = [1.0, -1.690_659_293_182_41, 0.732_480_774_215_85];

/// BS.1770 high-pass stage, 48 kHz.
const HIGH_B: [f64; 3] = [1.0, -2.0, 1.0];
const HIGH_A: [f64; 3] = [1.0, -1.990_047_454_833_98, 0.990_072_250_366_21];

const SAMPLE_RATE: usize = 48_000;
const BLOCK: usize = SAMPLE_RATE / 5 * 2; // 400 ms
const HOP: usize = BLOCK / 4;

/// A direct-form-I biquad, enough for two fixed filters.
struct Biquad {
    b: [f64; 3],
    a: [f64; 3],
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl Biquad {
    fn new(b: [f64; 3], a: [f64; 3]) -> Self {
        Self {
            b,
            a,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn process(&mut self, x: f64) -> f64 {
        let y = self.b[0] * x + self.b[1] * self.x1 + self.b[2] * self.x2
            - self.a[1] * self.y1
            - self.a[2] * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

fn k_weight(samples: &[f32]) -> Vec<f64> {
    let mut shelf = Biquad::new(SHELF_B, SHELF_A);
    let mut high = Biquad::new(HIGH_B, HIGH_A);
    samples
        .iter()
        .map(|sample| high.process(shelf.process(f64::from(*sample))))
        .collect()
}

fn block_loudness(mean_square: f64) -> f64 {
    -0.691 + 10.0 * mean_square.max(1e-20).log10()
}

/// Integrated loudness in LUFS. `-inf` for silence.
///
/// Tracks shorter than one 400 ms block are measured over their whole length rather than
/// rejected: a fragment is still a thing to compare.
pub fn integrated(channels: &[&[f32]]) -> f64 {
    let weighted: Vec<Vec<f64>> = channels.iter().map(|channel| k_weight(channel)).collect();
    let frames = weighted.first().map_or(0, |channel| channel.len());
    if frames == 0 {
        return f64::NEG_INFINITY;
    }

    let mut blocks: Vec<f64> = Vec::new();
    if frames < BLOCK {
        let mean_square: f64 = weighted
            .iter()
            .map(|channel| channel.iter().map(|s| s * s).sum::<f64>() / frames as f64)
            .sum();
        blocks.push(mean_square);
    } else {
        let mut start = 0;
        while start + BLOCK <= frames {
            let mean_square: f64 = weighted
                .iter()
                .map(|channel| {
                    channel[start..start + BLOCK]
                        .iter()
                        .map(|s| s * s)
                        .sum::<f64>()
                        / BLOCK as f64
                })
                .sum();
            blocks.push(mean_square);
            start += HOP;
        }
    }

    let above_absolute: Vec<f64> = blocks
        .iter()
        .copied()
        .filter(|mean_square| block_loudness(*mean_square) > -70.0)
        .collect();
    if above_absolute.is_empty() {
        return f64::NEG_INFINITY;
    }

    let ungated = above_absolute.iter().sum::<f64>() / above_absolute.len() as f64;
    let relative_gate = block_loudness(ungated) - 10.0;
    let gated: Vec<f64> = above_absolute
        .iter()
        .copied()
        .filter(|mean_square| block_loudness(*mean_square) > relative_gate)
        .collect();
    let mean = gated.iter().sum::<f64>() / gated.len() as f64;
    block_loudness(mean)
}

/// The gain, in dB, that would bring `current` to `target`.
pub fn gain_db(target: f64, current: f64) -> f64 {
    if !target.is_finite() || !current.is_finite() {
        return 0.0;
    }
    target - current
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(amplitude: f32, frames: usize) -> Vec<f32> {
        (0..frames)
            .map(|index| {
                amplitude * (std::f32::consts::TAU * 1_000.0 * index as f32 / 48_000.0).sin()
            })
            .collect()
    }

    #[test]
    fn given_a_full_scale_1khz_sine_when_measured_then_it_reads_about_minus_half_a_lufs() {
        // The standard's own calibration: a full-scale 1 kHz sine is ~-0.7 LUFS.
        let left = sine(1.0, 48_000 * 2);
        let loudness = integrated(&[&left, &left]);
        assert!(
            (loudness + 0.7).abs() < 1.0,
            "expected about -0.7 LUFS, got {loudness}"
        );
    }

    #[test]
    fn given_a_sine_20_db_quieter_when_measured_then_it_reads_20_lower() {
        let loud = sine(1.0, 48_000 * 2);
        let quiet = sine(0.1, 48_000 * 2);
        let delta = integrated(&[&loud, &loud]) - integrated(&[&quiet, &quiet]);
        assert!((delta - 20.0).abs() < 0.5, "delta {delta} dB");
    }

    #[test]
    fn given_silence_when_measured_then_it_is_minus_infinity() {
        let silence = vec![0.0f32; 48_000];
        assert_eq!(integrated(&[&silence, &silence]), f64::NEG_INFINITY);
    }

    #[test]
    fn given_two_measurements_when_asked_for_a_gain_then_it_is_the_difference() {
        assert!((gain_db(-14.0, -15.5) - 1.5).abs() < 1e-9);
        assert_eq!(gain_db(-14.0, f64::NEG_INFINITY), 0.0);
    }

    #[test]
    fn given_a_track_shorter_than_a_block_when_measured_then_it_still_returns_a_number() {
        let short = sine(1.0, 1_000);
        assert!(integrated(&[&short, &short]).is_finite());
    }
}
