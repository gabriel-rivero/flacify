//! Reproducible noise for the sampler's starting point.
//!
//! Uses the Irwin-Hall sum of twelve uniforms, which is unit variance and needs only
//! multiplication and addition. A Box-Muller or Marsaglia draw is cheaper but goes
//! through `sin`, `cos` or `ln`, whose last bits are not identical across platforms or
//! libm versions; this stays bit-exact on every target, which is what ADR-006 promises.
//! The cost is that the tails stop at six sigma, which for a starting point the model
//! immediately denoises does not matter.

/// Fill `out` with standard normal samples derived from `seed`.
pub fn fill_normal(out: &mut [f32], seed: u64) {
    let mut state = State { word: seed };
    for sample in out.iter_mut() {
        let mut sum = 0.0f32;
        for _ in 0..12 {
            sum += state.next_unit();
        }
        *sample = sum - 6.0;
    }
}

struct State {
    word: u64,
}

impl State {
    /// A uniform in [0, 1), one per splitmix64 step.
    fn next_unit(&mut self) -> f32 {
        self.word = self.word.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.word;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // 24 bits, the mantissa of an f32, so every value is representable.
        ((z >> 40) as f32) * (1.0 / 16_777_216.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn given_the_same_seed_when_filled_twice_then_the_noise_is_identical() {
        let mut a = vec![0.0; 1024];
        let mut b = vec![0.0; 1024];
        fill_normal(&mut a, 0xDEAD_BEEF);
        fill_normal(&mut b, 0xDEAD_BEEF);
        assert_eq!(a, b);
    }

    #[test]
    fn given_a_different_seed_when_filled_then_the_noise_differs() {
        let mut a = vec![0.0; 1024];
        let mut b = vec![0.0; 1024];
        fill_normal(&mut a, 1);
        fill_normal(&mut b, 2);
        assert_ne!(a, b);
    }

    #[test]
    fn given_a_long_run_when_filled_then_it_is_zero_mean_and_unit_variance() {
        let mut noise = vec![0.0; 200_000];
        fill_normal(&mut noise, 7);
        let mean: f64 = noise.iter().map(|s| f64::from(*s)).sum::<f64>() / noise.len() as f64;
        let var: f64 = noise
            .iter()
            .map(|s| (f64::from(*s) - mean).powi(2))
            .sum::<f64>()
            / noise.len() as f64;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "variance {var}");
        assert!(noise.iter().all(|s| s.abs() <= 6.0));
    }
}
