//! Polyphase resampling, ported from `scipy.signal.resample_poly`.
//!
//! The reference preprocessing resamples down to twice the detected cutoff and back up,
//! which is how it band-limits the model input: a Kaiser-windowed sinc with ten lobes on
//! each side, evaluated as a polyphase FIR so only one phase is touched per output
//! sample.
//!
//! scipy zero-pads the filter so that output sample 0 is centred; the padding only
//! changes which convolution samples are kept, never their values, so this port keeps
//! the unpadded taps and applies the same offset when it reads them.

use crate::error::{Error, Result};

/// Ten lobes of the sinc per side, matching scipy's `half_len = 10 * max_rate`.
const LOBES: usize = 10;
const KAISER_BETA: f64 = 5.0;

/// A designed rational resampler, reusable across windows at the same cutoff.
#[derive(Clone, Debug)]
pub struct Resampler {
    up: usize,
    down: usize,
    taps: Vec<f64>,
    /// Leading zero taps scipy prepends so that output sample 0 is centred.
    pre_pad: usize,
    /// Offset of the first kept output sample, in input samples, as scipy computes it.
    pre_remove: usize,
}

impl Resampler {
    pub fn new(up: usize, down: usize) -> Result<Self> {
        if up == 0 || down == 0 {
            return Err(Error::Filter(format!(
                "resample ratios must be positive, got {up}/{down}"
            )));
        }
        let g = gcd(up, down);
        let (up, down) = (up / g, down / g);

        let max_rate = up.max(down);
        let half_len = LOBES * max_rate;
        let count = 2 * half_len + 1;
        let cutoff = 1.0 / max_rate as f64;
        let mut taps = Vec::with_capacity(count);
        let mut sum = 0.0;
        for i in 0..count {
            let m = i as f64 - (count as f64 - 1.0) / 2.0;
            let value = cutoff * sinc(cutoff * m) * kaiser(i, count, KAISER_BETA);
            taps.push(value);
            sum += value;
        }
        for tap in &mut taps {
            *tap = *tap / sum * up as f64;
        }

        let pre_pad = down - half_len % down;
        let pre_remove = (half_len + pre_pad) / down;
        Ok(Self {
            up,
            down,
            taps,
            pre_pad,
            pre_remove,
        })
    }

    pub fn up(&self) -> usize {
        self.up
    }

    pub fn down(&self) -> usize {
        self.down
    }

    pub fn is_passthrough(&self) -> bool {
        self.up == 1 && self.down == 1
    }

    /// Resample `x`, returning `ceil(len(x) * up / down)` samples.
    pub fn process(&self, x: &[f64]) -> Vec<f64> {
        if self.is_passthrough() {
            return x.to_vec();
        }
        let n_in = x.len();
        let n_out = output_count(n_in, self.up, self.down);
        let pre_pad = self.pre_pad;

        // scipy appends whatever zero taps make the convolution long enough to hold the
        // slice it keeps. Those taps are zero, so only the bound below matters here.
        let mut post_pad = 0;
        while convolved_len(
            n_in,
            self.taps.len() + pre_pad + post_pad,
            self.up,
            self.down,
        ) < n_out + self.pre_remove
        {
            post_pad += 1;
        }

        let mut out = Vec::with_capacity(n_out);
        for i in 0..n_out {
            // Index into the convolution of the upsampled input with the filter.
            let n = self.pre_remove + i;
            let centre = n * self.down;
            let phase = centre % self.up;
            // Largest read offset that maps back inside `x`.
            let last = centre / self.up;
            let mut r = last.saturating_sub(n_in.saturating_sub(1));
            if pre_pad > phase {
                r = r.max((pre_pad - phase).div_ceil(self.up));
            }
            let mut acc = 0.0;
            while r <= last {
                let tap = phase + r * self.up - pre_pad;
                if tap >= self.taps.len() {
                    break;
                }
                acc += self.taps[tap] * x[last - r];
                r += 1;
            }
            out.push(acc);
        }
        out
    }
}

/// `ceil(((n_in - 1) * up + taps) / down)`, the length `upfirdn` produces.
fn convolved_len(n_in: usize, taps: usize, up: usize, down: usize) -> usize {
    let numerator = (n_in.saturating_sub(1)) as u128 * up as u128 + taps as u128;
    numerator.div_ceil(down as u128) as usize
}

/// `ceil(n_in * up / down)`, the length `resample_poly` returns.
fn output_count(n_in: usize, up: usize, down: usize) -> usize {
    (n_in as u128 * up as u128).div_ceil(down as u128) as usize
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let pix = std::f64::consts::PI * x;
        pix.sin() / pix
    }
}

/// Symmetric Kaiser window, matching `numpy.kaiser(n, beta)`.
fn kaiser(i: usize, n: usize, beta: f64) -> f64 {
    let ratio = 2.0 * i as f64 / (n as f64 - 1.0) - 1.0;
    let arg = (1.0 - ratio * ratio).max(0.0);
    bessel_i0(beta * arg.sqrt()) / bessel_i0(beta)
}

/// Modified Bessel function of the first kind, order zero, by its series expansion.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let half = x / 2.0;
    for k in 1..64 {
        term *= half / k as f64;
        let contribution = term * term;
        sum += contribution;
        if contribution < sum * 1e-17 {
            break;
        }
    }
    sum
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn given_a_ratio_of_one_when_resampled_then_it_is_a_copy() {
        let r = Resampler::new(4800, 4800).unwrap();
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        assert!(r.is_passthrough());
        assert_eq!(r.process(&x), x);
    }

    #[test]
    fn given_a_dc_signal_when_resampled_then_the_level_is_preserved() {
        let r = Resampler::new(15_328, 48_000).unwrap();
        let x = vec![0.5; 16_384];
        let y = r.process(&x);
        assert_eq!(y.len(), (16_384u128 * 15_328).div_ceil(48_000) as usize);
        // Skip the ramp at both ends where the zero-padded filter is still filling.
        for sample in &y[64..y.len() - 64] {
            assert!((sample - 0.5).abs() < 2e-3, "{sample}");
        }
    }

    #[test]
    fn given_an_upsampling_ratio_when_resampled_then_the_length_matches_ceil() {
        let r = Resampler::new(3, 2).unwrap();
        assert_eq!(r.process(&vec![0.0; 1001]).len(), 1502);
        assert_eq!(r.process(&vec![0.0; 1000]).len(), 1500);
    }
}
