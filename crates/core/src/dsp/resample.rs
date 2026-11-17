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
//!
//! Input before the first sample and past the last one reads as zero, and each output
//! sample sums its taps in a fixed order, so splitting the input into chunks cannot
//! change a value. An output sample is final once its newest input sample has been
//! pushed: at most the last `post_roll()` outputs of everything pushed so far wait for
//! the next push, and `finish()` resolves them against the end-of-input zeros.

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
        let n_out = output_count(x.len(), self.up, self.down);
        let mut out = Vec::with_capacity(n_out);
        for i in 0..n_out {
            out.push(self.output_sample(x, 0, i));
        }
        out
    }

    /// Output sample `i` of the whole signal, read from a window whose first sample is
    /// input index `first_index`.
    fn output_sample(&self, window: &[f64], first_index: usize, i: usize) -> f64 {
        // Index into the convolution of the upsampled input with the padded filter.
        let centre = (self.pre_remove + i) * self.down;
        let phase = centre % self.up;
        // Newest input sample this output reads, in window coordinates.
        let last = centre / self.up - first_index;
        // Largest read offset that maps back inside `window`.
        let mut r = last.saturating_sub(window.len().saturating_sub(1));
        if self.pre_pad > phase {
            r = r.max((self.pre_pad - phase).div_ceil(self.up));
        }
        let mut acc = 0.0;
        while r <= last {
            let tap = phase + r * self.up - self.pre_pad;
            if tap >= self.taps.len() {
                break;
            }
            acc += self.taps[tap] * window[last - r];
            r += 1;
        }
        acc
    }

    /// Input samples before a chunk start that the filter still reads, and the number a
    /// caller keeping its own overlap must repeat in front of a chunk.
    pub fn pre_roll(&self) -> usize {
        if self.is_passthrough() {
            return 0;
        }
        // The furthest tap back from `last` needs `phase + r * up - pre_pad < taps.len()`.
        (self.pre_pad + self.taps.len() - 1) / self.up
    }

    /// At most this many output samples at the end of everything pushed so far are not
    /// final yet: output `i` reads input up to `(pre_remove + i) * down / up`, so the
    /// first one still waiting is `n_out - pre_remove`.
    pub fn post_roll(&self) -> usize {
        if self.is_passthrough() {
            return 0;
        }
        self.pre_remove
    }

    /// Starts a pass over an input that arrives in chunks.
    pub fn chunked(&self) -> ChunkedResampler<'_> {
        ChunkedResampler::new(self)
    }
}

/// A resampling pass whose input arrives in chunks.
///
/// Concatenating every `process_chunk` result and then `finish` equals `process` over
/// the whole input, bit for bit.
#[derive(Debug)]
pub struct ChunkedResampler<'a> {
    resampler: &'a Resampler,
    /// Input samples not yet dropped, oldest first, ending at the newest pushed sample.
    window: Vec<f64>,
    /// Input samples pushed so far.
    consumed: usize,
    /// Output samples returned so far.
    emitted: usize,
}

impl<'a> ChunkedResampler<'a> {
    fn new(resampler: &'a Resampler) -> Self {
        Self {
            resampler,
            window: Vec::new(),
            consumed: 0,
            emitted: 0,
        }
    }

    /// Resamples `chunk`, returning the samples whose values are final at this position.
    pub fn process_chunk(&mut self, chunk: &[f64]) -> Vec<f64> {
        if self.resampler.is_passthrough() {
            self.consumed += chunk.len();
            self.emitted += chunk.len();
            return chunk.to_vec();
        }
        self.window.extend_from_slice(chunk);
        self.consumed += chunk.len();
        let end_output = self
            .signal_output_count()
            .saturating_sub(self.resampler.post_roll());
        let out = self.emit(end_output);
        self.retain_pre_roll();
        out
    }

    /// Resolves the outputs that waited for more input, and returns them.
    pub fn finish(mut self) -> Vec<f64> {
        let end_output = self.signal_output_count();
        self.emit(end_output)
    }

    /// Input samples the pass still holds; at most `pre_roll()`, and zero for a
    /// passthrough ratio.
    pub fn retained_input_len(&self) -> usize {
        self.window.len()
    }

    /// Output samples a whole input of the length pushed so far would produce.
    fn signal_output_count(&self) -> usize {
        output_count(self.consumed, self.resampler.up, self.resampler.down)
    }

    /// Computes output samples `[emitted, end_output)` from the window.
    fn emit(&mut self, end_output: usize) -> Vec<f64> {
        let first_index = self.consumed - self.window.len();
        let mut out = Vec::with_capacity(end_output - self.emitted);
        for i in self.emitted..end_output {
            // Outputs that wait for input start inside the window, so this cannot wrap.
            out.push(self.resampler.output_sample(&self.window, first_index, i));
        }
        self.emitted = end_output;
        out
    }

    /// Drops the input no later output reads, keeping the `pre_roll()` tail.
    fn retain_pre_roll(&mut self) {
        let keep = self.resampler.pre_roll();
        if self.window.len() > keep {
            let excess = self.window.len() - keep;
            self.window.drain(..excess);
        }
    }
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

    /// A fixed-seed signal, so a failure reproduces.
    fn pseudo_random_signal(len: usize) -> Vec<f64> {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as f64 / u64::MAX as f64 * 2.0 - 1.0
            })
            .collect()
    }

    /// Bit patterns, so `-0.0` against `0.0` counts as the difference it is.
    fn bits(samples: &[f64]) -> Vec<u64> {
        samples.iter().map(|sample| sample.to_bits()).collect()
    }

    #[test]
    fn given_the_44_1_khz_ratios_when_asked_for_their_rolls_then_the_pre_roll_is_the_reach() {
        let r = Resampler::new(147, 160).unwrap();
        assert_eq!((r.pre_roll(), r.post_roll()), (22, 11));
        let r = Resampler::new(160, 147).unwrap();
        assert_eq!((r.pre_roll(), r.post_roll()), (20, 11));
    }

    #[test]
    fn given_no_input_when_finished_then_nothing_comes_out() {
        let r = Resampler::new(147, 160).unwrap();
        let mut chunked = r.chunked();
        assert!(chunked.process_chunk(&[]).is_empty());
        assert_eq!(chunked.retained_input_len(), 0);
        assert!(chunked.finish().is_empty());
    }

    #[test]
    fn given_a_long_signal_when_resampled_in_chunks_then_every_sample_matches_process() {
        let x = pseudo_random_signal(100_000);
        for (up, down) in [(147, 160), (160, 147)] {
            let r = Resampler::new(up, down).unwrap();
            let whole = r.process(&x);
            for chunk_size in [1, 17, 1024, 4096, 7777, 16_384] {
                let mut chunked = r.chunked();
                let mut out = Vec::new();
                for chunk in x.chunks(chunk_size) {
                    out.extend(chunked.process_chunk(chunk));
                    assert!(chunked.retained_input_len() <= r.pre_roll());
                }
                out.extend(chunked.finish());
                let context = format!("{up}/{down} in chunks of {chunk_size}");
                assert_eq!(out.len(), whole.len(), "{context}");
                assert_eq!(bits(&out), bits(&whole), "{context}");
            }
        }
    }

    #[test]
    fn given_a_length_off_the_chunk_and_down_grid_when_chunked_then_the_joins_stay_aligned() {
        let r = Resampler::new(147, 160).unwrap();
        let x = pseudo_random_signal(10_001);
        assert_ne!(x.len() % 300, 0, "the last chunk must be a short one");
        assert_ne!(
            x.len() * 147 % 160,
            0,
            "the input must not land on the output grid"
        );
        let whole = r.process(&x);
        assert_eq!(whole.len(), (10_001u128 * 147).div_ceil(160) as usize);

        let mut chunked = r.chunked();
        let mut out = Vec::new();
        let mut pushed = 0;
        for chunk in x.chunks(300) {
            out.extend(chunked.process_chunk(chunk));
            // A decoder can hand an empty block; it must not shift the joins.
            out.extend(chunked.process_chunk(&[]));
            pushed += chunk.len();
            let ready = output_count(pushed, 147, 160).saturating_sub(r.post_roll());
            assert_eq!(out.len(), ready, "after {pushed} input samples");
        }
        assert_eq!(chunked.retained_input_len(), r.pre_roll());
        let tail = chunked.finish();
        assert_eq!(
            tail.len(),
            r.post_roll(),
            "the last chunk's tail is dropped"
        );
        out.extend(tail);
        assert_eq!(out.len(), whole.len());
        assert_eq!(bits(&out), bits(&whole));
    }

    #[test]
    fn given_the_passthrough_ratio_when_chunked_then_it_copies_and_keeps_no_history() {
        let r = Resampler::new(48_000, 48_000).unwrap();
        assert_eq!(r.pre_roll(), 0);
        assert_eq!(r.post_roll(), 0);

        let x = pseudo_random_signal(5_000);
        let mut chunked = r.chunked();
        let mut out = Vec::new();
        for chunk in x.chunks(512) {
            let samples = chunked.process_chunk(chunk);
            assert_eq!(samples, chunk);
            assert_eq!(chunked.retained_input_len(), 0);
            out.extend(samples);
        }
        assert!(chunked.finish().is_empty());
        assert_eq!(bits(&out), bits(&x));
    }
}
