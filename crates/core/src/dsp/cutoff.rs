//! Detection of the band limit in the source, a port of `UtilAudioSR.find_cutoff_freq`.
//!
//! The reference takes an STFT of channel 0 with a periodic Hann window, sums the
//! magnitude over time, walks down from the top bin until the cumulative energy drops
//! below 98.3% of the total, and calls the resulting bin the cutoff. It is a heuristic
//! on purpose: it finds where the energy stops, not where a filter was applied, which
//! is why a 16 kHz lowpass is reported as about 15.3 kHz.

use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

use crate::error::{Error, Result};
use crate::SAMPLE_RATE;

pub const N_FFT: usize = 2_048;
pub const HOP: usize = 480;
pub const PERCENTILE: f64 = 0.983;
/// Bin that maps to the Nyquist frequency in the reference's `bin / 1024 * 24000`.
pub const NYQUIST_BIN: f64 = 1_024.0;
/// Above this fraction of energy in the top band, the source looks full-band.
pub const FULL_BAND_RATIO: f64 = 0.01;
pub const HIGH_BAND_HZ: f64 = 21_000.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Analysis {
    /// Detected cutoff in Hz, exactly as the reference reports it.
    pub cutoff_hz: u32,
    /// Fraction of spectral energy at or above 21 kHz.
    pub high_band_ratio: f64,
}

impl Analysis {
    /// True when the source looks like it already has a full top octave.
    ///
    /// The reference's own bypass rule (`cutoff >= 24 kHz`) almost never fires on real
    /// music: natural spectral tilt puts 98.3% of the energy below 22.8 kHz even with
    /// no codec involved. Measured on the fixture corpus, a genuinely full-band signal
    /// scores 4.5e-2 here while a 16 kHz-limited one scores 1e-6, so the threshold sits
    /// in a four-decade gap.
    pub fn looks_full_band(&self) -> bool {
        self.high_band_ratio >= FULL_BAND_RATIO
    }
}

/// The cutoff and top-octave ratio of one channel, over the whole signal.
pub fn analyse(channel: &[f32]) -> Result<Analysis> {
    if channel.len() < N_FFT {
        return Err(Error::SignalTooShort {
            len: channel.len(),
            min: N_FFT,
        });
    }

    let mut spectrum = Spectrum::new();
    for start in (0..=channel.len() - N_FFT).step_by(HOP) {
        spectrum.accumulate(&channel[start..start + N_FFT])?;
    }
    Ok(spectrum.into_analysis())
}

/// An analysis whose channel arrives in chunks.
///
/// Pushing the samples of one channel in any chunking and then calling
/// [`finish`](Self::finish) gives exactly what [`analyse`] gives over the whole channel:
/// every window still starts on the same multiple of `HOP`, and each bin's magnitudes are
/// summed in the same order.
pub struct ChunkedAnalyser {
    spectrum: Spectrum,
    /// Samples pushed but not yet inside a window, oldest first.
    carry: Vec<f32>,
    /// Samples pushed so far.
    consumed: usize,
}

/// Starts an analysis whose channel arrives in chunks.
pub fn chunked() -> ChunkedAnalyser {
    ChunkedAnalyser {
        spectrum: Spectrum::new(),
        carry: Vec::with_capacity(N_FFT),
        consumed: 0,
    }
}

impl ChunkedAnalyser {
    /// Adds the next samples of the channel.
    pub fn push(&mut self, chunk: &[f32]) -> Result<()> {
        self.consumed += chunk.len();
        let mut rest = chunk;
        while !rest.is_empty() {
            // One window at a time, so a push that carries the whole channel still keeps
            // `carry` under a window.
            let take = (N_FFT - self.carry.len()).min(rest.len());
            self.carry.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
            if self.carry.len() == N_FFT {
                self.spectrum.accumulate(&self.carry)?;
                // The next window starts `HOP` later; what is left in front of it is the
                // overlap the lattice needs across chunk boundaries.
                self.carry.drain(..HOP);
            }
        }
        Ok(())
    }

    /// The analysis of everything pushed, or the error [`analyse`] gives for a signal
    /// too short to hold one window.
    pub fn finish(self) -> Result<Analysis> {
        if self.consumed < N_FFT {
            return Err(Error::SignalTooShort {
                len: self.consumed,
                min: N_FFT,
            });
        }
        Ok(self.spectrum.into_analysis())
    }
}

/// The FFT and the per-bin accumulator one analysis needs, fed one window at a time.
struct Spectrum {
    r2c: Arc<dyn RealToComplex<f64>>,
    window: Vec<f64>,
    input: Vec<f64>,
    spectrum: Vec<Complex<f64>>,
    scratch: Vec<Complex<f64>>,
    /// Every bin's magnitude summed over the windows so far.
    magnitude: Vec<f64>,
}

impl Spectrum {
    fn new() -> Self {
        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(N_FFT);
        let input = r2c.make_input_vec();
        let spectrum = r2c.make_output_vec();
        let scratch = r2c.make_scratch_vec();
        let magnitude = vec![0.0f64; spectrum.len()];
        Self {
            r2c,
            window: periodic_hann(N_FFT),
            input,
            spectrum,
            scratch,
            magnitude,
        }
    }

    /// Windows one whole FFT frame and adds its magnitudes to the accumulator.
    fn accumulate(&mut self, samples: &[f32]) -> Result<()> {
        debug_assert_eq!(samples.len(), N_FFT);
        for ((slot, sample), tap) in self.input.iter_mut().zip(samples).zip(&self.window) {
            *slot = f64::from(*sample) * tap;
        }
        self.r2c
            .process_with_scratch(&mut self.input, &mut self.spectrum, &mut self.scratch)
            .map_err(|e| Error::Filter(e.to_string()))?;
        for (acc, bin) in self.magnitude.iter_mut().zip(self.spectrum.iter()) {
            *acc += bin.norm();
        }
        Ok(())
    }

    fn into_analysis(self) -> Analysis {
        Analysis {
            cutoff_hz: cutoff_from_magnitude(&self.magnitude),
            high_band_ratio: high_band_ratio(&self.magnitude),
        }
    }
}

/// The same loop as `UtilAudioSR.find_cutoff`: the highest bin whose cumulative
/// energy is still below the threshold.
fn cutoff_from_magnitude(magnitude: &[f64]) -> u32 {
    let mut energy = Vec::with_capacity(magnitude.len());
    let mut running = 0.0;
    for value in magnitude {
        running += *value;
        energy.push(running);
    }
    if running <= 0.0 {
        return SAMPLE_RATE / 2;
    }

    let threshold = energy[energy.len() - 1] * PERCENTILE;
    let mut index = 0;
    for i in 1..energy.len() {
        if energy[energy.len() - i] < threshold {
            index = energy.len() - i;
            break;
        }
    }

    let mut cutoff = (index as f64 / NYQUIST_BIN) * f64::from(SAMPLE_RATE / 2);
    if cutoff < 1000.0 {
        cutoff = f64::from(SAMPLE_RATE / 2);
    }
    cutoff as u32
}

fn high_band_ratio(magnitude: &[f64]) -> f64 {
    let total: f64 = magnitude.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let bin_hz = f64::from(SAMPLE_RATE) / N_FFT as f64;
    let start = (HIGH_BAND_HZ / bin_hz).ceil() as usize;
    let high: f64 = magnitude.iter().skip(start).sum();
    high / total
}

/// `torch.hann_window` is periodic; `np.hanning` is symmetric. The difference is one
/// sample of the denominator and it changes every bin.
fn periodic_hann(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fixed-seed signal, so a failure reproduces.
    fn pseudo_random_signal(len: usize) -> Vec<f32> {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
            })
            .collect()
    }

    /// Both halves of the result, the ratio as bits so a last-bit change counts.
    fn bits(analysis: &Analysis) -> (u32, u64) {
        (analysis.cutoff_hz, analysis.high_band_ratio.to_bits())
    }

    fn analyse_in_chunks(signal: &[f32], chunk_size: usize) -> Result<Analysis> {
        let mut analyser = chunked();
        for chunk in signal.chunks(chunk_size) {
            analyser.push(chunk)?;
        }
        analyser.finish()
    }

    #[test]
    fn given_a_signal_when_pushed_in_chunks_of_any_size_then_it_analyses_as_the_whole_signal() {
        // `N_FFT` is one frame exactly; 100 137 leaves a short last chunk and a last
        // window that starts 297 samples after the one before it.
        for len in [N_FFT, N_FFT + HOP, 100_137] {
            let signal = pseudo_random_signal(len);
            let whole = analyse(&signal).expect("the whole signal");
            for chunk_size in [1, 7, 512, 4_096, len + 1] {
                let analysis = analyse_in_chunks(&signal, chunk_size)
                    .unwrap_or_else(|e| panic!("{len} samples in chunks of {chunk_size}: {e}"));
                assert_eq!(
                    bits(&analysis),
                    bits(&whole),
                    "{len} samples in chunks of {chunk_size}"
                );
            }
        }
    }

    #[test]
    fn given_a_signal_shorter_than_one_frame_when_chunked_then_it_fails_like_analyse() {
        let signal = pseudo_random_signal(N_FFT - 1);
        let error = analyse_in_chunks(&signal, 7).expect_err("no whole window fits");
        assert_eq!(error, analyse(&signal).expect_err("the reference agrees"));
    }

    #[test]
    fn given_an_empty_feed_when_finished_then_it_fails_like_analyse_with_no_samples() {
        let mut analyser = chunked();
        analyser.push(&[]).expect("an empty chunk adds nothing");
        let error = analyser.finish().expect_err("no samples to analyse");
        assert_eq!(error, analyse(&[]).expect_err("the reference agrees"));
        assert_eq!(error, Error::SignalTooShort { len: 0, min: N_FFT });
    }
}
