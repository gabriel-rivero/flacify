//! Detection of the band limit in the source, a port of `UtilAudioSR.find_cutoff_freq`.
//!
//! The reference takes an STFT of channel 0 with a periodic Hann window, sums the
//! magnitude over time, walks down from the top bin until the cumulative energy drops
//! below 98.3% of the total, and calls the resulting bin the cutoff. It is a heuristic
//! on purpose: it finds where the energy stops, not where a filter was applied, which
//! is why a 16 kHz lowpass is reported as about 15.3 kHz.

use realfft::RealFftPlanner;

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

pub fn analyse(channel: &[f32]) -> Result<Analysis> {
    if channel.len() < N_FFT {
        return Err(Error::SignalTooShort {
            len: channel.len(),
            min: N_FFT,
        });
    }

    let frames = 1 + (channel.len() - N_FFT) / HOP;
    let window = periodic_hann(N_FFT);

    let mut planner = RealFftPlanner::<f64>::new();
    let r2c = planner.plan_fft_forward(N_FFT);
    let bins = r2c.make_output_vec().len();
    let mut input = r2c.make_input_vec();
    let mut spectrum = r2c.make_output_vec();
    let mut scratch = r2c.make_scratch_vec();
    let mut magnitude = vec![0.0f64; bins];

    for frame in 0..frames {
        let start = frame * HOP;
        for i in 0..N_FFT {
            input[i] = channel[start + i] as f64 * window[i];
        }
        r2c.process_with_scratch(&mut input, &mut spectrum, &mut scratch)
            .map_err(|e| Error::Filter(e.to_string()))?;
        for (acc, bin) in magnitude.iter_mut().zip(spectrum.iter()) {
            *acc += bin.norm();
        }
    }

    Ok(Analysis {
        cutoff_hz: cutoff_from_magnitude(&magnitude),
        high_band_ratio: high_band_ratio(&magnitude),
    })
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
