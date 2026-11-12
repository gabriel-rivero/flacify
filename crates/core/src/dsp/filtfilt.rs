//! Zero-phase filtering, ported from `scipy.signal.sosfiltfilt`.
//!
//! Two passes over an odd-extended copy of the signal: forward, then backward from a
//! steady-state initial condition. The odd extension (a 180-degree rotation about each
//! endpoint) keeps the filter from seeing a step at the boundaries, and it is trimmed
//! afterwards, so the phase is exactly zero and the length is unchanged.

use crate::error::{Error, Result};

use super::cheby::Sos;

/// One biquad's transposed direct-form II state.
pub type State = [f64; 2];

/// Steady-state state for a unit step, per section, matching `scipy.signal.sosfilt_zi`.
///
/// Each section is seeded with the DC gain of the sections before it, so the cascade
/// as a whole starts at its steady state rather than settling into it.
pub fn sosfilt_zi(sos: &[Sos]) -> Vec<State> {
    let mut state = vec![[0.0; 2]; sos.len()];
    let mut scale = 1.0;
    for (index, section) in sos.iter().enumerate() {
        let [b0, b1, b2] = section.b;
        let [_, a1, a2] = section.a;
        let steady = (b0 + b1 + b2) / (1.0 + a1 + a2);
        let d1 = b1 - steady * a1;
        let d2 = b2 - steady * a2;
        state[index] = [scale * (d1 + d2), scale * d2];
        scale *= section.dc_gain();
    }
    state
}

/// Run every section in place over `x`, carrying `state` in and out.
pub fn sosfilt(sos: &[Sos], x: &mut [f64], state: &mut [State]) {
    for (section, z) in sos.iter().zip(state.iter_mut()) {
        let [b0, b1, b2] = section.b;
        let [_, a1, a2] = section.a;
        let [mut z1, mut z2] = *z;
        for sample in x.iter_mut() {
            let input = *sample;
            let output = b0 * input + z1;
            z1 = b1 * input - a1 * output + z2;
            z2 = b2 * input - a2 * output;
            *sample = output;
        }
        *z = [z1, z2];
    }
}

/// How many samples `sosfiltfilt` appends at each end, from `2 * sections + 1` taps.
pub fn edge(sos: &[Sos]) -> usize {
    3 * (2 * sos.len() + 1)
}

/// Forward-backward filter with an odd extension, so the result is zero-phase and
/// exactly as long as the input.
pub fn sosfiltfilt(sos: &[Sos], x: &[f64]) -> Result<Vec<f64>> {
    let pad = edge(sos);
    if x.len() <= pad {
        return Err(Error::SignalTooShort {
            len: x.len(),
            min: pad,
        });
    }

    let len = x.len();
    let mut y = vec![0.0; len + 2 * pad];
    // left: 2*x[0] - x[pad..0:-1], so y[0] mirrors x[pad] about x[0].
    for i in 0..pad {
        y[i] = 2.0 * x[0] - x[pad - i];
    }
    y[pad..pad + len].copy_from_slice(x);
    // right: 2*x[len-1] - x[len-2..len-2-pad:-1].
    for i in 0..pad {
        y[pad + len + i] = 2.0 * x[len - 1] - x[len - 2 - i];
    }

    let zi = sosfilt_zi(sos);
    let mut state: Vec<State> = zi.iter().map(|s| [s[0] * y[0], s[1] * y[0]]).collect();
    sosfilt(sos, &mut y, &mut state);

    let last = y[y.len() - 1];
    let mut state: Vec<State> = zi.iter().map(|s| [s[0] * last, s[1] * last]).collect();
    y.reverse();
    sosfilt(sos, &mut y, &mut state);
    y.reverse();

    Ok(y[pad..pad + len].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::cheby::cheby1_lowpass;

    #[test]
    fn given_a_unit_step_when_seeded_then_the_state_starts_at_steady_state() {
        let sos = cheby1_lowpass(8, 0.1, 16_000.0, 48_000.0).unwrap();
        let mut step = vec![1.0; 64];
        let mut state = sosfilt_zi(&sos);
        sosfilt(&sos, &mut step, &mut state);
        let steady: f64 = sos.iter().map(|s| s.dc_gain()).product();
        for sample in &step {
            assert!((sample - steady).abs() < 1e-12, "{sample} != {steady}");
        }
    }

    #[test]
    fn given_a_constant_signal_when_filtered_zero_phase_then_it_drops_by_the_double_pass_gain() {
        let sos = cheby1_lowpass(8, 0.1, 16_000.0, 48_000.0).unwrap();
        let x = vec![0.25; 4096];
        let y = sosfiltfilt(&sos, &x).unwrap();
        assert_eq!(y.len(), x.len());
        // Zero-phase means the filter runs twice, so DC sees the gain squared: -0.2 dB.
        let steady: f64 = sos.iter().map(|s| s.dc_gain()).product::<f64>().powi(2);
        for sample in &y {
            assert!((sample - 0.25 * steady).abs() < 1e-9, "{sample}");
        }
    }

    #[test]
    fn given_a_signal_shorter_than_the_padding_when_filtered_then_it_is_rejected() {
        let sos = cheby1_lowpass(8, 0.1, 16_000.0, 48_000.0).unwrap();
        assert_eq!(edge(&sos), 27);
        assert!(sosfiltfilt(&sos, &[0.0; 27]).is_err());
    }
}
