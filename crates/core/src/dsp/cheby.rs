//! Chebyshev type I lowpass design, ported from `scipy.signal.cheby1(..., output="sos")`.
//!
//! The reference implementation builds an eighth-order filter at 0.1 dB of ripple and
//! applies it zero-phase. Section order matters here only for rounding, but it is
//! reproduced exactly (poles closest to the unit circle last, gain on the first
//! section) so the fixtures can be compared at 1e-12 instead of at a loose tolerance.

use crate::error::{Error, Result};

/// One biquad. `a[0]` is always 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sos {
    pub b: [f64; 3],
    pub a: [f64; 3],
}

impl Sos {
    /// Gain at DC, which is the steady-state response to a unit step.
    pub fn dc_gain(&self) -> f64 {
        (self.b[0] + self.b[1] + self.b[2]) / (self.a[0] + self.a[1] + self.a[2])
    }
}

/// Design a lowpass Chebyshev I filter as a cascade of biquads.
///
/// `cutoff_hz` is the point where the gain first drops below `-ripple_db`, which is
/// not the half-power point. Only even orders are supported: an odd order needs a
/// first-order section and the model never asks for one.
pub fn cheby1_lowpass(
    order: usize,
    ripple_db: f64,
    cutoff_hz: f64,
    sample_rate: f64,
) -> Result<Vec<Sos>> {
    if order == 0 || order % 2 != 0 {
        return Err(Error::Filter(format!(
            "chebyshev order must be a positive even number, got {order}"
        )));
    }
    if !(cutoff_hz > 0.0 && cutoff_hz < sample_rate / 2.0) {
        return Err(Error::Filter(format!(
            "cutoff {cutoff_hz} Hz is outside (0, {})",
            sample_rate / 2.0
        )));
    }
    if ripple_db.is_nan() || ripple_db <= 0.0 {
        return Err(Error::Filter(format!(
            "ripple must be positive, got {ripple_db} dB"
        )));
    }

    let n = order;
    let eps = (10f64.powf(0.1 * ripple_db) - 1.0).sqrt();
    let mu = (1.0 / eps).asinh() / n as f64;

    // Analog prototype poles, arranged on an ellipse in the left half plane.
    let mut prototype = Vec::with_capacity(n);
    for k in 0..n {
        let theta = std::f64::consts::PI * (-(n as f64) + 1.0 + 2.0 * k as f64) / (2.0 * n as f64);
        // -sinh(mu + j*theta) without pulling in a complex crate.
        let re = -mu.sinh() * theta.cos();
        let im = -(mu.cosh()) * theta.sin();
        prototype.push(Cplx { re, im });
    }

    // Prototype gain: the product of the negated poles, de-rated for the DC droop
    // that an even-order Chebyshev has at the passband edge.
    let mut k_proto = Cplx::ONE;
    for pole in &prototype {
        k_proto = k_proto.mul(pole.neg());
    }
    let k_proto = k_proto.re / (1.0 + eps * eps).sqrt();

    // Prewarp so the digital cutoff lands exactly on `cutoff_hz`, then scale the
    // prototype and take the bilinear transform.
    let fs2 = 2.0 * sample_rate;
    let wo = fs2 * (std::f64::consts::PI * cutoff_hz / sample_rate).tan();
    let mut poles = Vec::with_capacity(n);
    let mut denom = Cplx::ONE;
    for pole in &prototype {
        let analog = Cplx {
            re: pole.re * wo,
            im: pole.im * wo,
        };
        denom = denom.mul(Cplx { re: fs2, im: 0.0 }.sub(analog));
        poles.push(
            Cplx { re: fs2, im: 0.0 }
                .add(analog)
                .div(Cplx { re: fs2, im: 0.0 }.sub(analog)),
        );
    }
    let k_z = k_proto * wo.powi(n as i32) / denom.re;

    // Poles come out as conjugate pairs mirrored about the middle of the list.
    let mut pairs: Vec<Cplx> = (0..n / 2).map(|i| poles[i]).collect();
    for (i, pole) in pairs.iter().enumerate() {
        let mirror = poles[n - 1 - i];
        if (pole.re - mirror.re).abs() > 1e-9 || (pole.im + mirror.im).abs() > 1e-9 {
            return Err(Error::Filter(
                "chebyshev poles did not come out in conjugate pairs".into(),
            ));
        }
    }
    // scipy pairs the "worst" pole (closest to the unit circle) with the last
    // section, so the cascade runs from the pole furthest inside the circle.
    pairs.sort_by(|a, b| {
        let da = (1.0 - a.abs()).abs();
        let db = (1.0 - b.abs()).abs();
        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut sos = Vec::with_capacity(n / 2);
    for pole in pairs {
        sos.push(Sos {
            b: [1.0, 2.0, 1.0],
            a: [1.0, -2.0 * pole.re, pole.abs().powi(2)],
        });
    }
    sos[0].b[0] *= k_z;
    sos[0].b[1] *= k_z;
    sos[0].b[2] *= k_z;
    Ok(sos)
}

#[derive(Clone, Copy, Debug)]
struct Cplx {
    re: f64,
    im: f64,
}

impl Cplx {
    const ONE: Self = Self { re: 1.0, im: 0.0 };

    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    fn div(self, other: Self) -> Self {
        let d = other.re * other.re + other.im * other.im;
        Self {
            re: (self.re * other.re + self.im * other.im) / d,
            im: (self.im * other.re - self.re * other.im) / d,
        }
    }

    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }

    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn given_the_reference_parameters_when_designed_then_dc_gain_is_the_ripple_floor() {
        let sos = cheby1_lowpass(8, 0.1, 15_328.0, 48_000.0).unwrap();
        assert_eq!(sos.len(), 4);
        let dc: f64 = sos.iter().map(|s| s.dc_gain()).product();
        // An even-order Chebyshev I sits 0.1 dB down at DC.
        assert!((dc - 0.988_553_1).abs() < 1e-6, "dc gain {dc}");
    }

    #[test]
    fn given_scipy_sos_for_a_known_cutoff_when_designed_then_the_sections_match() {
        // Printed from scipy 1.18.0: signal.cheby1(8, 0.1, 15328/24000, output="sos")
        let sos = cheby1_lowpass(8, 0.1, 15_328.0, 48_000.0).unwrap();
        let expected = [
            [
                0.020_297_707_921_188_568,
                0.040_595_415_842_377_136,
                0.020_297_707_921_188_568,
                1.0,
                -0.542_218_317_748_944_9,
                0.147_524_265_115_897_34,
            ],
            [
                1.0,
                2.0,
                1.0,
                1.0,
                0.016_620_298_844_417_795,
                0.405_838_617_965_300_9,
            ],
            [
                1.0,
                2.0,
                1.0,
                1.0,
                0.545_883_274_433_073_4,
                0.676_808_229_484_996_2,
            ],
            [
                1.0,
                2.0,
                1.0,
                1.0,
                0.851_342_238_905_575_6,
                0.895_243_126_070_636_7,
            ],
        ];
        assert_eq!(sos.len(), expected.len());
        for (got, want) in sos.iter().zip(expected.iter()) {
            for (g, w) in got.b.iter().chain(got.a.iter()).zip(want.iter()) {
                assert!((g - w).abs() < 1e-12, "got {got:?} want {want:?}");
            }
        }
    }

    #[test]
    fn given_a_cutoff_at_or_above_nyquist_when_designed_then_it_is_rejected() {
        assert!(cheby1_lowpass(8, 0.1, 24_000.0, 48_000.0).is_err());
        assert!(cheby1_lowpass(8, 0.1, 0.0, 48_000.0).is_err());
        assert!(cheby1_lowpass(7, 0.1, 15_000.0, 48_000.0).is_err());
    }
}
