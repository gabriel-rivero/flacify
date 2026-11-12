//! DSP that the model preprocessing depends on, ported from the reference
//! implementation. Each module is validated against scipy fixtures in
//! `crates/core/tests/fixtures.rs`.

pub mod cheby;
pub mod cutoff;
pub mod filtfilt;
pub mod resample;
