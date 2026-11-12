//! The model boundary: one window in, one window out, both `[channels, frames]` f32.
//!
//! Everything platform-specific (ONNX Runtime, CoreML, the session cache) lives behind
//! this trait, so the engine and its tests run without a model file.

use flacify_core::{Error, Result};

pub trait Model: Send {
    /// Run one window. `input` is planar `[channels * frames]` and the return value
    /// must have the same length.
    fn run(
        &mut self,
        input: &[f32],
        noise: &[f32],
        channels: usize,
        frames: usize,
    ) -> Result<Vec<f32>>;

    /// Number of noise samples for a window of this shape.
    ///
    /// FlashSR draws its starting noise on the VAE's latent grid, so the count is not
    /// the audio length. The engine asks instead of assuming, which lets the ONNX
    /// session report the graph's real input size without this trait changing. The
    /// mock models below take one noise sample per audio sample.
    fn noise_len(&self, channels: usize, frames: usize) -> usize {
        channels * frames
    }

    fn name(&self) -> &str {
        "model"
    }
}

/// Returns the input untouched. Proves the engine's plumbing without a network.
pub struct IdentityModel;

impl Model for IdentityModel {
    fn run(
        &mut self,
        input: &[f32],
        _noise: &[f32],
        _channels: usize,
        _frames: usize,
    ) -> Result<Vec<f32>> {
        Ok(input.to_vec())
    }

    fn name(&self) -> &str {
        "identity-model"
    }
}

/// Runs a fixed function over input and noise, so a test can tell whether the engine
/// passed the right buffer and whether the noise is what the seed says it should be.
pub struct FixedModel;

impl Model for FixedModel {
    fn run(
        &mut self,
        input: &[f32],
        noise: &[f32],
        _channels: usize,
        _frames: usize,
    ) -> Result<Vec<f32>> {
        if input.len() != noise.len() {
            return Err(Error::Inference(format!(
                "noise is {} samples but the window is {}",
                noise.len(),
                input.len()
            )));
        }
        Ok(input.iter().zip(noise).map(|(x, n)| x + 0.5 * n).collect())
    }

    fn name(&self) -> &str {
        "fixed-model"
    }
}
