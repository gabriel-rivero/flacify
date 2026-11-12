//! The `Upscaler` the pipeline drives: preprocess, add noise, call the model.
//!
//! The reference implementation hands the model an eighth-order-Chebyshev-filtered
//! window, band-limited by a resample round trip at twice the detected cutoff, and gives
//! the sampler a noise tensor of the same shape. This engine does the same, in that
//! order, and nothing else: windowing and crossfading belong to the pipeline.

use flacify_core::{Block, Error, Preprocessor, Result, Upscaler};

use crate::model::Model;
use crate::noise::fill_normal;

pub struct Engine<M: Model> {
    model: M,
    channels: usize,
    window: usize,
    preprocessor: Option<Preprocessor>,
}

impl<M: Model> Engine<M> {
    /// `cutoff_hz: None` means the source is already full-band and must not be
    /// preprocessed; the model is then called on the signal as it is.
    pub fn new(model: M, channels: usize, window: usize, cutoff_hz: Option<u32>) -> Result<Self> {
        let preprocessor = match cutoff_hz {
            Some(cutoff) => Some(Preprocessor::new(cutoff)?),
            None => None,
        };
        Ok(Self {
            model,
            channels,
            window,
            preprocessor,
        })
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn preprocessor(&self) -> Option<&Preprocessor> {
        self.preprocessor.as_ref()
    }
}

impl<M: Model> Upscaler for Engine<M> {
    fn process(&mut self, input: &Block, seed: u64) -> Result<Block> {
        if input.channels() != self.channels || input.frames() != self.window {
            return Err(Error::UpscalerShape {
                expected_channels: self.channels,
                expected_frames: self.window,
                got_channels: input.channels(),
                got_frames: input.frames(),
            });
        }

        let mut prepared = Vec::with_capacity(input.data().len());
        for channel in 0..self.channels {
            let samples = input.channel(channel);
            match &self.preprocessor {
                Some(preprocessor) => {
                    let filtered = preprocessor.process(samples)?;
                    prepared.extend(filtered.into_iter().map(|s| s as f32));
                }
                None => prepared.extend_from_slice(samples),
            }
        }

        let mut noise = vec![0.0f32; self.model.noise_len(self.channels, self.window)];
        fill_normal(&mut noise, seed);

        let output = self
            .model
            .run(&prepared, &noise, self.channels, self.window)?;
        Block::from_planar(self.channels, output)
    }

    fn name(&self) -> &str {
        self.model.name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{FixedModel, IdentityModel};

    fn ramp(channels: usize, frames: usize) -> Block {
        let data: Vec<f32> = (0..channels * frames)
            .map(|i| ((i % frames) as f32 / frames as f32) - 0.5)
            .collect();
        Block::from_planar(channels, data).unwrap()
    }

    #[test]
    fn given_no_preprocessing_when_the_engine_runs_then_the_input_reaches_the_model_untouched() {
        let mut engine = Engine::new(IdentityModel, 2, 1024, None).unwrap();
        let input = ramp(2, 1024);
        let out = engine.process(&input, 42).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn given_the_same_seed_when_the_engine_runs_twice_then_the_output_is_identical() {
        let mut engine = Engine::new(FixedModel, 1, 1024, None).unwrap();
        let input = ramp(1, 1024);
        let a = engine.process(&input, 9).unwrap();
        let b = engine.process(&input, 9).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn given_the_same_seed_when_the_noise_is_expected_then_the_engine_adds_that_noise() {
        let mut engine = Engine::new(FixedModel, 1, 256, None).unwrap();
        let input = ramp(1, 256);
        let out = engine.process(&input, 1234).unwrap();
        let mut noise = vec![0.0f32; 256];
        fill_normal(&mut noise, 1234);
        for (i, sample) in out.channel(0).iter().enumerate() {
            let expected = input.channel(0)[i] + 0.5 * noise[i];
            assert!((sample - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn given_a_window_of_the_wrong_shape_when_the_engine_runs_then_it_refuses() {
        let mut engine = Engine::new(IdentityModel, 2, 1024, None).unwrap();
        let input = ramp(2, 512);
        assert!(engine.process(&input, 0).is_err());
    }

    /// A stand-in for the real graph, whose noise lives on a smaller latent grid.
    struct LatentModel;

    impl Model for LatentModel {
        fn run(
            &mut self,
            input: &[f32],
            noise: &[f32],
            _channels: usize,
            frames: usize,
        ) -> Result<Vec<f32>> {
            assert_eq!(
                noise.len(),
                frames,
                "engine must use the model's noise length"
            );
            Ok(input.to_vec())
        }

        fn noise_len(&self, _channels: usize, frames: usize) -> usize {
            frames
        }
    }

    #[test]
    fn given_a_model_with_a_latent_noise_grid_when_the_engine_runs_then_it_sizes_the_noise() {
        let mut engine = Engine::new(LatentModel, 2, 256, None).unwrap();
        let input = ramp(2, 256);
        let out = engine.process(&input, 5).unwrap();
        assert_eq!(out, input);
    }
}
