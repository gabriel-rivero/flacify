use crate::block::Block;
use crate::error::Result;

/// One forward pass of the model over one window.
///
/// The trait is deliberately small: the pipeline owns windowing, crossfading and
/// the sample clock, so an engine only has to map a block to a block of the same
/// shape. `seed` exists because FlashSR's one-step sampler starts from noise; an
/// engine that ignores it must still be deterministic.
pub trait Upscaler: Send {
    fn process(&mut self, input: &Block, seed: u64) -> Result<Block>;

    fn name(&self) -> &str {
        "upscaler"
    }
}

/// Test and bring-up engines. None of these enhance anything.
pub mod mocks {
    use super::*;

    /// Returns the input unchanged. The pipeline must reproduce it sample for sample.
    pub struct Identity;

    impl Upscaler for Identity {
        fn process(&mut self, input: &Block, _seed: u64) -> Result<Block> {
            Ok(input.clone())
        }

        fn name(&self) -> &str {
            "identity"
        }
    }

    /// Applies a constant gain, to prove the loudness and clipping paths.
    pub struct Gain(pub f32);

    impl Upscaler for Gain {
        fn process(&mut self, input: &Block, _seed: u64) -> Result<Block> {
            let mut out = input.clone();
            for sample in out.data_mut() {
                *sample *= self.0;
            }
            Ok(out)
        }

        fn name(&self) -> &str {
            "gain"
        }
    }

    /// Shifts content later in time by `frames`, which is what a real engine does.
    /// Used to prove the A/B delay line actually aligns two branches.
    pub struct Delay(pub usize);

    impl Upscaler for Delay {
        fn process(&mut self, input: &Block, _seed: u64) -> Result<Block> {
            let frames = input.frames();
            let shift = self.0.min(frames);
            let mut out = Block::zeros(input.channels(), frames)?;
            for channel in 0..input.channels() {
                out.channel_mut(channel)[shift..]
                    .copy_from_slice(&input.channel(channel)[..frames - shift]);
            }
            Ok(out)
        }

        fn name(&self) -> &str {
            "delay"
        }
    }

    /// Adds deterministic noise derived from the seed. Proves reproducibility.
    pub struct Noise(pub f32);

    impl Upscaler for Noise {
        fn process(&mut self, input: &Block, seed: u64) -> Result<Block> {
            let mut out = input.clone();
            let mut state = seed | 1;
            for sample in out.data_mut() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let unit = ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
                *sample += unit * self.0;
            }
            Ok(out)
        }

        fn name(&self) -> &str {
            "noise"
        }
    }

    /// Succeeds `ok_calls` times, then fails. The pipeline must not hide it.
    pub struct FailAfter {
        pub ok_calls: usize,
        pub calls: usize,
    }

    impl FailAfter {
        pub fn new(ok_calls: usize) -> Self {
            Self { ok_calls, calls: 0 }
        }
    }

    impl Upscaler for FailAfter {
        fn process(&mut self, input: &Block, _seed: u64) -> Result<Block> {
            self.calls += 1;
            if self.calls > self.ok_calls {
                return Err(crate::error::Error::Inference("mock failure".into()));
            }
            Ok(input.clone())
        }

        fn name(&self) -> &str {
            "fail-after"
        }
    }
}
