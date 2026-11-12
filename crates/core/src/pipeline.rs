use crate::block::Block;
use crate::error::{Error, Result};
use crate::upscaler::Upscaler;

/// Turn a stream into an enhanced stream of exactly the same length.
///
/// Windows advance by `window - overlap` samples. The last `overlap` samples of each
/// window are held back and blended into the head of the next one, so the output is
/// continuous and its sample count equals the input's exactly:
///
/// - while at least one full window is buffered, emit `window - overlap` samples;
/// - on `finish`, emit whatever is left, which is at most one window.
///
/// The buffered samples and the held tail describe the same span of input, which is why
/// holding `overlap` samples does not lose any: they are the next window's head.
///
/// Invariants that the tests pin:
/// - output total == input total, for every length and every overlap;
/// - an identity upscaler reproduces the input within one ULP;
/// - the same input and seed produce the same output.
pub struct Pipeline<U: Upscaler> {
    upscaler: U,
    channels: usize,
    window: usize,
    overlap: usize,
    step: usize,
    seed: u64,
    input: Vec<Vec<f32>>,
    tail: Option<Vec<Vec<f32>>>,
    windows: u64,
    finished: bool,
}

impl<U: Upscaler> Pipeline<U> {
    pub fn new(
        upscaler: U,
        channels: usize,
        window: usize,
        overlap: usize,
        seed: u64,
    ) -> Result<Self> {
        if channels == 0 {
            return Err(Error::NoChannels);
        }
        if window < 2 {
            return Err(Error::WindowTooSmall { window });
        }
        // Never overlap more than half a window: two windows must still cover new input.
        let overlap = if window > 2 {
            overlap.min(window / 2 - 1)
        } else {
            0
        };
        Ok(Self {
            upscaler,
            channels,
            window,
            overlap,
            step: window - overlap,
            seed,
            input: vec![Vec::new(); channels],
            tail: None,
            windows: 0,
            finished: false,
        })
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn window(&self) -> usize {
        self.window
    }

    pub fn overlap(&self) -> usize {
        self.overlap
    }

    pub fn step(&self) -> usize {
        self.step
    }

    pub fn windows_rendered(&self) -> u64 {
        self.windows
    }

    pub fn upscaler(&self) -> &U {
        &self.upscaler
    }

    pub fn upscaler_mut(&mut self) -> &mut U {
        &mut self.upscaler
    }

    /// Feed more input. Returns whatever became emittable, which is usually nothing
    /// until a whole window has arrived.
    pub fn push(&mut self, input: &Block) -> Result<Vec<Block>> {
        if self.finished {
            return Err(Error::Finished);
        }
        if input.channels() != self.channels {
            return Err(Error::ChannelMismatch {
                got: input.channels(),
                expected: self.channels,
            });
        }
        for channel in 0..self.channels {
            self.input[channel].extend_from_slice(input.channel(channel));
        }
        let mut out = Vec::new();
        while self.input[0].len() >= self.window {
            out.push(self.render(false)?);
        }
        Ok(out)
    }

    /// Emit everything still buffered. After this the pipeline refuses more input.
    pub fn finish(&mut self) -> Result<Vec<Block>> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;
        if self.input[0].is_empty() {
            // Nothing buffered, so the only thing still owed is the held tail.
            return match self.tail.take() {
                Some(tail) => Ok(vec![Block::from_planar(self.channels, flatten(&tail))?]),
                None => Ok(Vec::new()),
            };
        }
        Ok(vec![self.render(true)?])
    }

    /// Render one window from the head of the buffer and take it out of the buffer.
    fn render(&mut self, is_last: bool) -> Result<Block> {
        let valid = self.input[0].len().min(self.window);

        let mut data = Vec::with_capacity(self.channels * self.window);
        for channel in 0..self.channels {
            data.extend_from_slice(&self.input[channel][..valid]);
            data.resize(data.len() + (self.window - valid), 0.0);
        }
        let chunk = Block::from_planar(self.channels, data)?;

        let advance = self.step.min(valid);
        for buffer in &mut self.input {
            buffer.drain(..advance);
        }

        let seed = self.window_seed();
        self.windows += 1;
        let mut out = self.upscaler.process(&chunk, seed)?;
        if out.channels() != self.channels || out.frames() != self.window {
            return Err(Error::UpscalerShape {
                expected_channels: self.channels,
                expected_frames: self.window,
                got_channels: out.channels(),
                got_frames: out.frames(),
            });
        }

        if let Some(tail) = self.tail.take() {
            let blend = self.overlap.min(valid).min(tail[0].len());
            let fall = curve_fall(blend);
            let rise = curve_rise(blend);
            let frames = out.frames();
            for (samples, previous) in out.data_mut().chunks_mut(frames).zip(&tail) {
                for i in 0..blend {
                    samples[i] = previous[i] * fall[i] + samples[i] * rise[i];
                }
            }
        }

        if is_last {
            return Ok(out.slice(0, valid));
        }
        let keep = valid - self.overlap;
        self.tail = Some(
            (0..self.channels)
                .map(|channel| out.channel(channel)[keep..valid].to_vec())
                .collect(),
        );
        Ok(out.slice(0, keep))
    }

    fn window_seed(&self) -> u64 {
        splitmix64(self.seed ^ self.windows.wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }
}

fn flatten(tail: &[Vec<f32>]) -> Vec<f32> {
    let mut data = Vec::with_capacity(tail.iter().map(Vec::len).sum());
    for channel in tail {
        data.extend_from_slice(channel);
    }
    data
}

/// Rising raised-cosine ramp: 0 -> 1.
pub fn curve_rise(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            0.5 - 0.5 * (std::f32::consts::PI * t).cos()
        })
        .collect()
}

/// Falling raised-cosine ramp: 1 -> 0. `rise[i] + fall[i] == 1`.
pub fn curve_fall(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            0.5 + 0.5 * (std::f32::consts::PI * t).cos()
        })
        .collect()
}

fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
