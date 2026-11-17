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
///
/// A range renders part of a track on its own: `start_at` places it on the window
/// lattice, `end_range` hands its crossfade tail to the next range, and only `finish`
/// ends the track. A track rendered in ranges that begin on `step` equals a single
/// render, sample for sample.
pub struct Pipeline<U: Upscaler> {
    upscaler: U,
    channels: usize,
    window: usize,
    overlap: usize,
    step: usize,
    seed: u64,
    input: Vec<Vec<f32>>,
    tail: Option<Tail>,
    windows: u64,
    finished: bool,
}

/// The generated tail a window holds back for the next window's crossfade.
///
/// A range that stops mid-track hands one to the range that follows, so that range's
/// first window blends into the samples a whole-track render would have produced.
#[derive(Clone, Debug, PartialEq)]
pub struct Tail {
    next_window: u64,
    samples: Block,
}

impl Tail {
    /// The window index this tail crossfades into, which `start_at` has to be given.
    pub fn next_window(&self) -> u64 {
        self.next_window
    }

    /// The held frames; there are `overlap` of them per channel.
    pub fn samples(&self) -> &Block {
        &self.samples
    }
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

    /// The absolute index of the next window, which is its whole-track index once
    /// `start_at` has run.
    pub fn windows_rendered(&self) -> u64 {
        self.windows
    }

    /// Position the pipeline at `windows`, the absolute index of its next window, and
    /// start from `tail` when the previous range ended with one. Seeds then follow the
    /// whole-track schedule. Call it before the first push of each range.
    pub fn start_at(&mut self, windows: u64, tail: Option<Tail>) -> Result<()> {
        let is_mid_range =
            self.tail.is_some() || self.input.iter().any(|channel| !channel.is_empty());
        if is_mid_range {
            return Err(Error::MidRange);
        }
        if let Some(tail) = tail {
            self.check_tail(windows, &tail)?;
            self.tail = Some(tail);
        }
        self.windows = windows;
        self.finished = false;
        Ok(())
    }

    /// End a range at a window boundary, mid-track: emit nothing more and hand back the
    /// tail the next range starts from. The end of the track goes through `finish`.
    ///
    /// A mid-track range of `n` steps is fed `n * step + overlap` frames, because its last
    /// window reads the `overlap` frames that also open the window after the range.
    pub fn end_range(&mut self) -> Result<Tail> {
        if self.finished {
            return Err(Error::Finished);
        }
        let buffered = self.input[0].len();
        if buffered != self.overlap {
            return Err(Error::RangeNotAligned {
                buffered,
                expected: self.overlap,
            });
        }
        let tail = self.tail.take().ok_or(Error::EmptyRange)?;
        for channel in &mut self.input {
            channel.clear();
        }
        self.finished = true;
        Ok(tail)
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
            out.push(self.render_holding()?);
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
            return Ok(self
                .tail
                .take()
                .map(|tail| vec![tail.samples])
                .unwrap_or_default());
        }
        Ok(vec![self.render_final()?])
    }

    /// Render one window that another window follows: hold its last `overlap` frames as
    /// the next window's crossfade and emit the rest.
    fn render_holding(&mut self) -> Result<Block> {
        let (out, valid) = self.process_window()?;
        debug_assert_eq!(valid, self.window, "push renders only full windows");
        let keep = valid - self.overlap;
        self.tail = Some(Tail {
            next_window: self.windows,
            samples: out.slice(keep, valid),
        });
        Ok(out.slice(0, keep))
    }

    /// Render the last window of the track, which may be short: emit every frame of real
    /// input and hold nothing back.
    fn render_final(&mut self) -> Result<Block> {
        let (out, valid) = self.process_window()?;
        Ok(out.slice(0, valid))
    }

    /// Render the head window of the buffer and take the step's input out of it. Returns
    /// the window's output and how many of its frames came from real input.
    fn process_window(&mut self) -> Result<(Block, usize)> {
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
            let blend = self.overlap.min(valid).min(tail.samples.frames());
            let fall = curve_fall(blend);
            let rise = curve_rise(blend);
            for channel in 0..self.channels {
                let previous = tail.samples.channel(channel);
                let samples = out.channel_mut(channel);
                for i in 0..blend {
                    samples[i] = previous[i] * fall[i] + samples[i] * rise[i];
                }
            }
        }
        Ok((out, valid))
    }

    /// A tail is only usable by a range at the window it crossfades into, and only by a
    /// pipeline with the window and overlap that produced it.
    fn check_tail(&self, windows: u64, tail: &Tail) -> Result<()> {
        if tail.samples.frames() != self.overlap || tail.samples.channels() != self.channels {
            return Err(Error::TailShape {
                expected_channels: self.channels,
                expected_frames: self.overlap,
                got_channels: tail.samples.channels(),
                got_frames: tail.samples.frames(),
            });
        }
        if tail.next_window != windows {
            return Err(Error::TailFromWindow {
                start_window: windows,
                tail_window: tail.next_window,
            });
        }
        Ok(())
    }

    fn window_seed(&self) -> u64 {
        splitmix64(self.seed ^ self.windows.wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }
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
