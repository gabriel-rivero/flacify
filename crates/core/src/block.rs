use crate::error::{Error, Result};

/// A planar block of audio: one contiguous slice per channel, no interleaving.
///
/// Planar because that is the layout the model wants (`[channels, frames]`), and
/// because the mixer reads each channel independently.
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    channels: usize,
    frames: usize,
    data: Vec<f32>,
}

impl Block {
    pub fn zeros(channels: usize, frames: usize) -> Result<Self> {
        if channels == 0 {
            return Err(Error::NoChannels);
        }
        Ok(Self {
            channels,
            frames,
            data: vec![0.0; channels * frames],
        })
    }

    /// Build from planar data; the length must be a multiple of `channels`.
    pub fn from_planar(channels: usize, data: Vec<f32>) -> Result<Self> {
        if channels == 0 {
            return Err(Error::NoChannels);
        }
        if data.len() % channels != 0 {
            return Err(Error::NotPlanar {
                samples: data.len(),
                channels,
            });
        }
        let frames = data.len() / channels;
        Ok(Self {
            channels,
            frames,
            data,
        })
    }

    /// Build from interleaved data, which is what decoders and devices use.
    pub fn from_interleaved(channels: usize, interleaved: &[f32]) -> Result<Self> {
        if channels == 0 {
            return Err(Error::NoChannels);
        }
        if interleaved.len() % channels != 0 {
            return Err(Error::NotPlanar {
                samples: interleaved.len(),
                channels,
            });
        }
        let frames = interleaved.len() / channels;
        let mut data = vec![0.0; interleaved.len()];
        for (i, sample) in interleaved.iter().enumerate() {
            let frame = i / channels;
            let channel = i % channels;
            data[channel * frames + frame] = *sample;
        }
        Ok(Self {
            channels,
            frames,
            data,
        })
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn frames(&self) -> usize {
        self.frames
    }

    pub fn is_empty(&self) -> bool {
        self.frames == 0
    }

    pub fn channel(&self, channel: usize) -> &[f32] {
        let start = channel * self.frames;
        &self.data[start..start + self.frames]
    }

    pub fn channel_mut(&mut self, channel: usize) -> &mut [f32] {
        let start = channel * self.frames;
        &mut self.data[start..start + self.frames]
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Interleave into `out`, which must be exactly `frames * channels` long.
    pub fn interleave_into(&self, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.frames * self.channels);
        for frame in 0..self.frames {
            for channel in 0..self.channels {
                out[frame * self.channels + channel] = self.data[channel * self.frames + frame];
            }
        }
    }

    /// A sub-range of frames, as a new block.
    pub fn slice(&self, start: usize, end: usize) -> Block {
        let end = end.min(self.frames);
        let start = start.min(end);
        let mut data = Vec::with_capacity(self.channels * (end - start));
        for channel in 0..self.channels {
            let base = channel * self.frames;
            data.extend_from_slice(&self.data[base + start..base + end]);
        }
        Block {
            channels: self.channels,
            frames: end - start,
            data,
        }
    }

    /// Pad with zeros up to `frames` frames.
    pub fn padded_to(&self, frames: usize) -> Block {
        if frames <= self.frames {
            return self.slice(0, frames);
        }
        let mut out = vec![0.0; self.channels * frames];
        for channel in 0..self.channels {
            let dst = &mut out[channel * frames..channel * frames + self.frames];
            dst.copy_from_slice(self.channel(channel));
        }
        Block {
            channels: self.channels,
            frames,
            data: out,
        }
    }
}
