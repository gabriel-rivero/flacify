//! Minimal WAV reading for `flacify probe`.
//!
//! symphonia replaces this when the decoder lands; until then this is enough to point
//! `probe` at the fixtures, which are written by scipy as float32 WAVs.

use anyhow::{Context, Result};
use hound::{SampleFormat, WavReader};

pub struct Audio {
    pub sample_rate: u32,
    /// Planar, normalised to [-1, 1].
    pub channels: Vec<Vec<f32>>,
}

pub fn read(path: &std::path::Path) -> Result<Audio> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    if channels == 0 {
        anyhow::bail!("{} has no channels", path.display());
    }

    let interleaved: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("reading float samples")?,
        SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("reading integer samples")?
        }
    };

    let frames = interleaved.len() / channels;
    let mut planar = vec![Vec::with_capacity(frames); channels];
    for frame in 0..frames {
        for (channel, buffer) in planar.iter_mut().enumerate() {
            buffer.push(interleaved[frame * channels + channel]);
        }
    }

    Ok(Audio {
        sample_rate: spec.sample_rate,
        channels: planar,
    })
}
