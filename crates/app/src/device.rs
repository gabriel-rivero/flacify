//! The audio device side: open an output stream that pulls from the mixer.
//!
//! The callback does exactly one thing per buffer: ask the mixer for interleaved f32,
//! convert to whatever the device wants, and get out of the way. No allocation, no
//! lock, no logging — every buffer it needs was allocated before the stream started,
//! and the UI talks to it through atomics only.
//!
//! If the device does not run at 48 kHz (many do not), both buffers are resampled once
//! here, before playback, so the callback never has to.

use std::sync::Arc;

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use flacify_core::dsp::resample::Resampler;

use crate::player::{Controls, Mixer};

/// Bigger than any buffer a device asked us for in practice; the callback clamps to it
/// rather than allocating.
const MAX_FRAMES: usize = 16_384;
const PREFERRED_BUFFER_FRAMES: u32 = 1024;

pub struct AudioEngine {
    /// Dropping this stops playback.
    _stream: cpal::Stream,
    pub controls: Arc<Controls>,
    pub device_rate: u32,
    pub device_channels: usize,
}

impl AudioEngine {
    /// `original` and `enhanced` are interleaved stereo at 48 kHz, same length.
    pub fn start(original: &[f32], enhanced: Option<&[f32]>) -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .context("no default output device")?;
        let supported = device
            .default_output_config()
            .context("asking the output device for its default configuration")?;

        let device_rate = supported.sample_rate();
        let device_channels = supported.channels() as usize;
        let sample_format = supported.sample_format();

        let original = prepare(original, device_rate, device_channels)?;
        let enhanced = match enhanced {
            Some(buffer) => Some(prepare(buffer, device_rate, device_channels)?),
            None => None,
        };

        let (mixer, controls) = Mixer::new(original, enhanced, device_channels, device_rate);
        let mut config: cpal::StreamConfig = supported.into();
        config.buffer_size = cpal::BufferSize::Fixed(PREFERRED_BUFFER_FRAMES);

        let stream = match sample_format {
            cpal::SampleFormat::F32 => build::<f32>(&device, &config, mixer, device_channels),
            cpal::SampleFormat::I16 => build::<i16>(&device, &config, mixer, device_channels),
            cpal::SampleFormat::U16 => build::<u16>(&device, &config, mixer, device_channels),
            other => anyhow::bail!("the output device wants {other}, which is not wired up yet"),
        }
        .context("building the output stream")?;

        stream.play().context("starting playback")?;
        Ok(Self {
            _stream: stream,
            controls,
            device_rate,
            device_channels,
        })
    }
}

fn build<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut mixer: Mixer,
    channels: usize,
) -> Result<cpal::Stream>
where
    T: cpal::SizedSample + cpal::FromSample<f32>,
{
    let mut scratch = vec![0.0f32; MAX_FRAMES * channels.max(1)];
    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            let len = data.len().min(scratch.len());
            mixer.fill(&mut scratch[..len]);
            for (out, sample) in data.iter_mut().zip(scratch[..len].iter()) {
                *out = T::from_sample(*sample);
            }
            // Only reachable with a device buffer larger than MAX_FRAMES.
            for out in data.iter_mut().skip(len) {
                *out = T::from_sample(0.0f32);
            }
        },
        |error| tracing::error!(%error, "output stream error"),
        None,
    )?;
    Ok(stream)
}

/// Resample once if needed, then map stereo onto the device's channel count.
fn prepare(interleaved: &[f32], device_rate: u32, device_channels: usize) -> Result<Vec<f32>> {
    let resampled = if device_rate == flacify_core::SAMPLE_RATE {
        interleaved.to_vec()
    } else {
        resample_stereo(interleaved, flacify_core::SAMPLE_RATE, device_rate)?
    };
    Ok(remap_channels(&resampled, 2, device_channels))
}

fn resample_stereo(interleaved: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    let divisor = gcd(from, to);
    let resampler = Resampler::new((to / divisor) as usize, (from / divisor) as usize)?;
    let frames = interleaved.len() / 2;
    let mut left = Vec::with_capacity(frames);
    let mut right = Vec::with_capacity(frames);
    for frame in 0..frames {
        left.push(f64::from(interleaved[frame * 2]));
        right.push(f64::from(interleaved[frame * 2 + 1]));
    }
    let left = resampler.process(&left);
    let right = resampler.process(&right);
    if left.len() != right.len() {
        anyhow::bail!(
            "resampling returned {} frames on the left and {} on the right",
            left.len(),
            right.len()
        );
    }
    let mut out = Vec::with_capacity(left.len() * 2);
    for frame in 0..left.len() {
        out.push(left[frame] as f32);
        out.push(right[frame] as f32);
    }
    Ok(out)
}

fn remap_channels(interleaved: &[f32], from: usize, to: usize) -> Vec<f32> {
    if from == to {
        return interleaved.to_vec();
    }
    let frames = interleaved.len() / from.max(1);
    let mut out = vec![0.0f32; frames * to];
    for frame in 0..frames {
        match to {
            1 => {
                let sum: f32 = (0..from)
                    .map(|channel| interleaved[frame * from + channel])
                    .sum();
                out[frame] = sum / from as f32;
            }
            _ => {
                for channel in 0..to.min(from) {
                    out[frame * to + channel] = interleaved[frame * from + channel];
                }
            }
        }
    }
    out
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a.max(1)
    } else {
        gcd(b, a % b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn given_a_mono_device_when_remapped_then_channels_are_averaged() {
        let stereo = vec![1.0, 0.0, 0.5, 0.5];
        assert_eq!(remap_channels(&stereo, 2, 1), vec![0.5, 0.5]);
    }

    #[test]
    fn given_a_four_channel_device_when_remapped_then_the_extra_channels_are_silent() {
        let stereo = vec![0.25, 0.5, 0.75, 1.0];
        assert_eq!(
            remap_channels(&stereo, 2, 4),
            vec![0.25, 0.5, 0.0, 0.0, 0.75, 1.0, 0.0, 0.0]
        );
    }

    #[test]
    fn given_a_44k1_device_when_resampled_then_the_length_follows_the_ratio() {
        let frames = 4410;
        let interleaved: Vec<f32> = (0..frames).flat_map(|_| [0.5f32, -0.5]).collect();
        let out = resample_stereo(&interleaved, 44_100, 48_000).unwrap();
        assert_eq!(out.len(), 4800 * 2);
        assert!((out[1000] - 0.5).abs() < 1e-3);
    }
}
