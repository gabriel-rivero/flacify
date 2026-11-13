//! The audio device side: open an output stream that pulls from the mixer.
//!
//! The callback does exactly one thing per buffer: ask the mixer for interleaved f32,
//! convert to whatever the device wants, and get out of the way. No allocation, no
//! lock, no logging — every buffer it needs was allocated before the stream started,
//! and the UI talks to it through atomics only.
//!
//! If the device does not run at 48 kHz (many do not), both buffers are resampled once
//! here, before playback, so the callback never has to.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use flacify_core::dsp::resample::Resampler;

use crate::frontier::Frontier;
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
    /// The signals themselves, shared with the mixer (an `Arc` clone, not a copy), so
    /// the window can draw a spectrum of what is playing.
    pub display_original: Arc<[f32]>,
    pub display_generated: Option<Arc<[f32]>>,
    pub stream_errors: Arc<AtomicUsize>,
}

impl AudioEngine {
    /// `original` and `enhanced` are interleaved stereo at 48 kHz, same length.
    pub fn start(original: &[f32], enhanced: Option<&[f32]>) -> Result<Self> {
        let output = Output::open()?;
        let prepared = prepare(original, output.rate, output.channels)?;
        let generated = match enhanced {
            Some(buffer) => Some(prepare(buffer, output.rate, output.channels)?),
            None => None,
        };
        let (mixer, controls) = Mixer::new(prepared, generated, output.channels, output.rate);
        output.start(mixer, controls)
    }

    /// Play now, generate behind: the frontier must be 48 kHz stereo, because a
    /// progressive render cannot be resampled per chunk without seams.
    pub fn start_progressive(original: &[f32], frontier: Arc<Frontier>) -> Result<Self> {
        let output = Output::open()?;
        if output.rate != flacify_core::SAMPLE_RATE || output.channels != 2 {
            anyhow::bail!(
                "progressive playback needs a 48 kHz stereo device; this one is {} Hz with {} channels",
                output.rate,
                output.channels
            );
        }
        let prepared = prepare(original, output.rate, output.channels)?;
        let (mixer, controls) =
            Mixer::progressive(prepared, frontier, output.channels, output.rate);
        output.start(mixer, controls)
    }
}

/// The device we found, before deciding what to feed it.
struct Output {
    device: cpal::Device,
    config: cpal::StreamConfig,
    format: cpal::SampleFormat,
    rate: u32,
    channels: usize,
}

impl Output {
    fn open() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .context("no default output device")?;
        let default = device
            .default_output_config()
            .context("asking the output device for its default configuration")?;

        // Prefer a stereo 48 kHz configuration when the device offers one: that is what
        // lets playback start before the render finishes. A 96 kHz default is common and
        // would otherwise force the slow path for no product reason.
        let preferred = device
            .supported_output_configs()
            .ok()
            .and_then(|mut configs| {
                configs
                    .find(|range| {
                        range.channels() == 2
                            && range.min_sample_rate() <= flacify_core::SAMPLE_RATE
                            && range.max_sample_rate() >= flacify_core::SAMPLE_RATE
                    })
                    .map(|range| range.with_sample_rate(flacify_core::SAMPLE_RATE))
            });
        let supported = preferred.unwrap_or(default);
        let rate = supported.sample_rate();
        let channels = supported.channels() as usize;
        let format = supported.sample_format();
        Ok(Self {
            device,
            config: supported.into(),
            format,
            rate,
            channels,
        })
    }

    fn start(mut self, mixer: Mixer, controls: Arc<Controls>) -> Result<AudioEngine> {
        let display_original = Arc::clone(mixer.original_arc());
        let display_generated = mixer.generated_arc();
        let stream_errors = Arc::new(AtomicUsize::new(0));
        // A fixed buffer size is a request, not a guarantee: WASAPI in shared mode and
        // some ALSA configurations refuse it. Ask for the small one, fall back to
        // whatever the device wants, and report the second failure with both contexts.
        let mut stream = None;
        let mut first_error = None;
        for buffer_size in [
            cpal::BufferSize::Fixed(PREFERRED_BUFFER_FRAMES),
            cpal::BufferSize::Default,
        ] {
            self.config.buffer_size = buffer_size;
            match open(
                &self.device,
                &self.config,
                self.format,
                mixer.clone(),
                self.channels,
                Arc::clone(&stream_errors),
            ) {
                Ok(opened) => {
                    stream = Some(opened);
                    break;
                }
                Err(error) => {
                    tracing::warn!(%error, "output stream refused a {:?} buffer", buffer_size);
                    first_error = first_error.or(Some(error));
                }
            }
        }
        let stream = stream.ok_or_else(|| {
            let error = first_error.unwrap_or_else(|| anyhow::anyhow!("no output stream"));
            anyhow::anyhow!("{error:#}")
        })?;

        stream.play().context("starting playback")?;
        Ok(AudioEngine {
            _stream: stream,
            controls,
            device_rate: self.rate,
            device_channels: self.channels,
            display_original,
            display_generated,
            stream_errors,
        })
    }
}

/// Whether progressive playback can work here: chunks are 48 kHz stereo, and
/// resampling them one second at a time would put a seam at every boundary.
pub fn progressive_supported() -> bool {
    match Output::open() {
        Ok(output) => output.rate == flacify_core::SAMPLE_RATE && output.channels == 2,
        Err(_) => false,
    }
}

/// One attempt with one sample format.
fn open(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    format: cpal::SampleFormat,
    mixer: Mixer,
    channels: usize,
    errors: Arc<AtomicUsize>,
) -> Result<cpal::Stream> {
    match format {
        cpal::SampleFormat::I8 => build::<i8>(device, config, mixer, channels, errors),
        cpal::SampleFormat::I16 => build::<i16>(device, config, mixer, channels, errors),
        cpal::SampleFormat::I32 => build::<i32>(device, config, mixer, channels, errors),
        cpal::SampleFormat::I64 => build::<i64>(device, config, mixer, channels, errors),
        cpal::SampleFormat::U8 => build::<u8>(device, config, mixer, channels, errors),
        cpal::SampleFormat::U16 => build::<u16>(device, config, mixer, channels, errors),
        cpal::SampleFormat::U32 => build::<u32>(device, config, mixer, channels, errors),
        cpal::SampleFormat::U64 => build::<u64>(device, config, mixer, channels, errors),
        cpal::SampleFormat::F32 => build::<f32>(device, config, mixer, channels, errors),
        cpal::SampleFormat::F64 => build::<f64>(device, config, mixer, channels, errors),
        other => anyhow::bail!("the output device wants {other}, which is not wired up yet"),
    }
}

fn build<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut mixer: Mixer,
    channels: usize,
    errors: Arc<AtomicUsize>,
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
        move |error| {
            errors.fetch_add(1, Ordering::Relaxed);
            tracing::error!(%error, "output stream error");
        },
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
