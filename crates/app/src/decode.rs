//! Decode an audio file into planar f32, then map it to the pipeline's 48 kHz stereo.
//!
//! `decode` handles any container and codec symphonia has registered; `to_stereo_48k` does
//! the channel mapping and the rational resample to the model rate.

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use flacify_core::dsp::resample::Resampler;
use flacify_core::SAMPLE_RATE;
use symphonia::core::audio::{SampleBuffer, SignalSpec};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Decoded audio in the order the stream carries it.
#[derive(Debug)]
pub struct Decoded {
    /// Sample rate of the stream, in hertz.
    pub sample_rate: u32,
    /// Channel count of the stream.
    pub channels: usize,
    /// Frames per channel; always `planar.len() / channels`.
    pub frames: usize,
    /// `channels * frames` samples, channel-major.
    pub planar: Vec<f32>,
}

/// Demux and decode every packet of the default track.
pub fn decode(path: &Path) -> Result<Decoded> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let source = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(std::ffi::OsStr::to_str) {
        hint.with_extension(extension);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            source,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .with_context(|| format!("probing {}", path.display()))?;
    let mut format = probed.format;

    let track = format
        .default_track()
        .with_context(|| format!("{} has no default track", path.display()))?;
    let track_id = track.id;
    let codec_params = track.codec_params.clone();

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .with_context(|| format!("{} uses an unsupported codec", path.display()))?;

    // One plane per channel, appended packet by packet; the planar output is their
    // concatenation, so channel-major order survives any packet size.
    let mut planes: Vec<Vec<f32>> = Vec::new();
    let mut layout: Option<SignalSpec> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(error) => {
                return Err(error).with_context(|| format!("demuxing {}", path.display()))
            }
        };
        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(message)) => {
                tracing::warn!(
                    file = %path.display(),
                    message,
                    "skipping an undecodable packet"
                );
                continue;
            }
            Err(SymphoniaError::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                continue
            }
            Err(error) => {
                return Err(error).with_context(|| format!("decoding {}", path.display()))
            }
        };

        let packet_spec = *decoded.spec();
        match layout {
            Some(expected) => anyhow::ensure!(
                packet_spec == expected,
                "{} changes its audio layout mid-stream",
                path.display()
            ),
            None => {
                layout = Some(packet_spec);
                planes = vec![Vec::new(); packet_spec.channels.count()];
            }
        }

        let packet_frames = decoded.frames();
        let mut buffer = SampleBuffer::<f32>::new(packet_frames as u64, packet_spec);
        buffer.copy_planar_ref(decoded);
        for (channel, plane) in planes.iter_mut().enumerate() {
            let start = channel * packet_frames;
            plane.extend_from_slice(&buffer.samples()[start..start + packet_frames]);
        }
    }

    let layout = layout.with_context(|| format!("{} decoded no audio", path.display()))?;
    let channels = layout.channels.count();
    anyhow::ensure!(channels > 0, "{} has no channels", path.display());

    let frames = planes.first().map_or(0, Vec::len);
    anyhow::ensure!(frames > 0, "{} decoded zero frames", path.display());

    Ok(Decoded {
        sample_rate: layout.rate,
        channels,
        frames,
        planar: planes.concat(),
    })
}

/// Map to stereo, then polyphase-resample to 48 kHz.
///
/// The result is `2 * frames` planar samples, left then right. Audio already at 48 kHz is
/// returned unchanged. One channel is duplicated to both sides; more than two keeps the
/// first two, which are front-left and front-right for the layouts we expect, so a real
/// downmix is not attempted yet.
pub fn to_stereo_48k(decoded: &Decoded) -> Result<Vec<f32>> {
    let [left, right] = stereo_planes(decoded)?;
    if decoded.sample_rate == SAMPLE_RATE {
        return Ok(concat_planes(&left, &right));
    }

    anyhow::ensure!(
        decoded.sample_rate > 0,
        "cannot resample audio at 0 Hz to {SAMPLE_RATE} Hz"
    );
    let (up, down) = reduced_ratio(SAMPLE_RATE, decoded.sample_rate);
    let resampler = Resampler::new(up, down)?;

    let left_resampled = resampler.process(&to_f64(&left));
    let right_resampled = resampler.process(&to_f64(&right));
    anyhow::ensure!(
        left_resampled.len() == right_resampled.len(),
        "resampler returned {} frames for left and {} for right",
        left_resampled.len(),
        right_resampled.len()
    );

    let planar = concat_planes(&to_f32(&left_resampled), &to_f32(&right_resampled));
    Ok(planar)
}

/// Split the planar input into the left and right planes `to_stereo_48k` resamples.
fn stereo_planes(decoded: &Decoded) -> Result<[Vec<f32>; 2]> {
    anyhow::ensure!(decoded.channels > 0, "decoded audio has no channels");
    anyhow::ensure!(
        decoded.planar.len() == decoded.channels * decoded.frames,
        "planar data is {} samples for {} channels x {} frames",
        decoded.planar.len(),
        decoded.channels,
        decoded.frames
    );

    let frames = decoded.frames;
    if decoded.channels == 1 {
        let mono = &decoded.planar[..frames];
        return Ok([mono.to_vec(), mono.to_vec()]);
    }
    if decoded.channels > 2 {
        tracing::warn!(
            channels = decoded.channels,
            "downmix is not implemented; keeping the first two channels"
        );
    }
    Ok([
        decoded.planar[..frames].to_vec(),
        decoded.planar[frames..2 * frames].to_vec(),
    ])
}

/// Express `target / source` as `up / down` with the greatest common divisor removed.
fn reduced_ratio(target: u32, source: u32) -> (usize, usize) {
    let divisor = gcd(target as usize, source as usize);
    (target as usize / divisor, source as usize / divisor)
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

fn to_f64(samples: &[f32]) -> Vec<f64> {
    samples.iter().map(|&sample| f64::from(sample)).collect()
}

fn to_f32(samples: &[f64]) -> Vec<f32> {
    samples.iter().map(|&sample| sample as f32).collect()
}

fn concat_planes(left: &[f32], right: &[f32]) -> Vec<f32> {
    let mut planar = Vec::with_capacity(left.len() + right.len());
    planar.extend_from_slice(left);
    planar.extend_from_slice(right);
    planar
}
