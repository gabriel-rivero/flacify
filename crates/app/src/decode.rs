//! Decode an audio file into planar f32, then map it to the pipeline's 48 kHz stereo.
//!
//! `AudioStream` pulls one decoded packet at a time at the file's native format; `decode`
//! collects those chunks into one buffer, and `to_stereo_48k` does the channel mapping and
//! the rational resample to the model rate.

use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use flacify_core::dsp::resample::Resampler;
use flacify_core::SAMPLE_RATE;
use symphonia::core::audio::{AudioBufferRef, SampleBuffer, SignalSpec};
use symphonia::core::codecs::{Decoder, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatOptions, FormatReader};
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

/// One decoded packet: `channels * frames` samples, channel-major.
#[derive(Debug)]
pub struct Chunk {
    /// Frames per channel carried by this packet.
    pub frames: usize,
    /// `channels * frames` samples, channel-major.
    pub planar: Vec<f32>,
}

/// A default track pulled one decoded packet at a time, at its native sample rate and
/// channel count.
///
/// Chunks follow the decoder's own packet boundaries, so their size varies; join them
/// channel by channel to get the same planar buffer `decode` returns.
pub struct AudioStream {
    source: PacketSource,
    sample_rate: u32,
    channels: usize,
    pending: Option<Chunk>,
}

impl AudioStream {
    /// Open the default track and pull its first chunk, so the format is known up front.
    pub fn open(path: &Path) -> Result<Self> {
        let mut source = PacketSource::open(path)?;
        let (layout, first) = source
            .pull()?
            .with_context(|| format!("{} decoded no audio", path.display()))?;
        let channels = layout.channels.count();
        anyhow::ensure!(channels > 0, "{} has no channels", path.display());

        Ok(Self {
            source,
            sample_rate: layout.rate,
            channels,
            pending: Some(first),
        })
    }

    /// Sample rate of the stream, in hertz.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Channel count of the stream.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Decode until `wanted` seconds of audio have arrived, or the stream ends.
    ///
    /// A container's own frame count is not enough to tell audio from a file that merely
    /// probed as audio: a disk image decoded as 2.5 s of stereo out of 90 MB. Counting
    /// decoded frames is what actually bounds a file's length, and it stops as soon as
    /// the wanted duration is reached, so a long file costs only that much work.
    pub fn holds_at_least_seconds(&mut self, wanted: f64) -> bool {
        let rate = f64::from(self.sample_rate);
        if rate <= 0.0 {
            return false;
        }
        let mut frames = 0usize;
        while (frames as f64) / rate < wanted {
            match self.next_chunk() {
                Ok(Some(chunk)) => frames = frames.saturating_add(chunk.frames),
                Ok(None) | Err(_) => break,
            }
        }
        (frames as f64) / rate >= wanted
    }

    /// Pull the next decoded packet, or `None` at the end of the stream.
    ///
    /// The first packet fixes the sample rate and channel count; a later packet whose audio
    /// layout differs in any way is an error, so every chunk joins into one planar buffer.
    pub fn next_chunk(&mut self) -> Result<Option<Chunk>> {
        if let Some(chunk) = self.pending.take() {
            return Ok(Some(chunk));
        }
        Ok(self.source.pull()?.map(|(_, chunk)| chunk))
    }
}

/// The demuxer and decoder behind `AudioStream`, plus the layout the first packet fixed.
struct PacketSource {
    path: PathBuf,
    format: Box<dyn FormatReader>,
    track_id: u32,
    decoder: Box<dyn Decoder>,
    layout: Option<SignalSpec>,
}

impl PacketSource {
    /// Probe the file, take its default track and build that track's decoder.
    fn open(path: &Path) -> Result<Self> {
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
        let format = probed.format;

        let track = format
            .default_track()
            .with_context(|| format!("{} has no default track", path.display()))?;
        let track_id = track.id;
        let codec_params = track.codec_params.clone();

        let decoder = symphonia::default::get_codecs()
            .make(&codec_params, &DecoderOptions::default())
            .with_context(|| format!("{} uses an unsupported codec", path.display()))?;

        Ok(Self {
            path: path.to_path_buf(),
            format,
            track_id,
            decoder,
            layout: None,
        })
    }

    /// Pull packets until one decodes, returning the stream's layout and its samples.
    fn pull(&mut self) -> Result<Option<(SignalSpec, Chunk)>> {
        loop {
            let packet = match self.format.next_packet() {
                Ok(packet) => packet,
                Err(SymphoniaError::IoError(error))
                    if error.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    return Ok(None)
                }
                Err(error) => {
                    return Err(error).with_context(|| format!("demuxing {}", self.path.display()))
                }
            };
            if packet.track_id() != self.track_id {
                continue;
            }

            let decoded = match self.decoder.decode(&packet) {
                Ok(decoded) => decoded,
                Err(SymphoniaError::DecodeError(message)) => {
                    tracing::warn!(
                        file = %self.path.display(),
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
                    return Err(error).with_context(|| format!("decoding {}", self.path.display()))
                }
            };

            let layout = *decoded.spec();
            match self.layout {
                Some(expected) => anyhow::ensure!(
                    layout == expected,
                    "{} changes its audio layout mid-stream",
                    self.path.display()
                ),
                None => self.layout = Some(layout),
            }

            return Ok(Some((layout, chunk_from(decoded))));
        }
    }
}

/// Copy one decoded packet into planar f32.
fn chunk_from(decoded: AudioBufferRef<'_>) -> Chunk {
    let frames = decoded.frames();
    let mut buffer = SampleBuffer::<f32>::new(frames as u64, *decoded.spec());
    buffer.copy_planar_ref(decoded);
    Chunk {
        frames,
        planar: buffer.samples().to_vec(),
    }
}

/// What the file says about itself, if anything. Cosmetic only: a missing tag is not
/// an error, and a file with none still plays.
#[derive(Default, Clone)]
pub struct Tags {
    pub title: Option<String>,
    pub artist: Option<String>,
}

pub fn tags(path: &Path) -> Tags {
    use lofty::file::TaggedFileExt;
    use lofty::tag::Accessor;

    let Ok(file) = lofty::read_from_path(path) else {
        return Tags::default();
    };
    let tag = file.primary_tag().or_else(|| file.first_tag());
    let clean = |value: Option<std::borrow::Cow<'_, str>>| {
        value
            .map(|text| text.trim().to_string())
            .filter(|text| !text.is_empty())
    };
    Tags {
        title: tag.and_then(|tag| clean(tag.title())),
        artist: tag.and_then(|tag| clean(tag.artist())),
    }
}

/// Demux and decode every packet of the default track.
///
/// A collector over [`AudioStream`]: accumulate each chunk's plane per channel, then
/// concatenate the planes.
pub fn decode(path: &Path) -> Result<Decoded> {
    let mut stream = AudioStream::open(path)?;
    let mut planes: Vec<Vec<f32>> = vec![Vec::new(); stream.channels()];

    while let Some(chunk) = stream.next_chunk()? {
        for (channel, plane) in planes.iter_mut().enumerate() {
            let start = channel * chunk.frames;
            plane.extend_from_slice(&chunk.planar[start..start + chunk.frames]);
        }
    }

    let frames = planes.first().map_or(0, Vec::len);
    anyhow::ensure!(frames > 0, "{} decoded zero frames", path.display());

    Ok(Decoded {
        sample_rate: stream.sample_rate(),
        channels: stream.channels(),
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
///
/// The whole-file mapping: `probe` uses it so a diagnosis writes nothing to disk, while
/// playback and rendering go through the chunked source tier.
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
    let [left, right] = stereo_planes_of(decoded.channels, &decoded.planar, decoded.frames)?;
    if decoded.channels > 2 {
        tracing::warn!(
            channels = decoded.channels,
            "downmix is not implemented; keeping the first two channels"
        );
    }
    Ok([left.to_vec(), right.to_vec()])
}

/// The left and right planes of one buffer of planar samples: one channel is duplicated
/// to both sides, more than two keeps the first two.
///
/// The caller decides whether keeping the first two of a real downmix is worth a warning,
/// because a whole-file pass warns once and a chunked pass must not warn per chunk.
pub(crate) fn stereo_planes_of(
    channels: usize,
    planar: &[f32],
    frames: usize,
) -> Result<[&[f32]; 2]> {
    anyhow::ensure!(channels > 0, "decoded audio has no channels");
    anyhow::ensure!(
        planar.len() == channels * frames,
        "planar data is {} samples for {channels} channels x {frames} frames",
        planar.len()
    );

    if channels == 1 {
        let mono = &planar[..frames];
        return Ok([mono, mono]);
    }
    Ok([&planar[..frames], &planar[frames..2 * frames]])
}

/// Express `target / source` as `up / down` with the greatest common divisor removed.
pub(crate) fn reduced_ratio(target: u32, source: u32) -> (usize, usize) {
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

pub(crate) fn to_f64(samples: &[f32]) -> Vec<f64> {
    samples.iter().map(|&sample| f64::from(sample)).collect()
}

pub(crate) fn to_f32(samples: &[f64]) -> Vec<f32> {
    samples.iter().map(|&sample| sample as f32).collect()
}

fn concat_planes(left: &[f32], right: &[f32]) -> Vec<f32> {
    let mut planar = Vec::with_capacity(left.len() + right.len());
    planar.extend_from_slice(left);
    planar.extend_from_slice(right);
    planar
}
