//! Writing 24-bit FLAC.
//!
//! FLAC carries integers, so every sample is rounded to its nearest 24-bit step and
//! clamped at full scale, with no dither: two encodes of one render are byte-identical
//! and the error stays within half a step (−144 dBFS). `flacenc` advertises the short
//! last frame as `STREAMINFO.min_blocksize` while every frame header claims fixed
//! blocking, which makes symphonia reject the stream; the writer restores both
//! block-size fields to the block size before serialising.
//!
//! **`flacenc` 0.5.1 drops a whole frame on some inputs, and which block size hits it
//! depends on the data.** Measured 2026-11-28 on one 236 s track: at block 4096 segment 1
//! lost 4096 frames and wrote 33.6 MB (≈ `4096 × 4096 × 2`) of nonsense; at 3456 segment
//! 6 lost 3456; at 3072 segment 7 lost 3072. Every failure was exactly one block short,
//! so the encoder is dropping the frame that contains a particular sample. 2304 and below
//! verified on all nine segments, so the block size is 2304 — and [`write`] decodes what
//! it wrote and refuses to publish anything that does not read back as the frames that
//! went in, because the next input that trips this should fail the write, not a track.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use flacenc::bitsink::ByteSink;
use flacenc::component::BitRepr;
use flacenc::config::Encoder;
use flacenc::error::Verify;
use flacenc::source::MemSource;

/// The fixed block size the encoder writes, and the value both `STREAMINFO` fields report.
///
/// 2304 is the largest size that verified on every segment of the track that exposed the
/// bug. 4096 is the crate's own default, which is what made this easy to hit; 3456 and
/// 3072 each dropped a frame on one segment of the same file. Nothing else depends on the
/// value: a segment is a whole number of frames, not of blocks.
pub const BLOCK: usize = 2304;

/// Samples per 24-bit step; also the magnitude of the negative full scale.
const FULL_SCALE: f32 = 8_388_608.0;

/// Round one sample to the 24-bit integer FLAC stores, clamping beyond full scale.
fn quantise(sample: f32) -> i32 {
    let scaled = (sample * FULL_SCALE).round();
    scaled.clamp(-FULL_SCALE, FULL_SCALE - 1.0) as i32
}

/// Write planar float32 audio as a 24-bit FLAC.
///
/// The file is decoded back and its frame count compared before it is left in place, so
/// a stream the encoder mangled becomes an error here instead of a track that will not
/// play. The verified bytes are what lands; nothing reads a half-written file.
pub fn write(path: &Path, sample_rate: u32, channels: usize, planar: &[f32]) -> Result<()> {
    if channels == 0 || planar.len() % channels != 0 {
        anyhow::bail!(
            "planar buffer of {} samples does not divide into {channels} channels",
            planar.len()
        );
    }
    let frames = planar.len() / channels;
    anyhow::ensure!(frames > 0, "cannot write a FLAC stream with no frames");
    let mut interleaved = Vec::with_capacity(planar.len());
    for frame in 0..frames {
        for channel in 0..channels {
            interleaved.push(quantise(planar[channel * frames + frame]));
        }
    }

    let bytes = encode(&interleaved, channels, sample_rate)?;
    verify(&bytes, frames, channels, sample_rate, path)?;

    let mut file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    file.write_all(&bytes)
        .with_context(|| format!("writing {}", path.display()))
}

/// The FLAC stream for these interleaved 24-bit samples.
fn encode(interleaved: &[i32], channels: usize, sample_rate: u32) -> Result<Vec<u8>> {
    let source = MemSource::from_samples(interleaved, channels, 24, sample_rate as usize);
    let config = Encoder::default()
        .into_verified()
        .map_err(|(_, error)| anyhow::anyhow!("FLAC encoder config: {error}"))?;
    let mut stream = flacenc::encode_with_fixed_block_size(&config, source, BLOCK)
        .map_err(|error| anyhow::anyhow!("encoding as FLAC: {error}"))?;
    // Without this, the short last frame makes the stream look variable-blocksize.
    stream
        .stream_info_mut()
        .set_block_sizes(BLOCK, BLOCK)
        .context("setting the FLAC block size")?;

    let mut sink = ByteSink::with_capacity(stream.count_bits());
    stream
        .write(&mut sink)
        .map_err(|error| anyhow::anyhow!("serialising FLAC bits: {error}"))?;
    Ok(sink.as_slice().to_vec())
}

/// Decode `bytes` and require exactly `frames` of `channels` at `sample_rate`.
///
/// The encoder is a dependency with its own bugs; a stream it produces is only a FLAC
/// once a decoder agrees. This runs once per segment rather than per window, on data
/// already in memory, so it is cheaper than the write it guards.
fn verify(
    bytes: &[u8],
    frames: usize,
    channels: usize,
    sample_rate: u32,
    path: &Path,
) -> Result<()> {
    let decoded = crate::decode::decode_bytes(bytes)
        .with_context(|| format!("reading back the FLAC written for {}", path.display()))?;
    anyhow::ensure!(
        decoded.channels == channels && decoded.sample_rate == sample_rate,
        "{} was written as {sample_rate} Hz × {channels} ch but reads back as {} Hz × {} ch",
        path.display(),
        decoded.sample_rate,
        decoded.channels
    );
    anyhow::ensure!(
        decoded.frames == frames,
        "{} was written with {frames} frames but reads back as {}; the encoder produced a \
         stream that does not decode (flacenc block size {BLOCK})",
        path.display(),
        decoded.frames
    );
    Ok(())
}
