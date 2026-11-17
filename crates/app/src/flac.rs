//! Writing 24-bit FLAC.
//!
//! FLAC carries integers, so every sample is rounded to its nearest 24-bit step and
//! clamped at full scale, with no dither: two encodes of one render are byte-identical
//! and the error stays within half a step (−144 dBFS). `flacenc` advertises the short
//! last frame as `STREAMINFO.min_blocksize` while every frame header claims fixed
//! blocking, which makes symphonia reject the stream; the writer restores both
//! block-size fields to the block size before serialising.

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
pub const BLOCK: usize = 4096;

/// Samples per 24-bit step; also the magnitude of the negative full scale.
const FULL_SCALE: f32 = 8_388_608.0;

/// Round one sample to the 24-bit integer FLAC stores, clamping beyond full scale.
fn quantise(sample: f32) -> i32 {
    let scaled = (sample * FULL_SCALE).round();
    scaled.clamp(-FULL_SCALE, FULL_SCALE - 1.0) as i32
}

/// Write planar float32 audio as a 24-bit FLAC.
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

    let source = MemSource::from_samples(&interleaved, channels, 24, sample_rate as usize);
    let config = Encoder::default()
        .into_verified()
        .map_err(|(_, error)| anyhow::anyhow!("FLAC encoder config: {error}"))?;
    let mut stream = flacenc::encode_with_fixed_block_size(&config, source, BLOCK)
        .map_err(|error| anyhow::anyhow!("encoding {} as FLAC: {error}", path.display()))?;
    // Without this, the short last frame makes the stream look variable-blocksize.
    stream
        .stream_info_mut()
        .set_block_sizes(BLOCK, BLOCK)
        .with_context(|| format!("setting the FLAC block size of {}", path.display()))?;

    let mut sink = ByteSink::with_capacity(stream.count_bits());
    stream
        .write(&mut sink)
        .map_err(|error| anyhow::anyhow!("serialising FLAC bits: {error}"))?;
    let mut file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    file.write_all(sink.as_slice())
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}
