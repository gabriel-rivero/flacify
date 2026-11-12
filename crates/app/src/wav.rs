//! Writing 32-bit float WAV.
//!
//! Reading is `decode::decode`, which speaks every format symphonia knows; this module
//! only exists because the output of a render should be something a person can open.

use anyhow::{Context, Result};

/// Write planar float32 audio as a 32-bit float WAV.
///
/// Float, not 16-bit: the whole product argument is about content above 16 kHz, and
/// rounding to 16 bits would put a floor under any A/B comparison.
pub fn write(
    path: &std::path::Path,
    sample_rate: u32,
    channels: usize,
    planar: &[f32],
) -> Result<()> {
    if channels == 0 || planar.len() % channels != 0 {
        anyhow::bail!(
            "planar buffer of {} samples does not divide into {channels} channels",
            planar.len()
        );
    }
    let spec = hound::WavSpec {
        channels: channels as u16,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let frames = planar.len() / channels;
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("creating {}", path.display()))?;
    for frame in 0..frames {
        for channel in 0..channels {
            writer.write_sample(planar[channel * frames + frame])?;
        }
    }
    writer
        .finalize()
        .with_context(|| format!("finalising {}", path.display()))?;
    Ok(())
}
