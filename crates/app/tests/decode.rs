//! Integration tests for the `decode` module, written against real WAV files on disk.
//!
//! `decode` lives in the binary, so the module is mounted here by path; this keeps the
//! tests runnable without changing `main.rs`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use hound::{SampleFormat, WavSpec, WavWriter};

#[path = "../src/decode.rs"]
mod decode;

static NEXT_FILE: AtomicU32 = AtomicU32::new(0);

/// A WAV file in the temp directory that deletes itself when the test ends.
struct TempWav {
    path: PathBuf,
}

impl TempWav {
    fn write(
        name: &str,
        sample_rate: u32,
        channels: u16,
        frames: usize,
        sample: impl Fn(usize, usize) -> f32,
    ) -> Self {
        let unique = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "flacify-decode-{}-{unique}-{name}.wav",
            std::process::id()
        ));
        let spec = WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut writer = WavWriter::create(&path, spec).expect("creating a temp wav");
        for frame in 0..frames {
            for channel in 0..usize::from(channels) {
                writer
                    .write_sample(sample(frame, channel))
                    .expect("writing a sample");
            }
        }
        writer.finalize().expect("finalizing a temp wav");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempWav {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[test]
fn given_a_48k_stereo_wav_when_decoded_then_it_reports_the_written_format() {
    let frames = 1_000;
    let file = TempWav::write("48k-stereo", 48_000, 2, frames, |frame, channel| {
        0.1 * (frame + channel) as f32
    });

    let decoded = decode::decode(file.path()).expect("decoding a 48 kHz stereo wav");

    assert_eq!(decoded.sample_rate, 48_000);
    assert_eq!(decoded.channels, 2);
    assert_eq!(decoded.frames, frames);
    assert_eq!(decoded.planar.len(), 2 * frames);
    for frame in 0..frames {
        for channel in 0..2 {
            let expected = 0.1 * (frame + channel) as f32;
            let actual = decoded.planar[channel * frames + frame];
            assert!(
                (actual - expected).abs() < 1e-6,
                "frame {frame} channel {channel}: {actual} != {expected}"
            );
        }
    }

    let planar = decode::to_stereo_48k(&decoded).expect("mapping 48 kHz stereo");
    assert_eq!(planar, decoded.planar);
}

#[test]
fn given_a_44k1_stereo_wav_when_mapped_then_the_length_is_ceil_of_the_ratio() {
    let frames = 4_410;
    let file = TempWav::write("44k1-stereo", 44_100, 2, frames, |_, _| 0.25);

    let decoded = decode::decode(file.path()).expect("decoding a 44.1 kHz stereo wav");
    assert_eq!(decoded.frames, frames);

    let planar = decode::to_stereo_48k(&decoded).expect("resampling to 48 kHz");
    let expected_frames = (decoded.frames * 48_000).div_ceil(44_100);
    assert_eq!(
        planar.len(),
        2 * expected_frames,
        "expected {} frames, measured {}",
        expected_frames,
        planar.len() / 2
    );
}

#[test]
fn given_a_constant_half_signal_when_resampled_then_the_level_is_preserved() {
    let frames = 4_410;
    let file = TempWav::write("44k1-dc", 44_100, 2, frames, |_, _| 0.5);

    let decoded = decode::decode(file.path()).expect("decoding a constant wav");
    let planar = decode::to_stereo_48k(&decoded).expect("resampling to 48 kHz");

    let out_frames = planar.len() / 2;
    let left = &planar[..out_frames];
    // Skip the ramp at each end, where the zero-padded filter is still filling.
    let margin = 256;
    for (frame, sample) in left
        .iter()
        .enumerate()
        .take(out_frames - margin)
        .skip(margin)
    {
        assert!(
            (sample - 0.5).abs() < 1e-3,
            "frame {frame}: {sample} is not 0.5"
        );
    }
}

#[test]
fn given_a_48k_mono_wav_when_mapped_then_both_channels_are_identical() {
    let frames = 500;
    let file = TempWav::write("48k-mono", 48_000, 1, frames, |frame, _| {
        0.001 * frame as f32
    });

    let decoded = decode::decode(file.path()).expect("decoding a 48 kHz mono wav");
    assert_eq!(decoded.channels, 1);
    assert_eq!(decoded.frames, frames);

    let planar = decode::to_stereo_48k(&decoded).expect("mapping mono to stereo");
    assert_eq!(planar.len(), 2 * frames);
    assert_eq!(&planar[..frames], decoded.planar.as_slice());
    assert_eq!(&planar[..frames], &planar[frames..]);
}

#[test]
fn given_a_three_channel_wav_when_mapped_then_the_first_two_channels_are_kept() {
    let frames = 400;
    let file = TempWav::write("48k-3ch", 48_000, 3, frames, |_, channel| match channel {
        0 => 0.1,
        1 => 0.2,
        _ => 0.3,
    });

    let decoded = decode::decode(file.path()).expect("decoding a three channel wav");
    assert_eq!(decoded.channels, 3);

    let planar = decode::to_stereo_48k(&decoded).expect("mapping three channels to stereo");
    assert_eq!(planar.len(), 2 * frames);
    for (frame, (&left, &right)) in planar[..frames].iter().zip(&planar[frames..]).enumerate() {
        assert!((left - 0.1).abs() < 1e-6, "frame {frame} left: {left}");
        assert!((right - 0.2).abs() < 1e-6, "frame {frame} right: {right}");
    }
    assert!(
        !planar.iter().any(|&sample| (sample - 0.3).abs() < 1e-6),
        "the third channel leaked into the stereo output"
    );
}

#[test]
fn given_a_wav_without_tags_when_asked_then_it_reports_none_rather_than_failing() {
    let file = TempWav::write("untagged", 48_000, 2, 480, |frame, _| {
        (frame as f32 * 0.01).sin() * 0.1
    });
    let tags = decode::tags(file.path());
    assert!(tags.title.is_none());
    assert!(tags.artist.is_none());
}

#[test]
fn given_a_missing_file_when_decoded_then_the_error_names_it() {
    let path =
        std::env::temp_dir().join(format!("flacify-decode-missing-{}.wav", std::process::id()));

    let error = decode::decode(&path).expect_err("a missing file must fail");
    let message = format!("{error:#}");

    assert!(message.contains("opening"), "{message}");
    assert!(message.contains(&path.display().to_string()), "{message}");
}

#[test]
fn given_a_wav_with_no_frames_when_decoded_then_it_is_an_error() {
    let file = TempWav::write("empty", 48_000, 2, 0, |_, _| 0.0);

    let error = decode::decode(file.path()).expect_err("a file with no audio must fail");
    let message = format!("{error:#}");

    assert!(message.contains("decoded no audio"), "{message}");
}

#[test]
fn given_a_zero_sample_rate_when_mapped_then_it_is_an_error() {
    let decoded = decode::Decoded {
        sample_rate: 0,
        channels: 2,
        frames: 1,
        planar: vec![0.0, 0.0],
    };

    let error = decode::to_stereo_48k(&decoded).expect_err("0 Hz cannot be resampled");
    let message = format!("{error:#}");

    assert!(message.contains("0 Hz"), "{message}");
}

#[test]
fn given_planar_data_that_does_not_fit_the_shape_when_mapped_then_it_is_an_error() {
    let decoded = decode::Decoded {
        sample_rate: 48_000,
        channels: 2,
        frames: 3,
        planar: vec![0.0; 4],
    };

    let error = decode::to_stereo_48k(&decoded).expect_err("a broken shape must fail");
    let message = format!("{error:#}");

    assert!(message.contains("4 samples"), "{message}");
}
