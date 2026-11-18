//! Integration tests for the `decode` module, written against real audio files on disk.
//!
//! `decode` and `flac` live in the binary, so the modules are mounted here by path; this
//! keeps the tests runnable without changing `main.rs`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use hound::{SampleFormat, WavSpec, WavWriter};

#[path = "../src/decode.rs"]
mod decode;
#[allow(dead_code)] // only the FLAC writer is used here
#[path = "../src/flac.rs"]
mod flac;

static NEXT_FILE: AtomicU32 = AtomicU32::new(0);

/// A temp path that deletes itself when the test ends.
struct TempPath {
    path: PathBuf,
}

impl TempPath {
    fn new(name: &str, extension: &str) -> Self {
        let unique = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "flacify-decode-{}-{unique}-{name}.{extension}",
            std::process::id()
        ));
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Collect every chunk a stream yields.
fn stream_chunks(stream: &mut decode::AudioStream) -> Vec<decode::Chunk> {
    let mut chunks = Vec::new();
    while let Some(chunk) = stream.next_chunk().expect("streaming a decoded packet") {
        chunks.push(chunk);
    }
    chunks
}

/// Join chunks the way `decode` collects them: one plane per channel, packet by packet.
fn join_chunks(chunks: &[decode::Chunk], channels: usize) -> Vec<f32> {
    let mut planes = vec![Vec::new(); channels];
    for chunk in chunks {
        for (channel, plane) in planes.iter_mut().enumerate() {
            let start = channel * chunk.frames;
            plane.extend_from_slice(&chunk.planar[start..start + chunk.frames]);
        }
    }
    planes.concat()
}

/// The exact bit pattern of every sample, so the comparison catches one changed LSB.
fn sample_bits(samples: &[f32]) -> Vec<u32> {
    samples.iter().map(|sample| sample.to_bits()).collect()
}

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

#[test]
fn given_a_48k_stereo_wav_when_streamed_then_the_joined_chunks_equal_the_whole_decode() {
    let frames = 1_000;
    let file = TempWav::write("48k-stereo-stream", 48_000, 2, frames, |frame, channel| {
        0.1 * (frame + channel) as f32
    });

    let decoded = decode::decode(file.path()).expect("decoding a 48 kHz stereo wav");
    let mut stream = decode::AudioStream::open(file.path()).expect("opening a 48 kHz stereo wav");
    assert_eq!(stream.sample_rate(), 48_000);
    assert_eq!(stream.channels(), 2);

    let chunks = stream_chunks(&mut stream);
    let streamed_frames: usize = chunks.iter().map(|chunk| chunk.frames).sum();

    assert_eq!(streamed_frames, decoded.frames);
    assert_eq!(
        sample_bits(&join_chunks(&chunks, 2)),
        sample_bits(&decoded.planar),
        "the streamed samples differ from the whole decode"
    );
}

#[test]
fn given_a_48k_mono_wav_when_streamed_then_the_joined_chunks_equal_the_whole_decode() {
    let frames = 500;
    let file = TempWav::write("48k-mono-stream", 48_000, 1, frames, |frame, _| {
        0.001 * frame as f32
    });

    let decoded = decode::decode(file.path()).expect("decoding a 48 kHz mono wav");
    let mut stream = decode::AudioStream::open(file.path()).expect("opening a 48 kHz mono wav");
    assert_eq!(stream.channels(), 1);

    let chunks = stream_chunks(&mut stream);
    let streamed_frames: usize = chunks.iter().map(|chunk| chunk.frames).sum();

    assert_eq!(streamed_frames, decoded.frames);
    assert_eq!(
        sample_bits(&join_chunks(&chunks, 1)),
        sample_bits(&decoded.planar),
        "the streamed samples differ from the whole decode"
    );
}

#[test]
fn given_a_48k_three_channel_wav_when_streamed_then_the_joined_chunks_equal_the_whole_decode() {
    let frames = 400;
    let file = TempWav::write(
        "48k-3ch-stream",
        48_000,
        3,
        frames,
        |_, channel| match channel {
            0 => 0.1,
            1 => 0.2,
            _ => 0.3,
        },
    );

    let decoded = decode::decode(file.path()).expect("decoding a three channel wav");
    let mut stream = decode::AudioStream::open(file.path()).expect("opening a three channel wav");
    assert_eq!(stream.channels(), 3);

    let chunks = stream_chunks(&mut stream);
    let streamed_frames: usize = chunks.iter().map(|chunk| chunk.frames).sum();

    assert_eq!(streamed_frames, decoded.frames);
    assert_eq!(
        sample_bits(&join_chunks(&chunks, 3)),
        sample_bits(&decoded.planar),
        "the streamed samples differ from the whole decode"
    );
}

#[test]
fn given_a_44k1_stereo_wav_when_streamed_then_it_reports_the_native_rate_and_matches_the_whole_decode(
) {
    let frames = 4_410;
    let file = TempWav::write("44k1-stereo-stream", 44_100, 2, frames, |frame, _| {
        (frame as f32 * 0.01).sin() * 0.1
    });

    let decoded = decode::decode(file.path()).expect("decoding a 44.1 kHz stereo wav");
    let mut stream = decode::AudioStream::open(file.path()).expect("opening a 44.1 kHz stereo wav");

    assert_eq!(stream.sample_rate(), 44_100);
    assert_eq!(stream.channels(), 2);

    let chunks = stream_chunks(&mut stream);
    let streamed_frames: usize = chunks.iter().map(|chunk| chunk.frames).sum();

    assert_eq!(streamed_frames, decoded.frames);
    assert_eq!(
        sample_bits(&join_chunks(&chunks, 2)),
        sample_bits(&decoded.planar),
        "the streamed samples differ from the whole decode"
    );
}

/// A stereo signal whose samples are all exact 24-bit values: small integers scaled by a
/// power of two.
fn exact_24_bit_signal(frames: usize) -> Vec<f32> {
    let mut planar = Vec::with_capacity(2 * frames);
    for frame in 0..frames {
        planar.push(((frame as i32 * 7_919) % 2_048 - 1_024) as f32 / 8_388_608.0);
    }
    for frame in 0..frames {
        planar.push(((frame as i32 * 104_729) % 4_096 - 2_048) as f32 / 8_388_608.0);
    }
    planar
}

#[test]
fn given_a_flac_from_the_cache_writer_when_streamed_then_it_reports_48k_stereo_24_bit_and_matches_the_whole_decode(
) {
    let frames = 10_003;
    let planar = exact_24_bit_signal(frames);
    let file = TempPath::new("cache", "flac");
    flac::write(file.path(), 48_000, 2, &planar).expect("writing the cache flac");

    let decoded = decode::decode(file.path()).expect("decoding the cache flac");
    let mut stream = decode::AudioStream::open(file.path()).expect("opening the cache flac");

    assert_eq!(stream.sample_rate(), 48_000);
    assert_eq!(stream.channels(), 2);

    let chunks = stream_chunks(&mut stream);
    let streamed_frames: usize = chunks.iter().map(|chunk| chunk.frames).sum();
    let joined = join_chunks(&chunks, 2);

    assert!(
        chunks.len() > 1,
        "one packet would not exercise a packet boundary"
    );
    assert_eq!(streamed_frames, decoded.frames);
    assert_eq!(
        sample_bits(&joined),
        sample_bits(&decoded.planar),
        "the streamed samples differ from the whole decode"
    );
    for (index, sample) in joined.iter().enumerate() {
        let steps = f64::from(*sample) * 8_388_608.0;
        assert!(
            steps.fract() == 0.0,
            "sample {index}: {sample} is not a 24-bit value"
        );
    }
}

#[test]
fn given_a_file_that_is_not_audio_when_streamed_then_the_error_names_the_path() {
    let file = TempPath::new("not-audio", "wav");
    std::fs::write(file.path(), b"this is not a RIFF file").expect("writing junk");

    let error = match decode::AudioStream::open(file.path()) {
        Ok(_) => panic!("junk is not audio"),
        Err(error) => error,
    };
    let message = format!("{error:#}");

    assert!(message.contains("probing"), "{message}");
    assert!(
        message.contains(&file.path().display().to_string()),
        "{message}"
    );
}

#[test]
fn given_a_short_file_when_asked_for_a_longer_one_then_the_probe_says_it_is_short() {
    // The check a batch walk uses to refuse a container that probed as audio but cannot
    // hold the bytes it occupies: a file has to deliver the audio its size promises.
    let file = TempPath::new("too-short", "wav");
    let spec = WavSpec {
        channels: 2,
        sample_rate: 48_000,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(file.path(), spec).expect("creating the wav");
    for sample in 0..4_800 {
        writer
            .write_sample((sample % 1000) as i16)
            .expect("writing a sample");
    }
    writer.finalize().expect("finalising the wav");

    let mut stream = decode::AudioStream::open(file.path()).expect("opening the wav");
    assert!(stream.holds_at_least_seconds(0.05), "it holds 0.1 s");
    let mut stream = decode::AudioStream::open(file.path()).expect("opening the wav");
    assert!(!stream.holds_at_least_seconds(10.0), "it cannot hold 10 s");
}
