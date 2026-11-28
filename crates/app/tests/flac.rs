//! Round-trip tests for the FLAC writer, written against real files on disk.
//!
//! `flac` and `decode` live in the binary, so the modules are mounted here by path, as
//! `decode`'s own tests are.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

#[allow(dead_code)] // only the decoder entry point is used here
#[path = "../src/decode.rs"]
mod decode;
#[path = "../src/flac.rs"]
mod flac;

static NEXT_FILE: AtomicU32 = AtomicU32::new(0);

/// A FLAC file in the temp directory that deletes itself when the test ends.
struct TempFlac {
    path: PathBuf,
}

impl TempFlac {
    fn new(name: &str) -> Self {
        let unique = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "flacify-flac-{}-{unique}-{name}.flac",
            std::process::id()
        ));
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempFlac {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// A stereo signal whose samples are all exact 24-bit values: small integers scaled by a
/// power of two. 10 003 frames is two full blocks plus a short last frame.
fn exact_signal(frames: usize) -> Vec<f32> {
    let mut planar = Vec::with_capacity(2 * frames);
    for frame in 0..frames {
        planar.push(((frame as i32 * 7_919) % 2_048 - 1_024) as f32 / 8_388_608.0);
    }
    for frame in 0..frames {
        planar.push(((frame as i32 * 104_729) % 4_096 - 2_048) as f32 / 8_388_608.0);
    }
    planar
}

/// A stereo signal that no 24-bit value represents exactly.
fn unrepresentable_signal(frames: usize) -> Vec<f32> {
    let mut planar = Vec::with_capacity(2 * frames);
    for channel in 0..2 {
        for frame in 0..frames {
            planar.push((frame as f32 * 0.017 + channel as f32 * 0.3).sin() * 0.6);
        }
    }
    planar
}

/// The 24-bit quantisation of a sample, as the decoder returns it. The scale is a power
/// of two, so an exact sample survives the round trip without an epsilon.
fn quantised(sample: f32) -> f32 {
    (f64::from(sample) * 8_388_608.0).round() as f32 / 8_388_608.0
}

fn write_signal(file: &TempFlac, planar: &[f32]) {
    flac::write(file.path(), 48_000, 2, planar).expect("writing the flac");
}

#[test]
fn given_a_signal_exact_at_24_bits_when_round_tripped_then_every_sample_and_the_frame_count_match()
{
    let frames = 10_003;
    let planar = exact_signal(frames);
    let file = TempFlac::new("exact");
    write_signal(&file, &planar);

    let decoded = decode::decode(file.path()).expect("decoding the written flac");

    assert_eq!(decoded.sample_rate, 48_000);
    assert_eq!(decoded.channels, 2);
    assert_eq!(decoded.frames, frames);
    assert_eq!(decoded.planar.len(), planar.len());
    for (index, (&input, &actual)) in planar.iter().zip(&decoded.planar).enumerate() {
        let expected = quantised(input);
        assert_eq!(actual, expected, "sample {index}: {actual} != {expected}");
    }
}

#[test]
fn given_a_signal_not_exact_at_24_bits_when_round_tripped_then_no_sample_is_off_by_more_than_2_pow_minus_23(
) {
    let frames = 12_345;
    let planar = unrepresentable_signal(frames);
    let file = TempFlac::new("inexact");
    write_signal(&file, &planar);

    let decoded = decode::decode(file.path()).expect("decoding the written flac");

    assert_eq!(decoded.channels, 2);
    assert_eq!(decoded.frames, frames);
    assert_eq!(decoded.planar.len(), planar.len());
    let mut max_error = 0.0f64;
    for (&input, &actual) in planar.iter().zip(&decoded.planar) {
        max_error = max_error.max((f64::from(input) - f64::from(actual)).abs());
    }
    assert!(
        max_error <= 2f64.powi(-23),
        "max error {max_error} exceeds 2^-23"
    );
}

#[test]
fn given_a_signal_that_fills_whole_blocks_when_round_tripped_then_no_block_is_lost() {
    // Exactly 338 blocks of 4096: a whole multiple, the boundary a short tail does not
    // exercise. The writer patches both STREAMINFO block sizes, and this is the shape
    // that would expose a tail being miscounted as absent.
    let frames = 338 * flac::BLOCK;
    let planar = unrepresentable_signal(frames);
    let file = TempFlac::new("whole-blocks");
    write_signal(&file, &planar);

    let decoded = decode::decode(file.path()).expect("decoding the written flac");
    assert_eq!(
        decoded.frames,
        frames,
        "a whole-block signal lost {} frames",
        frames as i64 - decoded.frames as i64
    );
}

#[test]
fn given_the_source_tier_segment_length_when_round_tripped_then_no_block_is_lost() {
    // The exact length the source tier writes into one segment: 337 full blocks plus a
    // 3968-frame tail. Pins the writer at the real segment size, not a convenient one.
    let frames = 1_384_320;
    assert_eq!(frames % flac::BLOCK, 3968, "the tail this case is about");
    let planar = unrepresentable_signal(frames);
    let file = TempFlac::new("tier-segment");
    write_signal(&file, &planar);

    let decoded = decode::decode(file.path()).expect("decoding the written flac");
    assert_eq!(
        decoded.frames,
        frames,
        "lost {} frames, one block is {}",
        frames as i64 - decoded.frames as i64,
        flac::BLOCK
    );
}

#[test]
fn given_a_signal_that_is_one_block_short_of_whole_when_round_tripped_then_the_count_holds() {
    // The neighbouring case: the short tail must survive too.
    let frames = 338 * flac::BLOCK - flac::BLOCK;
    let planar = unrepresentable_signal(frames);
    let file = TempFlac::new("short-tail");
    write_signal(&file, &planar);

    let decoded = decode::decode(file.path()).expect("decoding the written flac");
    assert_eq!(decoded.frames, frames);
}

#[test]
fn given_a_written_file_when_streaminfo_is_read_then_both_block_sizes_equal_the_written_block() {
    let frames = 9_000;
    let file = TempFlac::new("streaminfo");
    write_signal(&file, &exact_signal(frames));

    let bytes = std::fs::read(file.path()).expect("reading the written flac");
    assert_eq!(&bytes[..4], b"fLaC", "not a FLAC stream");
    // STREAMINFO is the mandatory first metadata block, so its payload starts at byte 8.
    let min_block_size = u16::from_be_bytes([bytes[8], bytes[9]]);
    let max_block_size = u16::from_be_bytes([bytes[10], bytes[11]]);
    assert_eq!(usize::from(min_block_size), flac::BLOCK);
    assert_eq!(usize::from(max_block_size), flac::BLOCK);
}

#[test]
fn given_a_buffer_that_does_not_fill_whole_channels_when_written_then_it_is_an_error() {
    let file = TempFlac::new("shape");
    let error =
        flac::write(file.path(), 48_000, 2, &[0.0; 3]).expect_err("3 samples are not 2 channels");
    assert!(
        format!("{error:#}").contains("does not divide"),
        "{error:#}"
    );
}

#[test]
fn given_an_empty_buffer_when_written_then_it_fails_instead_of_writing_an_unreadable_file() {
    let file = TempFlac::new("empty");
    let error = flac::write(file.path(), 48_000, 2, &[]).expect_err("an empty stream is not FLAC");
    assert!(format!("{error:#}").contains("no frames"), "{error:#}");
}

#[test]
fn given_the_same_signal_when_written_twice_then_the_files_are_byte_identical() {
    let planar = unrepresentable_signal(4_800);
    let first = TempFlac::new("repeat-a");
    let second = TempFlac::new("repeat-b");
    write_signal(&first, &planar);
    write_signal(&second, &planar);

    let first_bytes = std::fs::read(first.path()).expect("reading the first flac");
    let second_bytes = std::fs::read(second.path()).expect("reading the second flac");
    assert_eq!(first_bytes, second_bytes);
}

#[test]
fn given_a_written_file_when_probed_then_symphonia_reports_48k_stereo_24_bit() {
    let file = TempFlac::new("probe");
    write_signal(&file, &exact_signal(4_800));

    let source = MediaSourceStream::new(
        Box::new(File::open(file.path()).expect("opening the written flac")),
        Default::default(),
    );
    let mut hint = Hint::new();
    hint.with_extension("flac");
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            source,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .expect("probing the written flac");
    let track = probed.format.default_track().expect("a default track");
    let params = &track.codec_params;

    assert_eq!(params.sample_rate, Some(48_000));
    assert_eq!(params.channels.map(|channels| channels.count()), Some(2));
    assert_eq!(params.bits_per_sample, Some(24));
}
