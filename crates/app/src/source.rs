//! The source tier: one sequential pass writes a file as 48 kHz stereo FLAC segments.
//!
//! `cache/source/<source_hash>/` holds `source.json` and `src-NNNNNN.flac` segments of the
//! length the render cache uses, so a reader can decode one frame range instead of the
//! whole track. An entry is built in `<hash>.tmp` and renamed; a manifest its files do not
//! keep up with is an error, not a miss.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use flacify_core::dsp::resample::Resampler;
use flacify_core::SAMPLE_RATE;
use serde::{Deserialize, Serialize};

use crate::cache::{now, Segment};
use crate::decode::{self, AudioStream};

/// The entry layout's own version. A manifest from another version is rebuilt, not read.
pub const VERSION: &str = "flacify-source-1";

/// The one file in an entry that is not audio.
const MANIFEST: &str = "source.json";

/// Frames of the 48 kHz output one waveform peak covers: one second.
const PEAK_FRAMES: usize = SAMPLE_RATE as usize;

/// One second of the waveform: the RMS level of each channel.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SecondPeaks {
    pub left_rms: f32,
    pub right_rms: f32,
}

/// The RMS of each second of planar 48 kHz stereo, last partial second included.
///
/// Summed in sample order, so a whole buffer gives the same values the chunked pass in
/// [`SegmentWriter`] measures.
pub(crate) fn second_peaks(planar: &[f32], frames: usize) -> Vec<SecondPeaks> {
    let mut peaks = Vec::with_capacity(frames.div_ceil(PEAK_FRAMES));
    let mut start = 0;
    while start < frames {
        let end = (start + PEAK_FRAMES).min(frames);
        let mut left_squares = 0.0;
        let mut right_squares = 0.0;
        for frame in start..end {
            let left = f64::from(planar[frame]);
            let right = f64::from(planar[frames + frame]);
            left_squares += left * left;
            right_squares += right * right;
        }
        let count = (end - start) as f64;
        peaks.push(SecondPeaks {
            left_rms: (left_squares / count).sqrt() as f32,
            right_rms: (right_squares / count).sqrt() as f32,
        });
        start = end;
    }
    peaks
}

/// What `source.json` records: what the entry was made from and how it is stored.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Meta {
    pub version: String,
    /// The input's container, from its extension: `mp3`, `flac`, `wav`.
    pub format: String,
    /// Sample rate of the input, in hertz.
    pub native_sample_rate: u32,
    /// Channel count of the input.
    pub native_channels: usize,
    /// Frames of 48 kHz stereo in the entry.
    pub frames: usize,
    /// Every segment of the entry, in order from the first frame.
    pub segments: Vec<Segment>,
    /// Each second of the output, last partial second included.
    pub peaks: Vec<SecondPeaks>,
    /// Frames in every segment but the last, which may be short.
    pub frames_per_segment: usize,
    /// Unix seconds.
    pub created_at: u64,
}

/// The directory that holds every source entry: `<cache>/source`.
#[derive(Debug)]
pub struct Tier {
    root: PathBuf,
    frames_per_segment: usize,
}

impl Tier {
    /// `cache_root` is the render cache's directory; entries land in its `source/`.
    ///
    /// The segment length is passed in rather than derived here so the entry records the
    /// value it was written with and a reader can reject one built for other settings.
    pub fn new(cache_root: &Path, frames_per_segment: usize) -> Result<Self> {
        anyhow::ensure!(
            frames_per_segment > 0,
            "a source segment needs at least one frame"
        );
        Ok(Self {
            root: cache_root.join("source"),
            frames_per_segment,
        })
    }

    /// Where the entry for one source file lives, complete or not.
    pub fn entry_dir(&self, source_hash: &str) -> PathBuf {
        self.root.join(source_hash)
    }

    /// The entry for `input`, built in one pass unless a complete one is already there.
    ///
    /// A directory with no manifest, or one from another version, is rebuilt; a manifest
    /// that parses but does not describe its files is an error.
    pub fn build_or_reuse(&self, input: &Path, source_hash: &str) -> Result<Entry> {
        let dir = self.entry_dir(source_hash);
        if dir.exists() {
            let manifest = dir.join(MANIFEST);
            if manifest.is_file() {
                let meta = read_meta(&manifest)?;
                if meta.version == VERSION {
                    validate(&manifest, &meta, self.frames_per_segment)?;
                    ensure_segments(&manifest, &dir, &meta)?;
                    return Ok(Entry { dir, meta });
                }
            }
            clear(&dir)?;
        }
        self.build(input, source_hash)
    }

    /// One decode+resample pass into `<hash>.tmp`, then the rename that publishes it.
    fn build(&self, input: &Path, source_hash: &str) -> Result<Entry> {
        let temp_dir = self.root.join(format!("{source_hash}.tmp"));
        if temp_dir.exists() {
            clear(&temp_dir)?;
        }
        std::fs::create_dir_all(&temp_dir)
            .with_context(|| format!("creating {}", temp_dir.display()))?;

        let pass = self.write_pass(input, &temp_dir)?;
        let frames = pass.segments.iter().map(|segment| segment.frames).sum();
        let meta = Meta {
            version: VERSION.to_string(),
            format: format_of(input),
            native_sample_rate: pass.native_sample_rate,
            native_channels: pass.native_channels,
            frames,
            segments: pass.segments,
            peaks: pass.peaks,
            frames_per_segment: self.frames_per_segment,
            created_at: now(),
        };
        // Last, so a pass that dies before this leaves a directory nothing reads.
        write_manifest(&temp_dir, &meta)?;

        let dir = self.entry_dir(source_hash);
        if dir.exists() {
            clear(&dir)?;
        }
        std::fs::rename(&temp_dir, &dir)
            .with_context(|| format!("publishing {}", dir.display()))?;
        Ok(Entry { dir, meta })
    }

    /// Decode and resample `input` in order, writing a segment each time one fills.
    fn write_pass(&self, input: &Path, dir: &Path) -> Result<PassOutput> {
        let mut stream = AudioStream::open(input)?;
        let native_sample_rate = stream.sample_rate();
        let native_channels = stream.channels();
        anyhow::ensure!(
            native_sample_rate > 0,
            "{} decodes at 0 Hz",
            input.display()
        );

        // A 48 kHz source reduces to 1/1, which the resampler copies through.
        let (up, down) = decode::reduced_ratio(SAMPLE_RATE, native_sample_rate);
        let resampler = Resampler::new(up, down)
            .with_context(|| format!("resampling {} to {SAMPLE_RATE} Hz", input.display()))?;
        let mut left_resampler = resampler.chunked();
        let mut right_resampler = resampler.chunked();
        let mut writer = SegmentWriter::new(dir, self.frames_per_segment);
        let mut has_warned_downmix = false;

        while let Some(chunk) = stream.next_chunk()? {
            let [chunk_left, chunk_right] =
                decode::stereo_planes_of(native_channels, &chunk.planar, chunk.frames)?;
            if native_channels > 2 && !has_warned_downmix {
                tracing::warn!(
                    channels = native_channels,
                    file = %input.display(),
                    "downmix is not implemented; keeping the first two channels"
                );
                has_warned_downmix = true;
            }
            let left_out =
                decode::to_f32(&left_resampler.process_chunk(&decode::to_f64(chunk_left)));
            let right_out =
                decode::to_f32(&right_resampler.process_chunk(&decode::to_f64(chunk_right)));
            anyhow::ensure!(
                left_out.len() == right_out.len(),
                "resampler returned {} frames for left and {} for right",
                left_out.len(),
                right_out.len()
            );
            writer.push(&left_out, &right_out)?;
        }

        let left_tail = decode::to_f32(&left_resampler.finish());
        let right_tail = decode::to_f32(&right_resampler.finish());
        anyhow::ensure!(
            left_tail.len() == right_tail.len(),
            "resampler returned {} frames for left and {} for right",
            left_tail.len(),
            right_tail.len()
        );
        writer.push(&left_tail, &right_tail)?;

        let (segments, peaks) = writer.finish()?;
        Ok(PassOutput {
            native_sample_rate,
            native_channels,
            segments,
            peaks,
        })
    }
}

/// What one pass produced: the input's native format, and the output's segment table and
/// waveform.
struct PassOutput {
    native_sample_rate: u32,
    native_channels: usize,
    segments: Vec<Segment>,
    peaks: Vec<SecondPeaks>,
}

/// A complete entry on disk: the manifest, validated, and the directory holding it.
#[derive(Debug)]
pub struct Entry {
    dir: PathBuf,
    meta: Meta,
}

impl Entry {
    /// What the entry records about itself.
    pub fn meta(&self) -> &Meta {
        &self.meta
    }

    /// The whole entry as 48 kHz stereo planar samples.
    pub fn read_all(&self) -> Result<Vec<f32>> {
        self.read_range(0, self.meta.frames)
    }

    /// 48 kHz stereo planar samples for `[start, start + frames)`.
    ///
    /// Only the segments the range touches are decoded.
    pub fn read_range(&self, start: usize, frames: usize) -> Result<Vec<f32>> {
        let end = start
            .checked_add(frames)
            .with_context(|| format!("range {start} + {frames} frames overflows"))?;
        let manifest = self.manifest();
        anyhow::ensure!(
            end <= self.meta.frames,
            "range {start}..{end} is past the {} frames of {}",
            self.meta.frames,
            manifest.display()
        );
        if frames == 0 {
            return Ok(Vec::new());
        }

        let first = start / self.meta.frames_per_segment;
        let tail = self
            .meta
            .segments
            .get(first..)
            .with_context(|| format!("{} has no segment {first}", manifest.display()))?;
        let mut left = Vec::with_capacity(frames);
        let mut right = Vec::with_capacity(frames);
        let mut segment_start = first * self.meta.frames_per_segment;
        for segment in tail {
            if segment_start >= end {
                break;
            }
            let decoded = self.decode(segment)?;
            let from = start.saturating_sub(segment_start);
            let to = (end - segment_start).min(segment.frames);
            left.extend_from_slice(&decoded.planar[from..to]);
            right.extend_from_slice(&decoded.planar[segment.frames + from..segment.frames + to]);
            segment_start += segment.frames;
        }
        anyhow::ensure!(
            left.len() == frames,
            "the segments of {} hold {} of {frames} frames for {start}..{end}",
            manifest.display(),
            left.len()
        );

        let mut planar = left;
        planar.extend_from_slice(&right);
        Ok(planar)
    }

    /// Decode one segment whole and check it is what the manifest promises.
    fn decode(&self, segment: &Segment) -> Result<decode::Decoded> {
        let path = self.dir.join(segment_name(segment.index));
        anyhow::ensure!(
            path.is_file(),
            "{} names a segment that is missing: {}",
            self.manifest().display(),
            path.display()
        );
        let decoded = decode::decode(&path).with_context(|| {
            format!(
                "reading segment {} of {}",
                segment.index,
                self.dir.display()
            )
        })?;
        anyhow::ensure!(
            decoded.sample_rate == SAMPLE_RATE && decoded.channels == 2,
            "{} is {} Hz with {} channels; source segments are 48 kHz stereo",
            path.display(),
            decoded.sample_rate,
            decoded.channels
        );
        anyhow::ensure!(
            decoded.frames == segment.frames,
            "{} holds {} frames, {} says {}",
            path.display(),
            decoded.frames,
            self.manifest().display(),
            segment.frames
        );
        Ok(decoded)
    }

    fn manifest(&self) -> PathBuf {
        self.dir.join(MANIFEST)
    }
}

/// Appends 48 kHz stereo to an entry's segments, one segment in memory at a time, and
/// measures each second of the waveform as it goes.
struct SegmentWriter<'a> {
    dir: &'a Path,
    frames_per_segment: usize,
    left: Vec<f32>,
    right: Vec<f32>,
    segments: Vec<Segment>,
    peaks: Vec<SecondPeaks>,
    second_left_squares: f64,
    second_right_squares: f64,
    second_frames: usize,
}

impl<'a> SegmentWriter<'a> {
    fn new(dir: &'a Path, frames_per_segment: usize) -> Self {
        Self {
            dir,
            frames_per_segment,
            left: Vec::with_capacity(frames_per_segment),
            right: Vec::with_capacity(frames_per_segment),
            segments: Vec::new(),
            peaks: Vec::new(),
            second_left_squares: 0.0,
            second_right_squares: 0.0,
            second_frames: 0,
        }
    }

    /// Append one chunk of 48 kHz stereo, closing the segment and the second as they fill.
    fn push(&mut self, left: &[f32], right: &[f32]) -> Result<()> {
        anyhow::ensure!(
            left.len() == right.len(),
            "{} left frames against {} right frames",
            left.len(),
            right.len()
        );
        let mut offset = 0;
        while offset < left.len() {
            let count = (self.frames_per_segment - self.left.len())
                .min(PEAK_FRAMES - self.second_frames)
                .min(left.len() - offset);
            self.left.extend_from_slice(&left[offset..offset + count]);
            self.right.extend_from_slice(&right[offset..offset + count]);
            for sample in &left[offset..offset + count] {
                self.second_left_squares += f64::from(*sample) * f64::from(*sample);
            }
            for sample in &right[offset..offset + count] {
                self.second_right_squares += f64::from(*sample) * f64::from(*sample);
            }
            self.second_frames += count;
            offset += count;
            if self.second_frames == PEAK_FRAMES {
                self.close_second();
            }
            if self.left.len() == self.frames_per_segment {
                self.write_segment()?;
            }
        }
        Ok(())
    }

    /// Write the last segment and the last second, however short, and hand back the
    /// manifest's segment table and waveform.
    fn finish(mut self) -> Result<(Vec<Segment>, Vec<SecondPeaks>)> {
        if self.second_frames > 0 {
            self.close_second();
        }
        if !self.left.is_empty() {
            self.write_segment()?;
        }
        anyhow::ensure!(
            !self.segments.is_empty(),
            "the source decoded no frames to store"
        );
        Ok((self.segments, self.peaks))
    }

    fn close_second(&mut self) {
        let frames = self.second_frames as f64;
        self.peaks.push(SecondPeaks {
            left_rms: (self.second_left_squares / frames).sqrt() as f32,
            right_rms: (self.second_right_squares / frames).sqrt() as f32,
        });
        self.second_left_squares = 0.0;
        self.second_right_squares = 0.0;
        self.second_frames = 0;
    }

    fn write_segment(&mut self) -> Result<()> {
        let index = self.segments.len();
        let frames = self.left.len();
        let mut planar = Vec::with_capacity(frames * 2);
        planar.extend_from_slice(&self.left);
        planar.extend_from_slice(&self.right);
        let path = self.dir.join(segment_name(index));
        crate::flac::write(&path, SAMPLE_RATE, 2, &planar)
            .with_context(|| format!("writing segment {index} of {}", self.dir.display()))?;
        let bytes = std::fs::metadata(&path)
            .with_context(|| format!("measuring {}", path.display()))?
            .len();
        self.segments.push(Segment {
            index,
            frames,
            bytes,
        });
        self.left.clear();
        self.right.clear();
        Ok(())
    }
}

fn segment_name(index: usize) -> String {
    format!("src-{index:06}.flac")
}

/// The input's container, as the probe hint reads it from the path.
fn format_of(input: &Path) -> String {
    input
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| "unknown".to_string())
}

/// A manifest has to describe the exact partition of its own frame count into its own
/// segments, with the segment length and waveform this build would have written.
fn validate(manifest: &Path, meta: &Meta, frames_per_segment: usize) -> Result<()> {
    anyhow::ensure!(
        meta.native_channels > 0,
        "{} says its source had no channels",
        manifest.display()
    );
    anyhow::ensure!(
        meta.native_sample_rate > 0,
        "{} says its source was 0 Hz",
        manifest.display()
    );
    anyhow::ensure!(meta.frames > 0, "{} promises no audio", manifest.display());
    anyhow::ensure!(
        meta.frames_per_segment == frames_per_segment,
        "{} says {}-frame segments, this build writes {frames_per_segment}",
        manifest.display(),
        meta.frames_per_segment
    );

    let mut remaining = meta.frames;
    for (position, segment) in meta.segments.iter().enumerate() {
        anyhow::ensure!(
            segment.index == position,
            "{} lists segment {} at position {position}",
            manifest.display(),
            segment.index
        );
        let expected = remaining.min(meta.frames_per_segment);
        anyhow::ensure!(
            segment.frames == expected,
            "{} says segment {} holds {} frames, the track needs {expected}",
            manifest.display(),
            segment.index,
            segment.frames
        );
        remaining -= expected;
    }
    anyhow::ensure!(
        remaining == 0,
        "{} promises {} frames but its segments hold {}",
        manifest.display(),
        meta.frames,
        meta.frames - remaining
    );

    let expected_peaks = meta.frames.div_ceil(PEAK_FRAMES);
    anyhow::ensure!(
        meta.peaks.len() == expected_peaks,
        "{} has {} waveform seconds for {} frames",
        manifest.display(),
        meta.peaks.len(),
        meta.frames
    );
    Ok(())
}

/// Every segment the manifest names has to be there; a decode check happens on read.
fn ensure_segments(manifest: &Path, dir: &Path, meta: &Meta) -> Result<()> {
    for segment in &meta.segments {
        let path = dir.join(segment_name(segment.index));
        anyhow::ensure!(
            path.is_file(),
            "{} names a segment that is missing: {}",
            manifest.display(),
            path.display()
        );
    }
    Ok(())
}

fn read_meta(path: &Path) -> Result<Meta> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

/// Write the manifest through a temporary file in the same directory, so a reader never
/// sees half of one.
fn write_manifest(dir: &Path, meta: &Meta) -> Result<()> {
    let path = dir.join(MANIFEST);
    let temp = dir.join(format!("{MANIFEST}.tmp"));
    std::fs::write(&temp, serde_json::to_string_pretty(meta)? + "\n")
        .with_context(|| format!("writing {}", temp.display()))?;
    std::fs::rename(&temp, &path).with_context(|| format!("replacing {}", path.display()))
}

/// Remove an entry or a leftover temporary directory, whichever is in the way.
fn clear(path: &Path) -> Result<()> {
    let removed = if path.is_dir() {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    };
    removed.with_context(|| format!("clearing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One 24-bit step: a sample already on this grid survives a FLAC round trip unchanged.
    const QUANTUM: f32 = 1.0 / 8_388_608.0;

    /// A unique cache root in the temp directory, deleted by the test that made it.
    fn temp_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "flacify-source-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("creating the test directory");
        root
    }

    /// A deterministic signal whose samples all sit on the 24-bit grid.
    fn grid_signal(frames: usize, channels: usize) -> Vec<f32> {
        let mut planar = Vec::with_capacity(frames * channels);
        for channel in 0..channels {
            for frame in 0..frames {
                let step = (frame as i64 * (7_919 + channel as i64 * 3_217)) % 8_192 - 4_096;
                planar.push(step as f32 * QUANTUM);
            }
        }
        planar
    }

    fn wav_source(
        root: &Path,
        name: &str,
        sample_rate: u32,
        channels: usize,
        frames: usize,
    ) -> PathBuf {
        let path = root.join(name);
        crate::wav::write(&path, sample_rate, channels, &grid_signal(frames, channels))
            .expect("writing the test source");
        path
    }

    /// The one-shot decode through the same channel and rate rules the source pass uses.
    fn one_shot(decoded_from: &Path) -> Vec<f32> {
        let decoded = decode::decode(decoded_from).expect("decoding the source");
        decode::to_stereo_48k(&decoded).expect("resampling the source")
    }

    /// Round to the 24-bit grid the way `flac::write` does, so a resampled source can be
    /// compared with what its segments hold.
    fn quantise(planar: &[f32]) -> Vec<f32> {
        planar
            .iter()
            .map(|&sample| (sample / QUANTUM).round() as i32 as f32 * QUANTUM)
            .collect()
    }

    fn bits(samples: &[f32]) -> Vec<u32> {
        samples.iter().map(|sample| sample.to_bits()).collect()
    }

    /// A built entry whose manifest can be edited by a test, plus its source path.
    struct Fixture {
        root: PathBuf,
        tier: Tier,
        source: PathBuf,
        entry: Entry,
    }

    impl Fixture {
        fn new(name: &str, sample_rate: u32, channels: usize, frames: usize) -> Self {
            let root = temp_root(name);
            let tier = Tier::new(&root, 48_000).expect("a tier");
            let source = wav_source(&root, "source.wav", sample_rate, channels, frames);
            let entry = tier
                .build_or_reuse(&source, "hash")
                .expect("building the entry");
            Self {
                root,
                tier,
                source,
                entry,
            }
        }

        fn manifest(&self) -> PathBuf {
            self.entry.dir.join(MANIFEST)
        }

        fn stored_meta(&self) -> Meta {
            read_meta(&self.manifest()).expect("reading the manifest")
        }

        fn store_meta(&self, meta: &Meta) {
            write_manifest(&self.entry.dir, meta).expect("rewriting the manifest");
        }

        fn cleanup(&self) {
            std::fs::remove_dir_all(&self.root).ok();
        }
    }

    #[test]
    fn given_a_44_1_khz_stereo_source_when_read_back_then_every_sample_is_the_one_shot_decode_on_the_24_bit_grid(
    ) {
        let fixture = Fixture::new("44-1-stereo", 44_100, 2, 88_200);
        let joined = fixture.entry.read_all().expect("reading the entry");
        let oracle = one_shot(&fixture.source);

        assert_eq!(
            joined.len(),
            96_000 * 2,
            "88 200 frames at 44.1 kHz become 96 000 at 48 kHz"
        );
        assert_eq!(joined.len(), oracle.len());
        assert_eq!(
            bits(&joined),
            bits(&quantise(&oracle)),
            "a resampled source must read back as its one-shot decode"
        );
        // The only difference from the unquantised decode is the 24-bit rounding the cache
        // format accepts.
        for (index, (&actual, &resampled)) in joined.iter().zip(oracle.iter()).enumerate() {
            assert!(
                (actual - resampled).abs() <= QUANTUM / 2.0,
                "sample {index}: {actual} is more than half a 24-bit step from {resampled}"
            );
        }
        assert_eq!(fixture.entry.meta().native_sample_rate, 44_100);
        assert_eq!(fixture.entry.meta().native_channels, 2);
        assert_eq!(fixture.entry.meta().format, "wav");
        fixture.cleanup();
    }

    #[test]
    fn given_a_48_khz_source_when_read_back_then_every_sample_is_the_one_shot_decode_bit_for_bit() {
        let fixture = Fixture::new("48-stereo", 48_000, 2, 72_000);
        let joined = fixture.entry.read_all().expect("reading the entry");
        assert_eq!(bits(&joined), bits(&one_shot(&fixture.source)));
        fixture.cleanup();
    }

    #[test]
    fn given_a_mono_source_when_read_back_then_every_sample_is_the_one_shot_decode() {
        let fixture = Fixture::new("mono", 44_100, 1, 44_100);
        let joined = fixture.entry.read_all().expect("reading the entry");
        let left = &joined[..fixture.entry.meta().frames];
        let right = &joined[fixture.entry.meta().frames..];
        assert_eq!(bits(left), bits(right), "mono is duplicated to both sides");
        assert_eq!(bits(&joined), bits(&quantise(&one_shot(&fixture.source))));
        assert_eq!(fixture.entry.meta().native_channels, 1);
        fixture.cleanup();
    }

    #[test]
    fn given_a_three_channel_source_when_read_back_then_every_sample_is_the_one_shot_decode() {
        let fixture = Fixture::new("three-channel", 44_100, 3, 44_100);
        let joined = fixture.entry.read_all().expect("reading the entry");
        assert_eq!(bits(&joined), bits(&quantise(&one_shot(&fixture.source))));
        assert_eq!(fixture.entry.meta().native_channels, 3);
        fixture.cleanup();
    }

    #[test]
    fn given_a_source_longer_than_one_segment_when_built_then_the_segments_partition_its_frames() {
        let fixture = Fixture::new("partition", 48_000, 2, 48_000 * 2 + 12_345);
        let meta = fixture.entry.meta();

        assert_eq!(meta.segments.len(), 3);
        assert_eq!(meta.segments[0].frames, 48_000);
        assert_eq!(meta.segments[1].frames, 48_000);
        assert_eq!(meta.segments[2].frames, 12_345, "the last segment is short");
        assert_eq!(
            meta.segments
                .iter()
                .map(|segment| segment.frames)
                .sum::<usize>(),
            meta.frames
        );
        for (position, segment) in meta.segments.iter().enumerate() {
            assert_eq!(segment.index, position);
            let path = fixture.entry.dir.join(segment_name(position));
            assert_eq!(
                std::fs::metadata(&path).expect("measuring a segment").len(),
                segment.bytes
            );
        }
        fixture.cleanup();
    }

    #[test]
    fn given_a_source_shorter_than_one_segment_when_built_then_there_is_exactly_one_segment() {
        let fixture = Fixture::new("short", 48_000, 2, 12_345);
        let meta = fixture.entry.meta();

        assert_eq!(meta.segments.len(), 1);
        assert_eq!(meta.segments[0].frames, 12_345);
        assert!(fixture.entry.dir.join("src-000000.flac").is_file());
        assert!(!fixture.entry.dir.join("src-000001.flac").exists());
        fixture.cleanup();
    }

    #[test]
    fn given_two_and_a_half_seconds_when_built_then_each_stored_peak_is_that_seconds_rms() {
        let fixture = Fixture::new("peaks", 48_000, 2, 48_000 * 2 + 12_000);
        let joined = fixture.entry.read_all().expect("reading the entry");
        let frames = fixture.entry.meta().frames;

        let expected = second_peaks(&joined, frames);
        assert_eq!(expected.len(), 3, "two full seconds and one partial");
        assert_eq!(
            fixture.entry.meta().peaks,
            expected,
            "the stored waveform is the RMS of each second, last one included"
        );
        fixture.cleanup();
    }

    #[test]
    fn given_a_range_across_a_segment_boundary_when_read_then_it_is_that_slice_of_the_whole() {
        let fixture = Fixture::new("range", 48_000, 2, 48_000 * 2 + 12_345);
        let frames = fixture.entry.meta().frames;
        let joined = fixture.entry.read_all().expect("reading the entry");

        let (start, count) = (40_000, 30_000);
        let range = fixture
            .entry
            .read_range(start, count)
            .expect("reading a range");

        assert_eq!(range.len(), count * 2);
        assert_eq!(
            bits(&range[..count]),
            bits(&joined[start..start + count]),
            "left"
        );
        assert_eq!(
            bits(&range[count..]),
            bits(&joined[frames + start..frames + start + count]),
            "right"
        );
        fixture.cleanup();
    }

    #[test]
    fn given_a_segment_that_does_not_decode_when_a_range_elsewhere_is_read_then_it_still_reads() {
        let fixture = Fixture::new("lazy", 48_000, 2, 48_000 * 2 + 12_345);
        let joined = fixture.entry.read_all().expect("reading the whole entry");
        std::fs::write(fixture.entry.dir.join("src-000001.flac"), b"not a flac")
            .expect("damaging a segment");

        let reopened = fixture
            .tier
            .build_or_reuse(&fixture.source, "hash")
            .expect("the entry's files are all still there");
        let range = reopened
            .read_range(0, 1_000)
            .expect("reading the first segment");
        assert_eq!(bits(&range[..1_000]), bits(&joined[..1_000]));
        assert!(
            reopened.read_all().is_err(),
            "the damaged segment is still an error when it is read"
        );
        fixture.cleanup();
    }

    #[test]
    fn given_a_range_past_the_last_frame_when_read_then_it_is_an_error() {
        let fixture = Fixture::new("past-end", 48_000, 2, 12_345);
        let error = fixture
            .entry
            .read_range(12_000, 1_000)
            .expect_err("the entry is shorter than the range");
        assert!(format!("{error:#}").contains("12345 frames"), "{error:#}");
        fixture.cleanup();
    }

    #[test]
    fn given_a_range_of_no_frames_when_read_then_it_is_empty() {
        let fixture = Fixture::new("empty-range", 48_000, 2, 12_345);
        assert!(fixture
            .entry
            .read_range(12_345, 0)
            .expect("an empty range at the end")
            .is_empty());
        fixture.cleanup();
    }

    #[test]
    fn given_a_manifest_that_names_a_missing_segment_when_opened_then_it_is_an_error() {
        let fixture = Fixture::new("missing-segment", 48_000, 2, 48_000 * 2 + 12_345);
        std::fs::remove_file(fixture.entry.dir.join("src-000001.flac"))
            .expect("removing a segment");

        let error = fixture
            .tier
            .build_or_reuse(&fixture.source, "hash")
            .expect_err("a missing segment is an error");
        let text = format!("{error:#}");
        assert!(
            text.contains(&fixture.manifest().display().to_string()),
            "{text}"
        );
        assert!(text.contains("src-000001.flac"), "{text}");
        fixture.cleanup();
    }

    #[test]
    fn given_a_manifest_whose_segments_do_not_add_up_when_opened_then_it_is_an_error() {
        let fixture = Fixture::new("short-manifest", 48_000, 2, 48_000 * 2 + 12_345);
        let mut meta = fixture.stored_meta();
        meta.segments.pop();
        fixture.store_meta(&meta);

        let error = fixture
            .tier
            .build_or_reuse(&fixture.source, "hash")
            .expect_err("a manifest without its last segment is an error");
        assert!(
            format!("{error:#}").contains("promises 108345 frames"),
            "{error:#}"
        );
        fixture.cleanup();
    }

    #[test]
    fn given_a_manifest_with_a_gap_in_its_indexes_when_opened_then_it_is_an_error() {
        let fixture = Fixture::new("index-gap", 48_000, 2, 48_000 * 2 + 12_345);
        let mut meta = fixture.stored_meta();
        meta.segments[1].index = 2;
        fixture.store_meta(&meta);

        let error = fixture
            .tier
            .build_or_reuse(&fixture.source, "hash")
            .expect_err("a gap is an error");
        assert!(format!("{error:#}").contains("position 1"), "{error:#}");
        fixture.cleanup();
    }

    #[test]
    fn given_a_manifest_whose_segment_length_disagrees_with_this_build_when_opened_then_it_is_an_error(
    ) {
        let fixture = Fixture::new("segment-length", 48_000, 2, 12_345);
        let mut meta = fixture.stored_meta();
        meta.frames_per_segment += 1;
        fixture.store_meta(&meta);

        let error = fixture
            .tier
            .build_or_reuse(&fixture.source, "hash")
            .expect_err("a segment length for other settings is an error");
        assert!(
            format!("{error:#}").contains("this build writes 48000"),
            "{error:#}"
        );
        fixture.cleanup();
    }

    #[test]
    fn given_a_manifest_with_the_wrong_number_of_waveform_seconds_when_opened_then_it_is_an_error()
    {
        let fixture = Fixture::new("peaks-count", 48_000, 2, 48_000 + 12_000);
        let mut meta = fixture.stored_meta();
        meta.peaks.pop();
        fixture.store_meta(&meta);

        let error = fixture
            .tier
            .build_or_reuse(&fixture.source, "hash")
            .expect_err("a waveform of the wrong length is an error");
        assert!(
            format!("{error:#}").contains("1 waveform seconds"),
            "{error:#}"
        );
        fixture.cleanup();
    }

    #[test]
    fn given_a_complete_entry_when_opened_again_then_it_is_reused_and_the_source_never_reopened() {
        let fixture = Fixture::new("reuse", 48_000, 2, 12_345);
        let mut meta = fixture.stored_meta();
        meta.created_at = 1;
        fixture.store_meta(&meta);
        let segment = fixture.entry.dir.join("src-000000.flac");
        let written = std::fs::metadata(&segment)
            .expect("measuring the segment")
            .modified()
            .expect("a modification time");
        std::fs::remove_file(&fixture.source).expect("removing the source");

        let reused = fixture
            .tier
            .build_or_reuse(&fixture.source, "hash")
            .expect("a complete entry needs no source and no second pass");

        assert_eq!(
            reused.meta().created_at,
            1,
            "a reused entry must not rewrite its manifest"
        );
        assert_eq!(
            std::fs::metadata(&segment)
                .expect("measuring the segment")
                .modified()
                .expect("a modification time"),
            written,
            "a reused entry must not rewrite its segments"
        );
        fixture.cleanup();
    }

    #[test]
    fn given_an_entry_from_another_version_when_opened_then_it_is_rebuilt() {
        let fixture = Fixture::new("version", 48_000, 2, 12_345);
        let mut meta = fixture.stored_meta();
        meta.version = "flacify-source-0".into();
        fixture.store_meta(&meta);

        let rebuilt = fixture
            .tier
            .build_or_reuse(&fixture.source, "hash")
            .expect("another version is a rebuild, not an error");

        assert_eq!(rebuilt.meta().version, VERSION);
        assert!(rebuilt.meta().created_at > 0);
        fixture.cleanup();
    }

    #[test]
    fn given_a_leftover_temporary_directory_when_built_then_the_entry_replaces_it() {
        let root = temp_root("stale-temp");
        let tier = Tier::new(&root, 48_000).expect("a tier");
        let source = wav_source(&root, "source.wav", 48_000, 2, 12_345);
        let temp = tier.entry_dir("hash").with_file_name("hash.tmp");
        std::fs::create_dir_all(&temp).expect("creating a stale temporary directory");
        std::fs::write(temp.join("src-000000.flac"), b"half a segment").expect("writing junk");

        let entry = tier.build_or_reuse(&source, "hash").expect("building");
        assert_eq!(entry.meta().segments.len(), 1);
        assert!(
            !temp.exists(),
            "the temporary directory is consumed by the rename"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_tier_with_a_zero_frame_segment_when_built_then_it_is_an_error() {
        let root = temp_root("zero-segment");
        let error = Tier::new(&root, 0).expect_err("a segment needs a frame");
        assert!(
            format!("{error:#}").contains("at least one frame"),
            "{error:#}"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_no_frames_when_the_segment_writer_finishes_then_it_is_an_error() {
        let root = temp_root("no-frames");
        let error = SegmentWriter::new(&root, 48_000)
            .finish()
            .expect_err("an entry with no audio is an error");
        assert!(
            format!("{error:#}").contains("no frames to store"),
            "{error:#}"
        );
        std::fs::remove_dir_all(root).ok();
    }
}
