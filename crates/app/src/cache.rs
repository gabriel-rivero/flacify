//! Render once, play forever: the segmented FLAC cache.
//!
//! Rendering a three-minute track costs about 95 s at RTF 0.54, and without a cache
//! every play pays it again. The cache is keyed by everything that can change the
//! output: the source file's bytes, the window and overlap, the pipeline version and the
//! model weights. A hit is exactly the audio a fresh render would have produced, and a
//! change to any of those misses instead of serving the wrong file.
//!
//! One entry is one track: `manifest.json` plus 24-bit FLAC segments of
//! `7 * (window - overlap)` frames each. The manifest is the source of truth for what
//! the entry holds; a segment it does not list is not part of the track. This module
//! reads and writes whole entries today; the segment layout is the one the incremental
//! renderer will fill.
//!
//! A finished entry is written to a temporary directory and renamed, so a half-written
//! render can never be read back; the cost is that an interrupted render starts over
//! rather than resuming. A corrupt entry is an error, not a miss: silence about one is
//! how you end up debugging audio that plays the wrong thing.
//!
//! Seeking needs nothing from the cache beyond what playback already does: the entry is
//! decoded into memory and the playhead is an index.

use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::render::Render;

/// Bump this when anything about the pipeline changes the samples: window, overlap,
/// preprocessing, the seed schedule, the limiter, or the layout a render is stored in.
///
/// 6 (2026-11-16): v5 stored each render window's planar chunk back to back, so the file
/// was block-major while the reader treated it as channel-major. A cache hit played the
/// right channel from the middle of the track. Every v5 entry has to be re-rendered.
///
/// 7 (2026-11-17): an entry is a manifest plus FLAC segments. Every v6 entry misses; its
/// directory is left for eviction rather than deleted here.
///
/// 8 (2026-11-17): the manifest's identity field is the source file's hash, not the
/// decoded samples' hash. Every v7 entry misses; its directory is left for eviction.
pub const CACHE_VERSION: &str = "flacify-cache-8";

/// Steps per segment. A segment has to start on a multiple of `window - overlap`, so its
/// length is derived from the settings and is not a wall-clock 30 s.
const STEPS_PER_SEGMENT: usize = 7;

/// The one file in an entry that is not audio.
const MANIFEST: &str = "manifest.json";

/// One FLAC file of an entry, as the manifest describes it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Segment {
    /// Position in the track; also the number in the file name.
    pub index: usize,
    pub frames: usize,
    pub bytes: u64,
}

/// The manifest of one entry: where the audio came from, what the limiter did to it, and
/// which segments hold it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Meta {
    pub version: String,
    pub frames: usize,
    pub sample_rate: u32,
    #[serde(default)]
    pub frames_per_segment: usize,
    pub cutoff_hz: u32,
    /// What produced this entry: the source file's hash and the render settings. Stored
    /// so `find` can match them when the model is not installed and its id cannot be
    /// computed. Empty on entries written before these fields were recorded; `annotate`
    /// fills them on the next ready play.
    #[serde(default)]
    pub source_hash: String,
    #[serde(default)]
    pub window: usize,
    #[serde(default)]
    pub overlap: usize,
    #[serde(default)]
    pub splice: bool,
    /// Peak before and after the limiter ran, so the UI can still say what happened.
    pub peak_before: f32,
    pub peak_after: f32,
    pub limited_samples: usize,
    /// Loudness gain applied to the generated side, in dB. Recorded so a cache hit can
    /// still report it.
    #[serde(default)]
    pub gain_db: f64,
    /// The device the render ran on. Informational: a cache hit on another device is
    /// accepted, which is a deliberate prototype compromise (fp32 devices agree to
    /// within the gate, but they are not bit-identical).
    pub device: String,
    /// Every segment of the track, in order from the first frame.
    #[serde(default)]
    pub segments: Vec<Segment>,
    /// Unix seconds, for eviction: `store` sets both, `load` moves `last_used`.
    #[serde(default)]
    pub last_used: u64,
    #[serde(default)]
    pub created_at: u64,
}

/// What identifies an entry, in a form that can also be written into `manifest.json`.
///
/// The directory name is their hash, but a lookup on a machine without the model cannot
/// recompute that hash: the model id comes from files that are gone. The stored fields let
/// [`Cache::find`] match the source file and the settings instead.
#[derive(Clone, Debug)]
pub struct Fields {
    /// blake3 of the source file's bytes, first 32 hex characters.
    pub source_hash: String,
    pub window: usize,
    pub overlap: usize,
    pub model_id: String,
    pub splice: bool,
}

impl Fields {
    pub fn key(&self) -> String {
        key(
            &self.source_hash,
            self.window,
            self.overlap,
            &self.model_id,
            self.splice,
        )
    }
}

#[derive(Debug)]
pub struct Cached {
    pub planar: Vec<f32>,
    pub meta: Meta,
}

pub struct Cache {
    root: PathBuf,
}

impl Cache {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn entry_dir(&self, key: &str) -> PathBuf {
        self.root.join(key)
    }

    /// `None` when there is no entry, or when it was written by another version; an error
    /// when there is one and it cannot be read, because silence about a corrupt cache
    /// entry is how you end up debugging audio.
    pub fn load(&self, key: &str) -> Result<Option<Cached>> {
        self.load_dir(&self.entry_dir(key))
    }

    /// Find an entry by the source file and settings it was made from, ignoring which model
    /// rendered it. Used when the model is not installed: the id cannot be computed, and
    /// an existing render is better than a setup screen. Only entries that recorded their
    /// fields can match; `annotate` fills them on a ready play.
    pub fn find(&self, fields: &Fields) -> Result<Option<Cached>> {
        let Ok(entries) = std::fs::read_dir(&self.root) else {
            return Ok(None);
        };
        for entry in entries.flatten() {
            if entry.file_name().to_string_lossy().ends_with(".tmp") {
                continue;
            }
            if !entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false) {
                continue;
            }
            let dir = entry.path();
            let Ok(Some(meta)) = self.read_meta(&dir) else {
                continue;
            };
            if meta.version != CACHE_VERSION
                || meta.source_hash != fields.source_hash
                || meta.window != fields.window
                || meta.overlap != fields.overlap
                || meta.splice != fields.splice
                || !dir.join(MANIFEST).is_file()
            {
                continue;
            }
            return self.load_dir(&dir);
        }
        Ok(None)
    }

    /// Fill in the identity fields on an entry written before they were recorded. Best
    /// effort: a cache hit that cannot be annotated still plays.
    pub fn annotate(&self, key: &str, fields: &Fields) -> Result<()> {
        let dir = self.entry_dir(key);
        let Some(mut meta) = self.read_meta(&dir)? else {
            return Ok(());
        };
        if !meta.source_hash.is_empty() {
            return Ok(());
        }
        meta.source_hash = fields.source_hash.clone();
        meta.window = fields.window;
        meta.overlap = fields.overlap;
        meta.splice = fields.splice;
        write_manifest(&dir, &meta)
    }

    fn load_dir(&self, dir: &Path) -> Result<Option<Cached>> {
        let Some(mut meta) = self.read_meta(dir)? else {
            return Ok(None);
        };
        if meta.version != CACHE_VERSION {
            return Ok(None);
        }
        let manifest = dir.join(MANIFEST);
        validate_manifest(&manifest, &meta)?;

        // Each segment is channel-major for its own frame range; the entry is channel-major
        // for the whole track, so the planes are gathered separately and joined once.
        let mut left = Vec::with_capacity(meta.frames);
        let mut right = Vec::with_capacity(meta.frames);
        for segment in &meta.segments {
            let path = dir.join(segment_name(segment.index));
            anyhow::ensure!(
                path.is_file(),
                "{} names a segment that is missing: {}",
                manifest.display(),
                path.display()
            );
            let decoded = crate::decode::decode(&path).with_context(|| {
                format!("reading segment {} of {}", segment.index, dir.display())
            })?;
            anyhow::ensure!(
                decoded.sample_rate == flacify_core::SAMPLE_RATE && decoded.channels == 2,
                "{} is {} Hz with {} channels; cache segments are 48 kHz stereo",
                path.display(),
                decoded.sample_rate,
                decoded.channels
            );
            anyhow::ensure!(
                decoded.frames == segment.frames,
                "{} holds {} frames, {} says {}",
                path.display(),
                decoded.frames,
                manifest.display(),
                segment.frames
            );
            left.extend_from_slice(&decoded.planar[..segment.frames]);
            right.extend_from_slice(&decoded.planar[segment.frames..]);
        }
        let mut planar = left;
        planar.extend_from_slice(&right);

        meta.last_used = now();
        if let Err(error) = write_manifest(dir, &meta) {
            // A playable entry matters more than its timestamp: eviction will read the
            // stale time as an older entry, not as a corrupt one.
            tracing::warn!(
                %error,
                entry = %dir.display(),
                "could not record when the cache entry was last used"
            );
        }
        Ok(Some(Cached { planar, meta }))
    }

    fn read_meta(&self, dir: &Path) -> Result<Option<Meta>> {
        let path = dir.join(MANIFEST);
        if !path.is_file() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let meta =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        Ok(Some(meta))
    }

    /// Write a finished render as one segment per `7 * (window - overlap)` frames.
    /// Everything lands in `<key>.tmp` first and is renamed, so a reader never sees a
    /// partial entry.
    pub fn store(
        &self,
        key: &str,
        render: &Render,
        cutoff_hz: u32,
        fields: &Fields,
    ) -> Result<PathBuf> {
        anyhow::ensure!(
            render.planar.len() % 2 == 0,
            "a render of {} samples is not whole stereo frames",
            render.planar.len()
        );
        let frames = render.planar.len() / 2;
        anyhow::ensure!(frames > 0, "a cache entry needs at least one frame");
        let segment_frames = frames_per_segment(fields.window, fields.overlap)?;

        let final_dir = self.entry_dir(key);
        let temp_dir = self.root.join(format!("{key}.tmp"));
        if temp_dir.exists() {
            std::fs::remove_dir_all(&temp_dir)
                .with_context(|| format!("clearing {}", temp_dir.display()))?;
        }
        std::fs::create_dir_all(&temp_dir)
            .with_context(|| format!("creating {}", temp_dir.display()))?;

        let mut segments = Vec::new();
        for (index, start) in (0..frames).step_by(segment_frames).enumerate() {
            let count = segment_frames.min(frames - start);
            let path = temp_dir.join(segment_name(index));
            let slice = stereo_slice(&render.planar, frames, start, count);
            crate::flac::write(&path, flacify_core::SAMPLE_RATE, 2, &slice)?;
            let bytes = std::fs::metadata(&path)
                .with_context(|| format!("measuring {}", path.display()))?
                .len();
            segments.push(Segment {
                index,
                frames: count,
                bytes,
            });
        }

        let created_at = now();
        let meta = Meta {
            version: CACHE_VERSION.to_string(),
            frames,
            sample_rate: flacify_core::SAMPLE_RATE,
            frames_per_segment: segment_frames,
            cutoff_hz,
            source_hash: fields.source_hash.clone(),
            window: fields.window,
            overlap: fields.overlap,
            splice: fields.splice,
            peak_before: render.limit.peak_before,
            peak_after: render.limit.peak_after,
            limited_samples: render.limit.limited_samples,
            gain_db: render.gain_db,
            device: render.device.clone(),
            segments,
            last_used: created_at,
            created_at,
        };
        write_manifest(&temp_dir, &meta)?;

        if final_dir.exists() {
            std::fs::remove_dir_all(&final_dir)
                .with_context(|| format!("replacing {}", final_dir.display()))?;
        }
        std::fs::rename(&temp_dir, &final_dir)
            .with_context(|| format!("publishing {}", final_dir.display()))?;
        Ok(final_dir)
    }
}

/// Frames per segment, derived from the settings so that every segment starts on a
/// multiple of the render step.
fn frames_per_segment(window: usize, overlap: usize) -> Result<usize> {
    let step = window
        .checked_sub(overlap)
        .filter(|step| *step > 0)
        .with_context(|| format!("window {window} is not longer than overlap {overlap}"))?;
    step.checked_mul(STEPS_PER_SEGMENT)
        .with_context(|| format!("{step} frames per step overflows a segment length"))
}

/// A manifest has to describe the exact partition of its own frame count into its own
/// segments. Anything else is a corrupt entry, and a corrupt entry is an error.
fn validate_manifest(manifest: &Path, meta: &Meta) -> Result<()> {
    let expected_frames = frames_per_segment(meta.window, meta.overlap)
        .with_context(|| format!("{} has no usable segment length", manifest.display()))?;
    anyhow::ensure!(
        meta.frames_per_segment == expected_frames,
        "{} says {}-frame segments, its window {} and overlap {} give {expected_frames}",
        manifest.display(),
        meta.frames_per_segment,
        meta.window,
        meta.overlap
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
    Ok(())
}

/// Cut one channel-major slice out of a whole-track buffer.
fn stereo_slice(planar: &[f32], total_frames: usize, start: usize, frames: usize) -> Vec<f32> {
    let mut slice = Vec::with_capacity(frames * 2);
    slice.extend_from_slice(&planar[start..start + frames]);
    let right = total_frames + start;
    slice.extend_from_slice(&planar[right..right + frames]);
    slice
}

fn segment_name(index: usize) -> String {
    format!("seg-{index:06}.flac")
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

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

/// One read while hashing a source file, so the file is never held in memory whole.
const HASH_CHUNK: usize = 1 << 20;

/// blake3 of a file's bytes, first 32 hex characters: the identity part of a cache key.
///
/// Keyed by the container, so the same audio re-encoded is a miss and a retag is a miss;
/// keying on the decoded samples instead costs a full decode before every lookup.
pub fn source_hash(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("opening {} to hash it", path.display()))?;
    let mut hasher = blake3::Hasher::new();
    let mut chunk = vec![0u8; HASH_CHUNK];
    loop {
        let read = file
            .read(&mut chunk)
            .with_context(|| format!("hashing {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&chunk[..read]);
    }
    Ok(hasher.finalize().to_hex()[..32].to_string())
}

/// Everything that can change the rendered samples, hashed into a directory name.
pub fn key(
    source_hash: &str,
    window: usize,
    overlap: usize,
    model_id: &str,
    splice_low_band: bool,
) -> String {
    let mut hasher = blake3::Hasher::new();
    for part in [
        CACHE_VERSION,
        source_hash,
        model_id,
        &window.to_string(),
        &overlap.to_string(),
        if splice_low_band { "splice" } else { "whole" },
    ] {
        hasher.update(part.as_bytes());
        hasher.update(b"\0");
    }
    hasher.finalize().to_hex()[..32].to_string()
}

/// Identifies the weights without reading 3.3 GB: their sizes and modification times.
/// A same-size, same-mtime revision would slip through, which is why `CACHE_VERSION`
/// exists and why that is written down rather than assumed.
pub fn model_id(models_dir: &Path) -> String {
    let mut hasher = blake3::Hasher::new();
    for name in ["student_ldm.pth", "sr_vocoder.pth", "vae.pth"] {
        let path = models_dir.join(name);
        hasher.update(name.as_bytes());
        match std::fs::metadata(&path) {
            Ok(meta) => {
                hasher.update(&meta.len().to_le_bytes());
                if let Ok(modified) = meta.modified() {
                    if let Ok(since) = modified.duration_since(std::time::UNIX_EPOCH) {
                        hasher.update(&since.as_secs().to_le_bytes());
                    }
                }
            }
            Err(_) => {
                hasher.update(b"missing");
            }
        }
    }
    hasher.finalize().to_hex()[..16].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limit::LimitReport;

    /// One 24-bit step: `flac` rounds to this, and a sample already on the grid survives
    /// the round trip unchanged.
    const QUANTUM: f32 = 1.0 / 8_388_608.0;

    fn temp_cache(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "flacify-cache-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        dir
    }

    /// A render whose samples are all exact 24-bit values, so a round trip either returns
    /// them unchanged or shows the bug.
    fn render_of(frames: usize) -> Render {
        let mut planar = Vec::with_capacity(frames * 2);
        for channel in 0..2 {
            for frame in 0..frames {
                let step = (frame as i32 * (7_919 + channel * 3_217)) % 4_096 - 2_048;
                planar.push(step as f32 * QUANTUM);
            }
        }
        Render {
            planar,
            limit: LimitReport {
                peak_before: 0.5,
                peak_after: 0.5,
                limited_samples: 0,
            },
            device: "cpu".to_string(),
            gain_db: 0.0,
            lufs_input: -14.0,
            lufs_output: -15.0,
        }
    }

    /// The shipped settings: a 245 760-frame window with a 48 000-frame overlap.
    fn fields() -> Fields {
        Fields {
            source_hash: "source".into(),
            window: 245_760,
            overlap: 48_000,
            model_id: "model".into(),
            splice: false,
        }
    }

    /// The same entry with 16 800-frame segments, so a test can span several of them
    /// without encoding a minute of audio.
    fn fields_with_short_segments() -> Fields {
        Fields {
            window: 4_800,
            overlap: 2_400,
            ..fields()
        }
    }

    fn manifest_path(cache: &Cache, key: &str) -> PathBuf {
        cache.entry_dir(key).join(MANIFEST)
    }

    fn stored_meta(cache: &Cache, key: &str) -> Meta {
        let path = manifest_path(cache, key);
        serde_json::from_str(&std::fs::read_to_string(&path).expect("reading the manifest"))
            .expect("parsing the manifest")
    }

    fn store_meta(cache: &Cache, key: &str, meta: &Meta) {
        std::fs::write(
            manifest_path(cache, key),
            serde_json::to_string_pretty(meta).expect("serialising the manifest"),
        )
        .expect("writing the manifest");
    }

    #[test]
    fn given_a_render_that_does_not_fill_its_last_segment_when_stored_and_loaded_then_the_samples_and_frame_count_match(
    ) {
        let root = temp_cache("roundtrip");
        let cache = Cache::new(root.clone());
        let fields = fields_with_short_segments();
        let frames = 16_800 * 2 + 5_000;
        let render = render_of(frames);
        cache
            .store("key", &render, 15_328, &fields)
            .expect("storing");

        let cached = cache.load("key").expect("loading").expect("an entry");
        assert_eq!(cached.meta.frames, frames);
        assert_eq!(cached.meta.cutoff_hz, 15_328);
        assert_eq!(cached.meta.segments.len(), 3);
        assert_eq!(cached.planar.len(), render.planar.len());
        for (index, (&expected, &actual)) in render.planar.iter().zip(&cached.planar).enumerate() {
            assert_eq!(actual, expected, "sample {index}: {actual} != {expected}");
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_stored_entry_when_read_back_then_each_manifest_segment_probes_and_decodes_and_the_frames_add_up(
    ) {
        let root = temp_cache("segments");
        let cache = Cache::new(root.clone());
        let fields = fields_with_short_segments();
        cache
            .store("key", &render_of(16_800 * 2 + 500), 12_000, &fields)
            .expect("storing");

        let dir = cache.entry_dir("key");
        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .expect("listing the entry")
            .map(|entry| {
                entry
                    .expect("an entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                "manifest.json",
                "seg-000000.flac",
                "seg-000001.flac",
                "seg-000002.flac"
            ]
        );

        let meta = stored_meta(&cache, "key");
        assert_eq!(meta.frames_per_segment, 16_800);
        assert_eq!(meta.segments.len(), 3);
        let mut decoded_frames = 0;
        for segment in &meta.segments {
            let path = dir.join(segment_name(segment.index));
            let decoded = crate::decode::decode(&path).expect("decoding a segment");
            assert_eq!(decoded.sample_rate, 48_000);
            assert_eq!(decoded.channels, 2);
            assert_eq!(decoded.frames, segment.frames);
            assert_eq!(
                std::fs::metadata(&path).expect("measuring a segment").len(),
                segment.bytes
            );
            decoded_frames += decoded.frames;
        }
        assert_eq!(decoded_frames, meta.frames);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_track_shorter_than_one_segment_when_stored_then_the_entry_holds_exactly_one_segment()
    {
        let root = temp_cache("one-segment");
        let cache = Cache::new(root.clone());
        cache
            .store("key", &render_of(2_400), 12_000, &fields())
            .expect("storing");

        let meta = stored_meta(&cache, "key");
        assert_eq!(meta.segments.len(), 1);
        assert_eq!(meta.segments[0].frames, 2_400);
        let dir = cache.entry_dir("key");
        assert!(dir.join("seg-000000.flac").is_file());
        assert!(!dir.join("seg-000001.flac").exists());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_an_entry_whose_last_used_is_old_when_loaded_then_the_manifest_records_the_new_time() {
        let root = temp_cache("last-used");
        let cache = Cache::new(root.clone());
        cache
            .store("key", &render_of(240), 12_000, &fields())
            .expect("storing");
        let mut meta = stored_meta(&cache, "key");
        assert!(meta.created_at > 0, "created_at is set on store");
        meta.last_used = 1;
        store_meta(&cache, "key", &meta);

        let cached = cache.load("key").expect("loading").expect("an entry");
        assert!(cached.meta.last_used > 1);
        assert_eq!(stored_meta(&cache, "key").last_used, cached.meta.last_used);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_no_entry_when_loaded_then_it_is_none_rather_than_an_error() {
        let root = temp_cache("missing");
        let cache = Cache::new(root.clone());
        assert!(cache.load("nothing-here").expect("no error").is_none());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_version_mismatch_when_loaded_then_it_is_a_miss() {
        let root = temp_cache("version");
        let cache = Cache::new(root.clone());
        cache
            .store("key", &render_of(240), 12_000, &fields())
            .expect("storing");
        let mut meta = stored_meta(&cache, "key");
        meta.version = "something-older".into();
        store_meta(&cache, "key", &meta);

        assert!(cache.load("key").expect("no error").is_none());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_manifest_that_names_a_missing_segment_when_loaded_then_it_is_an_error_naming_the_entry(
    ) {
        let root = temp_cache("missing-segment");
        let cache = Cache::new(root.clone());
        let fields = fields_with_short_segments();
        cache
            .store("key", &render_of(17_800), 12_000, &fields)
            .expect("storing");
        std::fs::remove_file(cache.entry_dir("key").join("seg-000001.flac"))
            .expect("removing a segment");

        let error = cache
            .load("key")
            .expect_err("a missing segment is an error");
        let text = format!("{error:#}");
        assert!(
            text.contains(&manifest_path(&cache, "key").display().to_string()),
            "{text}"
        );
        assert!(text.contains("seg-000001.flac"), "{text}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_manifest_with_a_gap_in_its_segment_indexes_when_loaded_then_it_is_an_error() {
        let root = temp_cache("index-gap");
        let cache = Cache::new(root.clone());
        let fields = fields_with_short_segments();
        cache
            .store("key", &render_of(17_800), 12_000, &fields)
            .expect("storing");
        let mut meta = stored_meta(&cache, "key");
        meta.segments[1].index = 2;
        store_meta(&cache, "key", &meta);

        let error = cache.load("key").expect_err("a gap is an error");
        assert!(format!("{error:#}").contains("position 1"), "{error:#}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_manifest_whose_segments_are_out_of_order_when_loaded_then_it_is_an_error() {
        let root = temp_cache("out-of-order");
        let cache = Cache::new(root.clone());
        let fields = fields_with_short_segments();
        cache
            .store("key", &render_of(17_800), 12_000, &fields)
            .expect("storing");
        let mut meta = stored_meta(&cache, "key");
        meta.segments.swap(0, 1);
        store_meta(&cache, "key", &meta);

        let error = cache.load("key").expect_err("out of order is an error");
        assert!(format!("{error:#}").contains("position 0"), "{error:#}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_manifest_whose_segment_length_disagrees_with_its_window_when_loaded_then_it_is_an_error(
    ) {
        let root = temp_cache("segment-length");
        let cache = Cache::new(root.clone());
        cache
            .store("key", &render_of(240), 12_000, &fields())
            .expect("storing");
        let mut meta = stored_meta(&cache, "key");
        meta.frames_per_segment += 1;
        store_meta(&cache, "key", &meta);

        let error = cache
            .load("key")
            .expect_err("a wrong segment length is an error");
        assert!(format!("{error:#}").contains("window 245760"), "{error:#}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_manifest_with_no_usable_segment_length_when_loaded_then_it_is_an_error() {
        let root = temp_cache("no-window");
        let cache = Cache::new(root.clone());
        cache
            .store("key", &render_of(240), 12_000, &fields())
            .expect("storing");
        let mut meta = stored_meta(&cache, "key");
        meta.window = 0;
        meta.overlap = 0;
        meta.frames_per_segment = 0;
        store_meta(&cache, "key", &meta);

        let error = cache.load("key").expect_err("no step is an error");
        assert!(
            format!("{error:#}").contains("no usable segment length"),
            "{error:#}"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_manifest_that_stops_before_the_last_frames_when_loaded_then_it_is_an_error() {
        let root = temp_cache("short-manifest");
        let cache = Cache::new(root.clone());
        let fields = fields_with_short_segments();
        cache
            .store("key", &render_of(17_800), 12_000, &fields)
            .expect("storing");
        let mut meta = stored_meta(&cache, "key");
        meta.segments.pop();
        store_meta(&cache, "key", &meta);

        let error = cache
            .load("key")
            .expect_err("a manifest without its last segment is an error");
        assert!(
            format!("{error:#}").contains("promises 17800 frames"),
            "{error:#}"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_manifest_that_understates_a_segment_when_loaded_then_it_is_an_error() {
        let root = temp_cache("segment-frames");
        let cache = Cache::new(root.clone());
        cache
            .store("key", &render_of(240), 12_000, &fields())
            .expect("storing");
        let mut meta = stored_meta(&cache, "key");
        meta.segments[0].frames = 200;
        store_meta(&cache, "key", &meta);

        let error = cache.load("key").expect_err("a short segment is an error");
        assert!(format!("{error:#}").contains("needs 240"), "{error:#}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_segment_that_decodes_to_another_length_when_loaded_then_it_is_an_error() {
        let root = temp_cache("segment-truncated");
        let cache = Cache::new(root.clone());
        cache
            .store("key", &render_of(240), 12_000, &fields())
            .expect("storing");
        let path = cache.entry_dir("key").join("seg-000000.flac");
        crate::flac::write(&path, 48_000, 2, &render_of(200).planar).expect("rewriting a segment");

        let error = cache.load("key").expect_err("a short decode is an error");
        assert!(
            format!("{error:#}").contains("holds 200 frames"),
            "{error:#}"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_segment_at_the_wrong_rate_when_loaded_then_it_is_an_error() {
        let root = temp_cache("segment-rate");
        let cache = Cache::new(root.clone());
        cache
            .store("key", &render_of(240), 12_000, &fields())
            .expect("storing");
        let path = cache.entry_dir("key").join("seg-000000.flac");
        crate::flac::write(&path, 44_100, 2, &render_of(240).planar).expect("rewriting a segment");

        let error = cache
            .load("key")
            .expect_err("a 44.1 kHz segment is an error");
        assert!(format!("{error:#}").contains("44100 Hz"), "{error:#}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_render_with_no_frames_when_stored_then_it_is_an_error() {
        let root = temp_cache("empty");
        let cache = Cache::new(root.clone());
        let error = cache
            .store("key", &render_of(0), 12_000, &fields())
            .expect_err("an entry needs a frame");
        assert!(
            format!("{error:#}").contains("at least one frame"),
            "{error:#}"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_buffer_that_is_not_whole_stereo_frames_when_stored_then_it_is_an_error() {
        let root = temp_cache("odd");
        let cache = Cache::new(root.clone());
        let mut render = render_of(2);
        render.planar.pop();
        let error = cache
            .store("key", &render, 12_000, &fields())
            .expect_err("3 samples are not whole frames");
        assert!(
            format!("{error:#}").contains("not whole stereo frames"),
            "{error:#}"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_an_entry_whose_manifest_cannot_be_rewritten_when_loaded_then_it_still_plays() {
        let root = temp_cache("read-only");
        let cache = Cache::new(root.clone());
        cache
            .store("key", &render_of(240), 12_000, &fields())
            .expect("storing");
        let dir = cache.entry_dir("key");
        let original = std::fs::metadata(&dir).expect("stat").permissions();
        let mut read_only_permissions = original.clone();
        read_only_permissions.set_readonly(true);
        std::fs::set_permissions(&dir, read_only_permissions).expect("making the entry read-only");

        let loaded = cache.load("key").expect("a read-only entry still loads");

        std::fs::set_permissions(&dir, original).expect("restoring the permissions");
        assert!(loaded.is_some());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_two_models_when_found_by_source_then_the_settings_match_and_other_source_does_not() {
        let root = temp_cache("find");
        let cache = Cache::new(root.clone());
        let render = render_of(4_800);
        let a = Fields {
            model_id: "model-a".into(),
            ..fields()
        };
        let b = Fields {
            model_id: "model-b".into(),
            ..fields()
        };
        cache
            .store(&a.key(), &render, 15_328, &a)
            .expect("storing a");
        cache
            .store(&b.key(), &render, 15_328, &b)
            .expect("storing b");

        // The model id is ignored: on a machine without the model it cannot be computed,
        // and either entry is the same source file and settings.
        let found = cache.find(&fields()).expect("searching").expect("an entry");
        assert_eq!(found.meta.frames, 4_800);
        assert_eq!(found.meta.source_hash, "source");

        let other = Fields {
            source_hash: "different".into(),
            ..fields()
        };
        assert!(cache.find(&other).expect("searching").is_none());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_an_entry_without_audio_fields_when_annotated_then_it_can_be_found() {
        let root = temp_cache("annotate");
        let cache = Cache::new(root.clone());
        let render = render_of(2_400);
        let key = fields().key();
        let blank = Fields {
            source_hash: String::new(),
            ..fields()
        };
        cache.store(&key, &render, 12_000, &blank).expect("storing");

        assert!(
            cache.find(&fields()).expect("searching").is_none(),
            "an entry without its fields cannot match"
        );
        cache.annotate(&key, &fields()).expect("annotating");
        assert!(
            cache.find(&fields()).expect("searching").is_some(),
            "the annotation makes it findable"
        );
        let meta = stored_meta(&cache, &key);
        assert_eq!(meta.source_hash, "source");
        assert_eq!(meta.segments.len(), 1, "annotating keeps the segment table");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_the_shipped_settings_when_a_segment_is_derived_then_it_holds_seven_steps() {
        assert_eq!(
            frames_per_segment(245_760, 48_000).expect("a segment"),
            1_384_320
        );
        assert!(frames_per_segment(48_000, 48_000).is_err(), "a zero step");
        assert!(
            frames_per_segment(24_000, 48_000).is_err(),
            "overlap is longer"
        );
        assert!(
            frames_per_segment(usize::MAX, 0).is_err(),
            "a segment length that overflows"
        );
    }

    #[test]
    fn given_the_same_inputs_when_keyed_twice_then_the_key_matches_and_a_setting_change_moves_it() {
        let a = key("source", 245_760, 48_000, "model", false);
        assert_eq!(a, key("source", 245_760, 48_000, "model", false));
        assert_eq!(a, fields().key());
        // Pinned so a change to the hash inputs shows up as a changed key, not as a silent
        // reuse of every v8 entry. The value is run-independent: blake3 over the six parts.
        assert_eq!(a, "17d39a88a4d96d9cb4d1c723672e442c");
        assert_ne!(a, key("source", 122_880, 48_000, "model", false));
        assert_ne!(a, key("source", 245_760, 24_000, "model", false));
        assert_ne!(a, key("source", 245_760, 48_000, "other-model", false));
        assert_ne!(a, key("source", 245_760, 48_000, "model", true));
        assert_eq!(a.len(), 32);
    }

    #[test]
    fn given_a_missing_models_directory_when_hashed_then_it_still_produces_an_id() {
        let id = model_id(Path::new("/nonexistent/flacify/models"));
        assert_eq!(id.len(), 16);
        assert_eq!(id, model_id(Path::new("/nonexistent/flacify/models")));
    }

    /// PCM that both containers carry unchanged: these samples already sit on the 24-bit
    /// grid, so a WAV and a FLAC of them hold the same audio.
    fn pcm_for_containers() -> Vec<f32> {
        let mut planar = Vec::with_capacity(4_800 * 2);
        for channel in 0..2 {
            for frame in 0..4_800 {
                planar.push(((frame * 31 + channel * 7) % 512 - 256) as f32 * QUANTUM);
            }
        }
        planar
    }

    #[test]
    fn given_the_same_pcm_as_wav_and_as_flac_when_hashed_then_the_hashes_differ_and_a_byte_copy_matches(
    ) {
        let root = temp_cache("source-hash-containers");
        std::fs::create_dir_all(&root).expect("creating the test directory");
        let wav = root.join("same-pcm.wav");
        let flac = root.join("same-pcm.flac");
        crate::wav::write(&wav, 48_000, 2, &pcm_for_containers()).expect("writing a wav");
        crate::flac::write(&flac, 48_000, 2, &pcm_for_containers()).expect("writing a flac");

        let wav_hash = source_hash(&wav).expect("hashing the wav");
        let flac_hash = source_hash(&flac).expect("hashing the flac");
        assert_ne!(
            wav_hash, flac_hash,
            "the key is the container, not the audio"
        );

        let copy = root.join("copy.wav");
        std::fs::copy(&wav, &copy).expect("copying the wav");
        assert_eq!(
            source_hash(&copy).expect("hashing the copy"),
            wav_hash,
            "the same bytes hash the same"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_file_of_several_chunks_when_hashed_then_it_is_blake3_of_all_its_bytes_in_32_characters(
    ) {
        let root = temp_cache("source-hash-bytes");
        std::fs::create_dir_all(&root).expect("creating the test directory");
        let path = root.join("bytes.bin");
        let bytes: Vec<u8> = (0..HASH_CHUNK * 2 + 3).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &bytes).expect("writing the file");

        let expected = blake3::hash(&bytes).to_hex().to_string();
        let actual = source_hash(&path).expect("hashing the file");
        assert_eq!(actual, expected[..32]);
        assert_eq!(actual.len(), 32);
        std::fs::remove_dir_all(root).ok();
    }
}
