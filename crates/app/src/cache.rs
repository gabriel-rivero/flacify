//! Render once, play forever: the FLAC cache.
//!
//! Rendering a three-minute track costs about 95 s at RTF 0.54, and without a cache
//! every play pays it again. The cache is keyed by everything that can change the
//! output — the decoded samples, the window and overlap, the pipeline version and the
//! model weights — so a hit is exactly the audio a fresh render would have produced,
//! and a change to any of those misses instead of serving the wrong file.
//!
//! One entry is one track: the finished stream as a 32-bit float WAV plus a small JSON
//! sidecar. A finished entry is written to a temporary directory and renamed, so a
//! half-written render can never be read back; the cost is that an interrupted render
//! starts over rather than resuming.
//!
//! **Why not FLAC yet.** The intent (ADR-003) is a FLAC cache, and `flacenc` writes one.
//! But symphonia 0.5.5 *and* claxon both decode `flacenc`'s stereo output with a broken
//! second channel — channel 0 exact, channel 1 zeroes and then noise — while libFLAC
//! (via libsndfile) reads the same bytes perfectly. It reproduces at 16 and 24 bits, with
//! side coding on and off, with LPC on and off, and single- or multi-threaded. Two
//! independent decoders agreeing means the file, not the readers, is the odd one out.
//! Float WAV is exact, costs about 69 MB for a three-minute track instead of ~30 MB, and
//! keeps the cache honest until that is resolved. Roadmap: cache, item 1.
//!
//! Seeking needs nothing from the cache beyond what playback already does: the file is
//! decoded into memory and the playhead is an index.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::render::Render;

/// Bump this when anything about the pipeline changes the samples: window, overlap,
/// preprocessing, the seed schedule, the limiter.
pub const CACHE_VERSION: &str = "flacify-cache-3";

/// 24-bit has about 144 dB of headroom, well under the float noise floor of the model.

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Meta {
    pub version: String,
    pub frames: usize,
    pub sample_rate: u32,
    pub cutoff_hz: u32,
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
}

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

    /// `None` when there is no entry; an error when there is one and it cannot be read,
    /// because silence about a corrupt cache entry is how you end up debugging audio.
    pub fn load(&self, key: &str) -> Result<Option<Cached>> {
        let dir = self.entry_dir(key);
        let meta_path = dir.join("meta.json");
        let audio_path = dir.join("audio.wav");
        if !meta_path.is_file() || !audio_path.is_file() {
            return Ok(None);
        }

        let meta: Meta = serde_json::from_str(
            &std::fs::read_to_string(&meta_path)
                .with_context(|| format!("reading {}", meta_path.display()))?,
        )
        .with_context(|| format!("parsing {}", meta_path.display()))?;

        if meta.version != CACHE_VERSION {
            return Ok(None);
        }

        let mut planar = read_audio(&audio_path)?;
        if planar.len() / 2 < meta.frames {
            anyhow::bail!(
                "{} holds {} frames, its metadata says {}",
                audio_path.display(),
                planar.len() / 2,
                meta.frames
            );
        }
        // The encoder pads the last block; the metadata is the authority.
        planar.truncate(meta.frames * 2);
        if planar.len() / 2 != meta.frames {
            anyhow::bail!(
                "{} holds {} frames, its metadata says {}",
                audio_path.display(),
                planar.len() / 2,
                meta.frames
            );
        }
        Ok(Some(Cached { planar, meta }))
    }

    /// Write a finished render. Everything lands in `<key>.tmp` first and is renamed,
    /// so a reader never sees a partial entry.
    pub fn store(&self, key: &str, render: &Render, cutoff_hz: u32) -> Result<PathBuf> {
        let frames = render.planar.len() / 2;
        let final_dir = self.entry_dir(key);
        let temp_dir = self.root.join(format!("{key}.tmp"));
        if temp_dir.exists() {
            std::fs::remove_dir_all(&temp_dir)
                .with_context(|| format!("clearing {}", temp_dir.display()))?;
        }
        std::fs::create_dir_all(&temp_dir)
            .with_context(|| format!("creating {}", temp_dir.display()))?;

        let audio_path = temp_dir.join("audio.wav");
        write_audio(&audio_path, &render.planar)?;

        let meta = Meta {
            version: CACHE_VERSION.to_string(),
            frames,
            sample_rate: flacify_core::SAMPLE_RATE,
            cutoff_hz,
            peak_before: render.limit.peak_before,
            peak_after: render.limit.peak_after,
            limited_samples: render.limit.limited_samples,
            gain_db: render.gain_db,
            device: render.device.clone(),
        };
        std::fs::write(
            temp_dir.join("meta.json"),
            serde_json::to_string_pretty(&meta)? + "\n",
        )
        .with_context(|| format!("writing metadata in {}", temp_dir.display()))?;

        if final_dir.exists() {
            std::fs::remove_dir_all(&final_dir)
                .with_context(|| format!("replacing {}", final_dir.display()))?;
        }
        std::fs::rename(&temp_dir, &final_dir)
            .with_context(|| format!("publishing {}", final_dir.display()))?;
        Ok(final_dir)
    }
}

/// Read a cached track back as planar stereo at 48 kHz. Also used by tests that need a
/// decoder for a file the cache wrote.
fn read_audio(path: &Path) -> Result<Vec<f32>> {
    let decoded = crate::decode::decode(path)?;
    if decoded.sample_rate != flacify_core::SAMPLE_RATE || decoded.channels != 2 {
        anyhow::bail!(
            "{} is {} Hz with {} channels; cache entries are 48 kHz stereo",
            path.display(),
            decoded.sample_rate,
            decoded.channels
        );
    }
    let planar = crate::decode::to_stereo_48k(&decoded)?;
    if planar.len() != decoded.frames * 2 {
        anyhow::bail!("{} decodes to an unexpected length", path.display());
    }
    Ok(planar)
}

/// Write the finished stream as a 32-bit float WAV, which is exact: every byte of the
/// model's output survives, and a person can open a cache entry to hear what it holds.
fn write_audio(path: &Path, planar: &[f32]) -> Result<()> {
    crate::wav::write(path, flacify_core::SAMPLE_RATE, 2, planar)
}

/// Everything that can change the rendered samples, hashed into a directory name.
///
/// The sample hash comes from the decoded track, not the file, so the same audio in a
/// different container or bitrate shares nothing and the same file repackaged is still a
/// miss. That is deliberate: the cache is cheap to rebuild and a wrong hit is not.
pub fn key(
    sample_hash: &str,
    window: usize,
    overlap: usize,
    model_id: &str,
    splice_low_band: bool,
) -> String {
    let mut hasher = blake3::Hasher::new();
    for part in [
        CACHE_VERSION,
        sample_hash,
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

    fn render_of(frames: usize) -> Render {
        let mut planar = Vec::with_capacity(frames * 2);
        for channel in 0..2 {
            for frame in 0..frames {
                let value = (frame as f32 * 0.01).sin() * 0.4 + channel as f32 * 0.1;
                planar.push(value);
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

    #[test]
    fn given_a_stored_render_when_loaded_then_the_samples_survive_the_round_trip() {
        let root = temp_cache("roundtrip");
        let cache = Cache::new(root.clone());
        let render = render_of(4_800);
        cache.store("key", &render, 15_328).expect("storing");

        let cached = cache.load("key").expect("loading").expect("an entry");
        assert_eq!(cached.meta.frames, 4_800);
        assert_eq!(cached.meta.cutoff_hz, 15_328);
        assert_eq!(cached.planar.len(), render.planar.len());
        println!(
            "DEBUG expected {:?} got {:?} len {} vs {}",
            &render.planar[..6],
            &cached.planar[..6],
            render.planar.len(),
            cached.planar.len()
        );
        let (index, worst) = cached
            .planar
            .iter()
            .zip(render.planar.iter())
            .enumerate()
            .map(|(i, (a, b))| (i, (a - b).abs()))
            .fold(
                (0usize, 0.0f32),
                |acc, item| if item.1 > acc.1 { item } else { acc },
            );
        let bad: Vec<usize> = cached
            .planar
            .iter()
            .zip(render.planar.iter())
            .enumerate()
            .filter(|(_, (a, b))| (*a - *b).abs() > 1e-3)
            .map(|(i, _)| i)
            .collect();
        for frame in [0usize, 1, 100, 1000, 2048, 3581, 4095, 4200, 4799] {
            println!(
                "DEBUG f{frame}: got {:.5} want {:.5}",
                cached.planar[4800 + frame],
                render.planar[4800 + frame]
            );
        }
        println!(
            "DEBUG worst at {index}: expected {:?} got {:?}; {} samples off by >1e-3, first {:?} last {:?}",
            render.planar.get(index),
            cached.planar.get(index),
            bad.len(),
            bad.first(),
            bad.last()
        );
        // 24-bit quantum is 6e-8; allow a little more for the encoder's rounding.
        assert!(worst < 1e-6, "worst sample error {worst}");
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
            .store("key", &render_of(240), 12_000)
            .expect("storing");
        let meta_path = cache.entry_dir("key").join("meta.json");
        let mut meta: Meta =
            serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
        meta.version = "something-older".into();
        std::fs::write(&meta_path, serde_json::to_string(&meta).unwrap()).unwrap();

        assert!(cache.load("key").expect("no error").is_none());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_the_same_inputs_when_keyed_twice_then_the_key_matches_and_a_window_change_moves_it() {
        let a = key("samples", 245_760, 48_000, "model", false);
        assert_eq!(a, key("samples", 245_760, 48_000, "model", false));
        assert_ne!(a, key("samples", 122_880, 48_000, "model", false));
        assert_ne!(a, key("samples", 245_760, 48_000, "other-model", false));
        assert_ne!(a, key("samples", 245_760, 48_000, "model", true));
        assert_eq!(a.len(), 32);
    }

    #[test]
    fn given_a_missing_models_directory_when_hashed_then_it_still_produces_an_id() {
        let id = model_id(Path::new("/nonexistent/flacify/models"));
        assert_eq!(id.len(), 16);
        assert_eq!(id, model_id(Path::new("/nonexistent/flacify/models")));
    }
}
