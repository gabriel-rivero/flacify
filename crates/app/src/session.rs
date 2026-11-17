//! One app run: the installed paths, one resident model, and a file becoming playback.
//!
//! The session exists so the worker outlives a track. `flacify play` is a long-lived
//! window that can take another file (drag and drop), and the second file must not pay
//! the 8–22 s weight load again. The worker starts lazily on the first render that needs
//! it and stops when the app does; a cache hit or a full-band file never loads it.
//!
//! [`Session::open`] is the whole of `play` minus the window: decode, decide, start the
//! device, and leave a background renderer filling a frontier. It returns a [`Playback`]
//! that owns the cancellation token for that renderer, so moving to another file stops
//! the old work instead of racing it for the GPU.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use flacify_core::SAMPLE_RATE;
use flacify_infer::python::WorkerConfig;
use flacify_infer::WorkerSession;

use crate::cache::{self, Cache};
use crate::decode;
use crate::device;
use crate::frontier::Frontier;
use crate::render;
use crate::render_status::RenderStatus;
use crate::segments;
use crate::setup;
use crate::source;
use crate::ui::TrackInfo;

/// The parts that survive one file: the worker, the cache, and the render settings.
pub struct Session {
    worker: Arc<WorkerSession>,
    cache: Arc<Cache>,
    model_id: String,
    splice: bool,
    /// False when the model is not installed. A full-band file or a cache hit still plays;
    /// anything that has to render asks for setup instead.
    ready: bool,
}

/// The file needs a render and the model is not installed. Not a failure: `play` turns
/// this into the first-run window.
#[derive(Debug)]
pub struct NeedsSetup;

impl std::fmt::Display for NeedsSetup {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("the model is not installed yet; run `flacify setup` once")
    }
}

impl std::error::Error for NeedsSetup {}

/// The file a playback belongs to: where it lives, the name the window shows, and its tags.
struct FileInfo {
    path: PathBuf,
    name: String,
    tags: decode::Tags,
}

impl FileInfo {
    fn of(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            name: file_name(path),
            tags: decode::tags(path),
        }
    }
}

/// Whether an error means the file needs the model installed rather than something broke.
pub fn needs_setup(error: &anyhow::Error) -> bool {
    error.downcast_ref::<NeedsSetup>().is_some()
}

/// Everything one file owns. Dropping it is not enough to stop the render, because the
/// render thread holds its own `Arc`s: call [`Playback::cancel`] first.
pub struct Playback {
    pub engine: device::AudioEngine,
    pub frontier: Option<Arc<Frontier>>,
    pub status: Option<Arc<RenderStatus>>,
    pub info: TrackInfo,
    pub path: PathBuf,
    /// Present when a background renderer is running for this file.
    pub(crate) cancel: Option<Arc<AtomicBool>>,
}

impl Playback {
    /// Stop this file's renderer. The thread returns at the next window boundary, which
    /// is the finest cancellation the protocol allows.
    pub fn cancel(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            cancel.store(true, Ordering::Relaxed);
        }
    }
}

impl Session {
    pub fn new(paths: &setup::Paths, splice: bool) -> Result<Self> {
        Ok(Self {
            worker: WorkerSession::new(worker_config(paths)),
            cache: Arc::new(Cache::new(paths.cache_dir())),
            model_id: cache::model_id(&paths.models_dir()),
            splice,
            ready: paths.is_ready(),
        })
    }

    pub fn worker(&self) -> Arc<WorkerSession> {
        Arc::clone(&self.worker)
    }

    pub fn cache(&self) -> &Cache {
        &self.cache
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// A render needs the worker; a cache hit or a full-band source does not.
    fn require_ready(&self) -> Result<()> {
        if self.ready {
            Ok(())
        } else {
            Err(anyhow::Error::new(NeedsSetup))
        }
    }

    /// Decode a file, decide how to get its enhanced side, and start the device.
    pub fn open(&self, file: &Path) -> Result<Playback> {
        let shown = FileInfo::of(file);
        let fields = cache::Fields {
            source_hash: cache::source_hash(file)?,
            window: render::WINDOW,
            overlap: render::OVERLAP,
            model_id: self.model_id.clone(),
            splice: self.splice,
        };
        let key = fields.key();

        // A complete entry plays as segments, so the track is never decoded whole. The
        // first play of an uncached track keeps the progressive path, which still holds
        // the whole track: replacing that is a later change, not this one.
        if let Some(rendered) = self.complete_entry(&key, &fields)? {
            return self.play_entry(file, &key, rendered, &fields, shown);
        }

        let FileInfo { path, name, tags } = shown;
        let render::Loaded {
            track,
            analysis,
            source_entry,
            ..
        } = render::load_hashed(file, self.cache.root(), fields.source_hash.clone())?;
        let original = track.interleaved();
        let cutoff_hz = analysis.cutoff_hz;
        let audio_secs = track.frames as f64 / f64::from(SAMPLE_RATE);

        // A full-band source has nothing to infer, so it never needs the model.
        if analysis.looks_full_band() {
            eprintln!(
                "already full-band ({:.4}% of the spectrum above 21 kHz); nothing to generate",
                analysis.high_band_ratio * 100.0
            );
            let engine = device::AudioEngine::start(&original, None)?;
            return Ok(Playback {
                engine,
                frontier: None,
                status: None,
                info: TrackInfo {
                    name,
                    title: tags.title,
                    artist: tags.artist,
                    cutoff_hz,
                    audio_secs,
                    device: "none".to_string(),
                    from_cache: false,
                    gain_db: 0.0,
                },
                path,
                cancel: None,
            });
        }

        // The remaining paths render, so they need the model and the source tier.
        self.require_ready()?;
        let entry = source_entry.context("the source tier a render reads")?;

        if !device::progressive_supported() {
            eprintln!(
                "this output device is not 48 kHz stereo, so the track is rendered \
                 before playback starts"
            );
            let started = Instant::now();
            let cancel = AtomicBool::new(false);
            let fields = fields.clone();
            let mut writer = self.cache.begin(
                &key,
                &cache::EntrySpec {
                    frames: track.frames,
                    cutoff_hz,
                    fields,
                },
            )?;
            let streamed = render::render_segments(
                &entry,
                render::Job {
                    session: self.worker(),
                    cutoff_hz: Some(cutoff_hz),
                    seed: render::seed_for(&track),
                    splice_low_band: self.splice,
                },
                &mut writer,
                &cancel,
                &mut |_| {},
                |_, _| Ok(()),
            )?;
            // The render completed the entry on disk; playback reads exactly the
            // audio a later cache hit would.
            let cached = self
                .cache
                .load(&key)?
                .context("the cache entry the render just completed")?;
            let enhanced = render::interleave_stereo(&cached.planar, track.frames);
            let engine = device::AudioEngine::start(&original, Some(&enhanced))?;
            eprintln!(
                "rendered {audio_secs:.0} s of audio in {:.0} s",
                started.elapsed().as_secs_f64()
            );
            return Ok(Playback {
                engine,
                frontier: None,
                status: None,
                info: TrackInfo {
                    name,
                    title: tags.title,
                    artist: tags.artist,
                    cutoff_hz,
                    audio_secs,
                    device: streamed.device,
                    from_cache: false,
                    gain_db: streamed.gain_db,
                },
                path,
                cancel: None,
            });
        }

        let frontier = Frontier::new(track.frames);
        let status = RenderStatus::new();
        let cancel = Arc::new(AtomicBool::new(false));
        let frames = track.frames;
        {
            let frontier = Arc::clone(&frontier);
            let cache = Arc::clone(&self.cache);
            let worker = self.worker();
            let status = Arc::clone(&status);
            let token = Arc::clone(&cancel);
            let splice = self.splice;
            let key = key.clone();
            let fields = fields.clone();
            std::thread::spawn(move || {
                // The seed is a whole-track hash by contract (ADR-006); the track
                // is dropped as soon as it is computed so the render holds a
                // segment rather than a track.
                let seed = render::seed_for(&track);
                drop(track);
                let spec = cache::EntrySpec {
                    frames,
                    cutoff_hz,
                    fields,
                };
                let result = cache.begin(&key, &spec).and_then(|mut writer| {
                    let rendered = render::render_segments_into_frontier(
                        &entry,
                        render::Job {
                            session: worker,
                            cutoff_hz: Some(cutoff_hz),
                            seed,
                            splice_low_band: splice,
                        },
                        &mut writer,
                        &frontier,
                        &token,
                        &mut |_| {},
                    );
                    // A killed or cancelled render leaves the segments it wrote
                    // for the next attempt to resume.
                    if rendered.is_err() {
                        writer.abandon();
                    }
                    rendered
                });
                match result {
                    Ok(_) => status.finish(),
                    // Another file took over; the window is not watching this one.
                    Err(error) if render::is_cancelled(&error) => {}
                    Err(error) => {
                        // The window shows this too; tracing is for the terminal.
                        tracing::error!(%error, "rendering failed");
                        status.fail(format!("{error:#}"));
                    }
                }
            });
        }
        let engine = match device::AudioEngine::start_progressive(&original, Arc::clone(&frontier))
        {
            Ok(engine) => engine,
            Err(error) => {
                // The renderer is already running behind this file. Do not
                // leave it filling a frontier nobody will play.
                cancel.store(true, Ordering::Relaxed);
                return Err(error);
            }
        };
        // A resident worker already has a device; the first render cannot say
        // until it has loaded, which is exactly why it says "rendering".
        let device = self
            .worker
            .device()
            .unwrap_or_else(|| "rendering".to_string());
        Ok(Playback {
            engine,
            frontier: Some(frontier),
            status: Some(status),
            info: TrackInfo {
                name,
                title: tags.title,
                artist: tags.artist,
                cutoff_hz,
                audio_secs,
                device,
                from_cache: false,
                gain_db: 0.0,
            },
            path,
            cancel: Some(cancel),
        })
    }

    /// The complete cache entry that can back this file. By key when the model is
    /// installed; by the source file and settings when it is not, because without the
    /// weights the id cannot be computed and an existing render still plays.
    fn complete_entry(&self, key: &str, fields: &cache::Fields) -> Result<Option<cache::Segments>> {
        match self.cache.segments(key)? {
            Some(rendered) => Ok(Some(rendered)),
            None if !self.ready => self.cache.find_segments(fields),
            None => Ok(None),
        }
    }

    /// Play a complete cache entry from its segments: neither side is decoded whole and
    /// nothing renders, which is what makes a second play cost no GPU time.
    fn play_entry(
        &self,
        file: &Path,
        key: &str,
        rendered: cache::Segments,
        fields: &cache::Fields,
        shown: FileInfo,
    ) -> Result<Playback> {
        let FileInfo { path, name, tags } = shown;
        if self.ready && rendered.meta().source_hash.is_empty() {
            if let Err(error) = self.cache.annotate(key, fields) {
                tracing::warn!(%error, "could not annotate the cache entry");
            }
        }
        let frames_per_segment = cache::frames_per_segment(render::WINDOW, render::OVERLAP)?;
        let tier = source::Tier::new(self.cache.root(), frames_per_segment)?;
        let source = tier.build_or_reuse(file, &fields.source_hash)?;
        let analysis = render::analyse_streaming(&source)
            .with_context(|| format!("analysing {}", file.display()))?;
        let frames = source.meta().frames;
        let audio_secs = frames as f64 / f64::from(SAMPLE_RATE);
        let cutoff_hz = analysis.cutoff_hz;
        let gain_db = rendered.meta().gain_db.unwrap_or(0.0);
        let device = rendered.meta().device.clone();

        // A device that is not 48 kHz stereo cannot take segments, so it keeps the path
        // every cache hit used before: both sides whole, resampled once by `start`.
        if !device::progressive_supported() {
            let has_enhanced = !analysis.looks_full_band();
            let original = render::interleave_stereo(&source.read_all()?, frames);
            let enhanced = match has_enhanced {
                true => Some(render::interleave_stereo(&rendered.read_all()?, frames)),
                false => None,
            };
            let engine = device::AudioEngine::start(&original, enhanced.as_deref())?;
            return Ok(Playback {
                engine,
                frontier: None,
                status: None,
                info: TrackInfo {
                    name,
                    title: tags.title,
                    artist: tags.artist,
                    cutoff_hz,
                    audio_secs,
                    device: if has_enhanced {
                        device
                    } else {
                        "none".to_string()
                    },
                    from_cache: true,
                    gain_db: if has_enhanced { gain_db } else { 0.0 },
                },
                path,
                cancel: None,
            });
        }

        // This build never renders full-band audio, so an entry for it can only come from
        // somewhere else. The source tier is still a passthrough, one segment at a time.
        if analysis.looks_full_band() {
            eprintln!(
                "already full-band ({:.4}% of the spectrum above 21 kHz); nothing to generate",
                analysis.high_band_ratio * 100.0
            );
            let reader = segments::Reader::open(Arc::new(source), None)?;
            let engine = device::AudioEngine::start_segmented(reader)?;
            return Ok(Playback {
                engine,
                frontier: None,
                status: None,
                info: TrackInfo {
                    name,
                    title: tags.title,
                    artist: tags.artist,
                    cutoff_hz,
                    audio_secs,
                    device: "none".to_string(),
                    from_cache: true,
                    gain_db: 0.0,
                },
                path,
                cancel: None,
            });
        }

        eprintln!("cache hit: {audio_secs:.0} s of audio, nothing to render");
        let reader = segments::Reader::open(Arc::new(source), Some(Arc::new(rendered)))?;
        let engine = device::AudioEngine::start_segmented(reader)?;
        Ok(Playback {
            engine,
            frontier: None,
            status: None,
            info: TrackInfo {
                name,
                title: tags.title,
                artist: tags.artist,
                cutoff_hz,
                audio_secs,
                device,
                from_cache: true,
                gain_db,
            },
            path,
            cancel: None,
        })
    }
}

fn worker_config(paths: &setup::Paths) -> WorkerConfig {
    WorkerConfig {
        program: paths.venv_python(),
        worker_script: paths.worker_script(),
        model_dir: paths.models_dir(),
        flashsr_dir: Some(paths.flashsr_dir()),
        device: "auto".to_string(),
        timeout: flacify_infer::python::DEFAULT_EXCHANGE_TIMEOUT,
        handshake_timeout: flacify_infer::python::DEFAULT_HANDSHAKE_TIMEOUT,
    }
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limit::LimitReport;

    /// A unique data directory in the temp directory, deleted by the test that made it.
    fn temp_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "flacify-session-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("creating the test directory");
        root
    }

    /// A complete cache entry for `source`, at the key this session would look for. The
    /// samples are irrelevant to the decision, so they are zeros.
    fn store_entry(session: &Session, source: &Path) -> (String, cache::Fields, usize) {
        let frames = 4_800;
        let fields = cache::Fields {
            source_hash: cache::source_hash(source).expect("hashing the source"),
            window: render::WINDOW,
            overlap: render::OVERLAP,
            model_id: session.model_id().to_string(),
            splice: false,
        };
        let render = render::Render {
            planar: vec![0.0f32; frames * 2],
            limit: LimitReport {
                peak_before: 0.0,
                peak_after: 0.0,
                limited_samples: 0,
            },
            device: "cpu".to_string(),
            gain_db: 0.0,
            lufs_input: -14.0,
            lufs_output: -14.0,
        };
        let key = fields.key();
        session
            .cache()
            .store(&key, &render, 12_000, &fields)
            .expect("storing the entry");
        (key, fields, frames)
    }

    #[test]
    fn given_a_complete_cache_entry_when_a_file_is_opened_then_playback_is_backed_by_segments() {
        let root = temp_root("entry");
        let source = root.join("source.wav");
        crate::wav::write(&source, 48_000, 2, &vec![0.0f32; 4_800 * 2]).expect("writing the wav");
        let session = Session::new(&setup::Paths::at(&root), false).expect("a session");
        let (key, fields, frames) = store_entry(&session, &source);

        let rendered = session
            .complete_entry(&key, &fields)
            .expect("looking the entry up")
            .expect("the complete entry backs this file whole");

        // The value carries no samples: playback reads it a segment at a time.
        assert_eq!(rendered.meta().frames, frames);
        assert_eq!(
            rendered.read_segment(0).expect("one segment").len(),
            frames * 2
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_no_complete_entry_when_a_file_is_opened_then_the_path_that_needs_a_track_is_taken() {
        let root = temp_root("no-entry");
        let source = root.join("source.wav");
        crate::wav::write(&source, 48_000, 2, &vec![0.0f32; 4_800 * 2]).expect("writing the wav");
        let session = Session::new(&setup::Paths::at(&root), false).expect("a session");
        let (key, fields, _frames) = store_entry(&session, &source);

        // Another file's key and source hash: neither the key nor the field search matches.
        let other = cache::Fields {
            source_hash: "another source".to_string(),
            ..fields.clone()
        };
        assert!(session
            .complete_entry(&other.key(), &other)
            .expect("looking the entry up")
            .is_none());
        assert!(session.cache().segments(&key).expect("a lookup").is_some());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_needs_setup_error_when_asked_then_it_is_recognized_and_others_are_not() {
        assert!(needs_setup(&anyhow::Error::new(NeedsSetup)));
        assert!(!needs_setup(&anyhow::anyhow!("the worker exited")));
        assert_eq!(
            NeedsSetup.to_string(),
            "the model is not installed yet; run `flacify setup` once"
        );
    }
}
