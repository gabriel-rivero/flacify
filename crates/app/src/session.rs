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

use anyhow::Result;
use flacify_core::SAMPLE_RATE;
use flacify_infer::python::WorkerConfig;
use flacify_infer::WorkerSession;

use crate::cache::{self, Cache};
use crate::decode;
use crate::device;
use crate::frontier::Frontier;
use crate::render;
use crate::render_status::RenderStatus;
use crate::setup;
use crate::ui::TrackInfo;

/// The parts that survive one file: the worker, the cache, and the render settings.
pub struct Session {
    worker: Arc<WorkerSession>,
    cache: Arc<Cache>,
    model_id: String,
    splice: bool,
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
        let config = worker_config(paths)?;
        Ok(Self {
            worker: WorkerSession::new(config),
            cache: Arc::new(Cache::new(paths.cache_dir())),
            model_id: cache::model_id(&paths.models_dir()),
            splice,
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

    /// Decode a file, decide how to get its enhanced side, and start the device.
    pub fn open(&self, file: &Path) -> Result<Playback> {
        let render::Loaded {
            track, analysis, ..
        } = render::load(file)?;
        let original = track.interleaved();
        let tags = decode::tags(file);
        let name = file_name(file);
        let cutoff_hz = analysis.cutoff_hz;
        let audio_secs = track.frames as f64 / f64::from(SAMPLE_RATE);
        let path = file.to_path_buf();

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

        let key = cache::key(
            &render::sample_hash(&track),
            render::WINDOW,
            render::OVERLAP,
            &self.model_id,
            self.splice,
        );

        match self.cache.load(&key)? {
            Some(cached) => {
                eprintln!("cache hit: {audio_secs:.0} s of audio, nothing to render");
                let enhanced = render::interleave_stereo(&cached.planar, track.frames);
                let engine = device::AudioEngine::start(&original, Some(&enhanced))?;
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
                        device: cached.meta.device,
                        from_cache: true,
                        gain_db: cached.meta.gain_db,
                    },
                    path,
                    cancel: None,
                })
            }
            None if !device::progressive_supported() => {
                eprintln!(
                    "this output device is not 48 kHz stereo, so the track is rendered \
                     before playback starts"
                );
                let started = Instant::now();
                let cancel = AtomicBool::new(false);
                let (rendered, from_cache) = render::enhance_or_cached(
                    &track,
                    render::Job {
                        session: self.worker(),
                        cutoff_hz: Some(cutoff_hz),
                        seed: render::seed_for(&track),
                        splice_low_band: self.splice,
                    },
                    render::Store {
                        cache: &self.cache,
                        model_id: &self.model_id,
                    },
                    &cancel,
                    &mut |_| {},
                )?;
                let enhanced = render::interleave_stereo(&rendered.planar, track.frames);
                let engine = device::AudioEngine::start(&original, Some(&enhanced))?;
                eprintln!(
                    "rendered {audio_secs:.0} s of audio in {:.0} s",
                    started.elapsed().as_secs_f64()
                );
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
                        device: rendered.device,
                        from_cache,
                        gain_db: rendered.gain_db,
                    },
                    path,
                    cancel: None,
                })
            }
            None => {
                let frontier = Frontier::new(track.frames);
                let status = RenderStatus::new();
                let cancel = Arc::new(AtomicBool::new(false));
                {
                    let frontier = Arc::clone(&frontier);
                    let cache = Arc::clone(&self.cache);
                    let worker = self.worker();
                    let status = Arc::clone(&status);
                    let token = Arc::clone(&cancel);
                    let splice = self.splice;
                    std::thread::spawn(move || {
                        let result = render::render_into_frontier(
                            &track,
                            render::Job {
                                session: worker,
                                cutoff_hz: Some(cutoff_hz),
                                seed: render::seed_for(&track),
                                splice_low_band: splice,
                            },
                            &frontier,
                            &token,
                            &mut |_| {},
                        );
                        match result {
                            Ok(render) => {
                                status.finish();
                                if let Err(error) = cache.store(&key, &render, cutoff_hz) {
                                    tracing::warn!(%error, "could not cache the render");
                                }
                            }
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
                let engine = match device::AudioEngine::start_progressive(
                    &original,
                    Arc::clone(&frontier),
                ) {
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
        }
    }
}

fn worker_config(paths: &setup::Paths) -> Result<WorkerConfig> {
    if !paths.is_ready() {
        anyhow::bail!(
            "the model is not installed yet. Run `flacify setup` once (about 6 GB), \
             or set FLACIFY_HOME if it lives somewhere else."
        );
    }
    Ok(WorkerConfig {
        program: paths.venv_python(),
        worker_script: paths.worker_script(),
        model_dir: paths.models_dir(),
        flashsr_dir: Some(paths.flashsr_dir()),
        device: "auto".to_string(),
    })
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}
