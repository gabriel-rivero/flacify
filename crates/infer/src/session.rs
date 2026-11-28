//! One model worker, kept for the life of the app instead of the life of a render.
//!
//! The worker costs 8–22 s to load its weights, and a player that pays that on every
//! file is a renderer with a window on it. This session starts the child lazily — the
//! first render that actually needs it — and reuses it for every later render in the
//! same process. Nothing starts it up front: a cache hit or a full-band file never
//! loads the model at all, and closing the app drops the session, which kills the child
//! (`Connection::drop`), so there is no daemon and no idle 3 GB.
//!
//! A failed exchange leaves the protocol stream in an unknown state. The model is dropped
//! on the spot so the child is killed, and the window is retried once on a fresh child, so
//! one worker death costs a load rather than the track. A worker that stops answering is
//! killed by the exchange watchdog and handled by the same retry.

use std::sync::{Arc, Mutex, MutexGuard};

use flacify_core::{Error, Result};

use crate::model::Model;
use crate::python::{Hello, PythonWorker, WorkerConfig, MODEL_LATENCY_SAMPLES};

/// A worker that lives as long as the caller holds the `Arc`.
pub struct WorkerSession {
    config: WorkerConfig,
    /// Only the fake worker uses these; the real sidecar takes none.
    extra_args: Vec<String>,
    worker: Mutex<Option<PythonWorker>>,
    /// The device the first load reported. Set once and never changed, so a poll taken
    /// during a render — when the worker lock is held for the whole forward pass — still
    /// answers instead of reporting "not known".
    device: std::sync::OnceLock<String>,
}

impl WorkerSession {
    pub fn new(config: WorkerConfig) -> Arc<Self> {
        Self::with_args(config, Vec::new())
    }

    /// [`new`](Self::new) with command-line arguments for the child. The real worker
    /// accepts none; this exists for the fake worker the tests drive.
    #[doc(hidden)]
    pub fn with_args(config: WorkerConfig, extra_args: Vec<String>) -> Arc<Self> {
        Arc::new(Self {
            config,
            extra_args,
            worker: Mutex::new(None),
            device: std::sync::OnceLock::new(),
        })
    }

    /// The model for one render. Starts the child if this is the first call.
    ///
    /// Takes the `Arc` by value so the returned handle owns a reference and the worker
    /// cannot be dropped while a render still holds it.
    pub fn model(self: Arc<Self>) -> Result<ResidentModel> {
        let hello = self.ensure()?;
        Ok(ResidentModel {
            session: self,
            hello,
        })
    }

    /// The accelerator the worker chose, once it has started (`cuda`, `mps`, `cpu`).
    ///
    /// `try_lock` is not enough on its own: a window holds the lock for its whole forward
    /// pass, so a poll during a render would miss and report "not known" for the rest of
    /// the track. The answer never changes once known, so it is recorded the first time it
    /// is read and served from there — the same reason the window re-reads instead of
    /// keeping the value it got at open, when there was nothing to read.
    pub fn device(&self) -> Option<String> {
        if let Some(known) = self.device.get() {
            return Some(known.clone());
        }
        let worker = self.worker.try_lock().ok()?;
        let device = worker
            .as_ref()
            .map(|worker| worker.hello().device.clone())?;
        let _ = self.device.set(device.clone());
        Some(device)
    }

    /// Whether the child is loaded right now. The lazy spawn is what makes this worth
    /// asking: the app must not claim a device it never started.
    pub fn is_loaded(&self) -> bool {
        self.worker
            .try_lock()
            .map(|worker| worker.is_some())
            .unwrap_or(false)
    }

    fn ensure(&self) -> Result<Hello> {
        let mut worker = self.lock()?;
        if let Some(running) = worker.as_ref() {
            return Ok(running.hello().clone());
        }
        // The lock is held across the load on purpose: two render threads must not each
        // start their own child and then race to replace it.
        let started = PythonWorker::spawn_with_args(&self.config, &self.extra_args)?;
        let hello = started.hello().clone();
        // Recorded at the moment it is known, not on first poll: a poll during the first
        // window would find the lock held and report "not known" forever after.
        let _ = self.device.set(hello.device.clone());
        *worker = Some(started);
        Ok(hello)
    }

    /// Drop whatever child is there and start a new one, for a render that outlived its
    /// worker.
    fn restart(&self) -> Result<()> {
        {
            let mut worker = self.lock()?;
            *worker = None;
        }
        self.ensure().map(|_| ())
    }

    fn lock(&self) -> Result<MutexGuard<'_, Option<PythonWorker>>> {
        self.worker
            .lock()
            .map_err(|_| Error::Inference("the model worker lock is poisoned".to_string()))
    }
}

/// A cheap handle a render can own: cloning it does not spawn anything.
#[derive(Clone)]
pub struct ResidentModel {
    session: Arc<WorkerSession>,
    hello: Hello,
}

impl ResidentModel {
    pub fn hello(&self) -> &Hello {
        &self.hello
    }

    /// One attempt. A failed exchange leaves the pipe in an unknown state, so the child is
    /// dropped and the next call starts fresh rather than trusting it.
    fn window(
        &self,
        input: &[f32],
        noise: &[f32],
        channels: usize,
        frames: usize,
    ) -> Result<Vec<f32>> {
        let mut worker = self.session.lock()?;
        let Some(running) = worker.as_mut() else {
            return Err(Error::Inference(
                "the model worker is not running; reload the file to start a new one".to_string(),
            ));
        };
        let result = running.run(input, noise, channels, frames);
        if result.is_err() {
            // Dropping the worker kills the child.
            *worker = None;
        }
        result
    }
}

impl Model for ResidentModel {
    fn run(
        &mut self,
        input: &[f32],
        noise: &[f32],
        channels: usize,
        frames: usize,
    ) -> Result<Vec<f32>> {
        match self.window(input, noise, channels, frames) {
            Ok(output) => Ok(output),
            Err(_) => {
                // The child died between windows. The model keeps no state across a
                // window, so a fresh child and one retry produce the same audio as the
                // first attempt would have; if the retry fails too, that error is the one
                // the render sees.
                self.session.restart()?;
                self.window(input, noise, channels, frames)
            }
        }
    }

    fn latency_samples(&self) -> usize {
        MODEL_LATENCY_SAMPLES
    }

    fn noise_len(&self, channels: usize, frames: usize) -> usize {
        self.hello.noise_len(channels, frames)
    }

    fn name(&self) -> &str {
        "flashsr-sidecar"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn config() -> WorkerConfig {
        WorkerConfig {
            program: PathBuf::from("/nonexistent/python"),
            worker_script: PathBuf::from("/nonexistent/worker.py"),
            model_dir: PathBuf::from("/nonexistent/models"),
            flashsr_dir: None,
            device: "cpu".to_string(),
            timeout: crate::python::DEFAULT_EXCHANGE_TIMEOUT,
            handshake_timeout: crate::python::DEFAULT_HANDSHAKE_TIMEOUT,
        }
    }

    fn hello() -> Hello {
        Hello {
            device: "cpu".to_string(),
            torch: "0".to_string(),
            window: 245_760,
            sample_rate: 48_000,
            noise_len_stereo: 1_920,
            latent_shape_stereo: vec![4, 8, 60],
            load_secs: 0.0,
            warmup_secs: 0.0,
        }
    }

    #[test]
    fn given_a_new_session_when_asked_then_nothing_has_started() {
        let session = WorkerSession::new(config());
        assert!(!session.is_loaded());
        assert!(session.device().is_none());
    }

    #[test]
    fn given_a_recorded_device_when_the_worker_lock_is_busy_then_it_still_answers() {
        // A window holds the worker lock for its whole forward pass, so `try_lock` during a
        // render fails and `device()` would report "not known" for the life of the track —
        // which is exactly how the status line said `model rendering` while the GPU worked.
        // Once known, the device is recorded and served without the lock.
        let session = WorkerSession::new(config());
        assert!(session.device.set("cuda".to_string()).is_ok());
        let _held = session.worker.lock().expect("holding the worker lock");
        assert_eq!(session.device().as_deref(), Some("cuda"));
    }

    #[test]
    fn given_the_model_window_when_sized_then_it_uses_the_latent_grid() {
        let session = WorkerSession::new(config());
        let model = ResidentModel {
            session,
            hello: hello(),
        };
        assert_eq!(model.noise_len(2, 245_760), 1_920);
    }

    #[test]
    fn given_a_shorter_window_when_sized_then_the_noise_scales_with_it() {
        let session = WorkerSession::new(config());
        let model = ResidentModel {
            session,
            hello: hello(),
        };
        // One granule of the same latent grid, not one sample per audio sample.
        assert_eq!(model.noise_len(2, 30_720), 240);
        assert_eq!(model.noise_len(2, 61_440), 480);
        assert_eq!(model.noise_len(1, 245_760), 960);
    }
}
