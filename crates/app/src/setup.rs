//! First-run provisioning of the Python sidecar (ADR-010).
//!
//! `ensure` installs uv, CPython 3.12, a virtualenv, the pinned wheels, the two worker
//! sources and the 3.3 GB of FlashSR weights into one data directory, and records each
//! finished step in `state.json` so a crash resumes instead of restarting.
//!
//! The rules live in pure functions; every side effect goes through [`Runner`], which is
//! what makes the state machine testable with no network, no uv and no Python.

use std::env;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};

/// The worker and its model wrapper, embedded at build time.
const WORKER_PY: &str = include_str!("../../../tools/worker.py");
const FLASHSR_ONNX_PY: &str = include_str!("../../../tools/flashsr_onnx.py");

const STATE_FILE: &str = "state.json";
const STATE_VERSION: u32 = 1;

const STEP_UV: &str = "uv";
const STEP_PYTHON: &str = "python";
const STEP_VENV: &str = "venv";
const STEP_WHEELS: &str = "wheels";
const STEP_FLASHSR: &str = "flashsr";
const STEP_WORKER: &str = "worker";
const STEP_WEIGHTS: &str = "weights";
const STEP_YTDLP: &str = "yt-dlp";

const HOME_ENV: &str = "FLACIFY_HOME";
const UV_ENV: &str = "FLACIFY_UV";

/// The PyTorch index that carries CUDA builds of the pinned torch. `cu132` is the tag
/// measured on Windows (2026-11-15) that publishes torch 2.15.0.
pub const CUDA_TORCH_INDEX: &str = "https://download.pytorch.org/whl/cu132";

/// Approximate download sizes for the setup screen. The weights are exact.
///
/// `WHEELS_BYTES` is the macOS/PyPI figure. A Linux CUDA install measured 3.20 GB of
/// compressed wheels on 2026-11-12 (PyPI's torch pulls the whole CUDA 13 stack), so the
/// screen understates that first run by about 0.7 GB; see the Linux section of
/// `docs/measurements.md`.
const UV_BYTES: u64 = 20_000_000;
const PYTHON_BYTES: u64 = 30_000_000;
const VENV_BYTES: u64 = 1_000_000;
const WHEELS_BYTES: u64 = 2_500_000_000;

/// The standalone yt-dlp binary, about the size of the Windows build. It is a single
/// self-contained executable, not a pip package: a URL then needs no Python and no venv,
/// which is what lets a machine that never installed the model still play a link.
const YTDLP_BYTES: u64 = 24_000_000;

/// The three weight files and their sizes in the upstream dataset repo.
const WEIGHT_FILES: [(&str, u64); 3] = [
    ("student_ldm.pth", 1_033_200_174),
    ("sr_vocoder.pth", 627_861_067),
    ("vae.pth", 1_655_458_345),
];
const WEIGHTS_BYTES: u64 = WEIGHT_FILES[0].1 + WEIGHT_FILES[1].1 + WEIGHT_FILES[2].1;
const WEIGHTS_REPO: &str = "jakeoneijk/FlashSR_weights";
const FLASHSR_ARCHIVE: &str =
    "https://codeload.github.com/jakeoneijk/FlashSR_Inference/tar.gz/refs/heads/main";

/// Downloads the inference source tree and unpacks it without its top-level directory.
///
/// The package is not installable from pip: its `setup.py` uses `find_packages`, which
/// skips every directory without an `__init__.py`, and `TorchJaekwon/Model` is one of
/// them. A pip install therefore produces an interpreter that fails on
/// `from TorchJaekwon.Model... import ...` — found on a clean install, 2026-11-12. The
/// source tree on `sys.path` has no such problem, and it is what the dev checkout uses.
const FLASHSR_SCRIPT: &str = r#"
import io, sys, tarfile, urllib.request
from pathlib import Path

url, dest, marker = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
with urllib.request.urlopen(url, timeout=120) as response:
    payload = response.read()
dest.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
    for member in archive.getmembers():
        parts = member.name.split("/", 1)
        if len(parts) != 2 or not parts[1]:
            continue
        member.name = parts[1]
        archive.extract(member, dest, filter="data")
if not (dest / marker).is_file():
    raise SystemExit(f"the archive did not contain {marker}")
print("extracted", dest / marker, flush=True)
"#;

/// The pip command, in one place so the exact versions are reviewable.
const PYTHON_WHEELS: &[&str] = &[
    "torch",
    "onnxruntime>=1.18",
    "onnxscript>=0.1",
    "numpy",
    "scipy>=1.10",
    "librosa>=0.9.2",
    "pyyaml",
    "tqdm",
    "huggingface_hub>=0.20",
    "torchaudio",
    "pydub",
    "psutil",
    // Upstream imports matplotlib at module load (TorchJaekwon/Util/UtilTorch.py) and
    // einops inside the UNet. Neither is obviously "inference", both are load-bearing;
    // found the hard way on a clean install, 2026-11-12.
    "matplotlib",
    "einops",
];

/// The receipt line that records which index torch came from: the only line in it that
/// depends on the environment rather than on the wheels.
const TORCH_INDEX_LINE: &str = "torch-index ";

/// Whether the receipt lists every wheel, whatever torch's source was.
fn wheels_installed(runner: &dyn Runner, paths: &Paths) -> bool {
    let Some(receipt) = runner.read(&paths.wheels_receipt()) else {
        return false;
    };
    receipt_wheels(&String::from_utf8_lossy(&receipt)) == PYTHON_WHEELS
}

/// The wheel lines of a receipt, without the index line.
///
/// Because of that line, a venv installed from PyTorch's CUDA index lists the same wheels
/// as one installed from PyPI — which is what lets `is_ready` ignore the opt-in.
fn receipt_wheels(text: &str) -> Vec<&str> {
    text.lines()
        .filter(|line| !line.starts_with(TORCH_INDEX_LINE))
        .collect()
}

/// Where the torch wheel comes from.
///
/// PyPI's torch is CPU-only on Windows and Linux; a CUDA build lives on PyTorch's own
/// index and is about 2.5 GB larger. Opting in is per machine:
///
/// ```text
/// flacify setup --cuda
/// ```
///
/// which is the same as exporting
///
/// ```text
/// FLACIFY_TORCH_INDEX=https://download.pytorch.org/whl/cu132
/// ```
///
/// The tag has to publish the torch version the pins resolve to: `cu132` is the one that
/// carries 2.15.0 (measured on Windows, 2026-11-15). The variable doubles as the signal to
/// upgrade an existing install, so setting it re-downloads torch.
///
/// macOS needs none of this: its PyPI wheel already has Metal.
fn torch_index() -> Option<String> {
    std::env::var("FLACIFY_TORCH_INDEX")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// What the wheels receipt records. Comparing this string is how a change to
/// `PYTHON_WHEELS` (or to the torch index) re-runs the step on an install that already
/// exists — without it, a missing wheel is invisible because the receipt file is still
/// there.
pub fn wheels_manifest() -> String {
    let index = torch_index().unwrap_or_default();
    format!("torch-index {index}\n{}\n", PYTHON_WHEELS.join("\n"))
}

/// Same idea for the embedded sidecar sources.
pub fn worker_manifest() -> String {
    format!("{WORKER_PY}\n{FLASHSR_ONNX_PY}")
}

/// Downloads the missing weights; each file argument is `name=bytes` and the script
/// fails when the file it got is a different size.
const WEIGHTS_SCRIPT: &str = r#"
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

repo_id, local_dir, *files = sys.argv[1:]
local = Path(local_dir)
local.mkdir(parents=True, exist_ok=True)
for spec in files:
    name, expected = spec.rsplit("=", 1)
    hf_hub_download(repo_id=repo_id, repo_type="dataset", local_dir=str(local), filename=name)
    size = (local / name).stat().st_size
    if size != int(expected):
        raise SystemExit(f"{name} is {size} bytes, expected {expected}")
    print(name, flush=True)
"#;

/// A marker written last inside a directory a step populates. Deleting the directory
/// deletes the marker, so its absence re-runs the step; no stable file name exists for
/// an installed interpreter or for pip's site-packages.
const STEP_RECEIPT: &str = ".flacify-ready";

/// The one directory the app owns, plus the paths inside it that never move.
#[derive(Clone)]
pub struct Paths {
    root: PathBuf,
}

impl Paths {
    /// Resolves the data directory from `FLACIFY_HOME`, else the platform default.
    pub fn discover() -> Result<Paths> {
        let root = match env::var_os(HOME_ENV) {
            Some(dir) if !dir.is_empty() => PathBuf::from(dir),
            _ => platform_data_dir()?,
        };
        Ok(Paths { root })
    }

    /// A root for tests, without touching the environment.
    #[cfg(test)]
    pub(crate) fn at(root: impl Into<PathBuf>) -> Paths {
        Paths { root: root.into() }
    }

    /// The data directory itself.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Where the embedded sources and the managed `uv` live.
    pub fn runtime_dir(&self) -> PathBuf {
        self.root.join("runtime")
    }

    /// The interpreter the worker runs on.
    pub fn venv_python(&self) -> PathBuf {
        let venv = self.venv_dir();
        if cfg!(windows) {
            venv.join("Scripts").join("python.exe")
        } else {
            venv.join("bin").join("python")
        }
    }

    /// Where the three `.pth` files land.
    /// Where rendered tracks are kept between runs.
    pub fn cache_dir(&self) -> PathBuf {
        self.root().join("cache")
    }

    pub fn models_dir(&self) -> PathBuf {
        self.root.join("models")
    }

    /// Where fetched audio lands, one directory per id. Outside `cache/` on purpose: the
    /// cache scan sweeps a directory it cannot read as an entry, and a download is not one.
    // The state-machine test mounts this module without the commands, so it cannot use it.
    #[cfg_attr(test, allow(dead_code))]
    pub fn downloads_dir(&self) -> PathBuf {
        self.root.join("downloads")
    }

    /// Where the app's own tools live: the `uv` it installed, and yt-dlp. On `PATH` for a
    /// child process we spawn, never in the user's environment.
    pub fn bin_dir(&self) -> PathBuf {
        self.runtime_dir().join("bin")
    }

    /// The yt-dlp this app installed, if it did.
    pub fn managed_ytdlp(&self) -> PathBuf {
        let name = if cfg!(windows) {
            "yt-dlp.exe"
        } else {
            "yt-dlp"
        };
        self.bin_dir().join(name)
    }

    /// The sidecar entry point.
    pub fn worker_script(&self) -> PathBuf {
        self.runtime_dir().join("worker.py")
    }

    /// True when every step the model needs is recorded as done and its output is on disk.
    ///
    /// Only `required` steps count: yt-dlp serves URLs, so an install without it is still
    /// a ready player for local files, and must not be sent to the setup screen.
    pub fn is_ready(&self) -> bool {
        let state = State::load(self);
        let uv = uv_on_disk(self);
        plan().iter().filter(|step| step.required).all(|step| {
            state.is_done(step.id)
                && step_outputs_exist(
                    step.id,
                    self,
                    &SystemRunner::default(),
                    uv.as_deref(),
                    false,
                )
        })
    }

    pub(crate) fn flashsr_script(&self) -> PathBuf {
        self.runtime_dir().join("flashsr_onnx.py")
    }

    pub(crate) fn state_file(&self) -> PathBuf {
        self.root.join(STATE_FILE)
    }

    pub(crate) fn python_receipt(&self) -> PathBuf {
        self.python_dir().join(STEP_RECEIPT)
    }

    pub(crate) fn wheels_receipt(&self) -> PathBuf {
        self.venv_dir().join(STEP_RECEIPT)
    }

    pub(crate) fn worker_receipt(&self) -> PathBuf {
        self.runtime_dir().join("worker.manifest")
    }

    /// The extracted FlashSR source tree, which the worker puts on `sys.path`.
    pub fn flashsr_dir(&self) -> PathBuf {
        self.runtime_dir().join("flashsr")
    }

    /// The file whose presence means the tree really unpacked.
    pub(crate) fn flashsr_sources(&self) -> PathBuf {
        self.flashsr_dir().join("FlashSR").join("FlashSR.py")
    }

    fn uv_binary(&self) -> PathBuf {
        let name = if cfg!(windows) { "uv.exe" } else { "uv" };
        self.runtime_dir().join("bin").join(name)
    }

    fn python_dir(&self) -> PathBuf {
        self.runtime_dir().join("python")
    }

    fn venv_dir(&self) -> PathBuf {
        self.root.join("venv")
    }
}

/// One provisioning step: stable id, label for the setup screen, download size.
pub struct Step {
    pub id: &'static str,
    pub label: &'static str,
    pub bytes: u64,
    /// Whether the model needs it. `is_ready` is about the model, so a tool that only a
    /// URL uses must not make a working install look incomplete — or send a local-only
    /// user to a 6 GB setup screen to fetch a 24 MB binary.
    pub required: bool,
}

/// The provisioning steps in order. Pure: callers sum `bytes` to show the total first.
pub fn plan() -> Vec<Step> {
    vec![
        Step {
            id: STEP_UV,
            label: "uv package manager",
            bytes: UV_BYTES,
            required: true,
        },
        Step {
            id: STEP_PYTHON,
            label: "Python 3.12",
            bytes: PYTHON_BYTES,
            required: true,
        },
        Step {
            id: STEP_VENV,
            label: "virtual environment",
            bytes: VENV_BYTES,
            required: true,
        },
        Step {
            id: STEP_WHEELS,
            label: "PyTorch and inference wheels",
            bytes: WHEELS_BYTES,
            required: true,
        },
        Step {
            id: STEP_FLASHSR,
            label: "FlashSR inference sources",
            bytes: 40_000_000,
            required: true,
        },
        Step {
            id: STEP_WORKER,
            label: "sidecar sources",
            bytes: 0,
            required: true,
        },
        Step {
            id: STEP_WEIGHTS,
            label: "FlashSR weights",
            bytes: WEIGHTS_BYTES,
            required: true,
        },
        // Not required by the model: only a URL uses it, and it is the one step a
        // local-only install can skip without losing anything.
        Step {
            id: STEP_YTDLP,
            label: "yt-dlp (fetching a URL)",
            bytes: YTDLP_BYTES,
            required: false,
        },
    ]
}

type ProgressSink = Arc<dyn Fn(&str, f32) + Send + Sync>;

/// Reports a step id and that step's progress in `0.0..=1.0`.
///
/// Cheap to clone: the sink is shared, which is what lets a size monitor report while a
/// blocking child runs.
#[derive(Clone, Default)]
pub struct Progress {
    report: Option<ProgressSink>,
}

impl Progress {
    pub fn new(report: impl Fn(&str, f32) + Send + Sync + 'static) -> Progress {
        Progress {
            report: Some(Arc::new(report)),
        }
    }

    fn emit(&self, step: &str, fraction: f32) {
        if let Some(report) = &self.report {
            report(step, fraction);
        }
    }
}

/// While a step's child runs, report how much of that step's target directory exists.
///
/// Parsing `uv`/`pip`/curl output would be version-fragile; a directory walk is dull but
/// it moves, and it cannot claim progress for a step that wrote nothing.
struct Monitor {
    stop: Arc<AtomicBool>,
    thread: std::thread::JoinHandle<()>,
}

impl Monitor {
    fn spawn(step: &Step, paths: &Paths, progress: Progress) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let id = step.id.to_string();
        let root = step_root(step.id, paths);
        let expected = step.bytes.max(1) as f32;
        let thread = std::thread::spawn(move || {
            while !flag.load(Ordering::Relaxed) {
                let fraction = (dir_size(&root) as f32 / expected).clamp(0.0, 1.0);
                progress.emit(&id, fraction);
                std::thread::sleep(Duration::from_millis(400));
            }
        });
        Self { stop, thread }
    }

    fn stop(self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.thread.join();
    }
}

/// The directory a step fills, for the size monitor.
fn step_root(id: &str, paths: &Paths) -> PathBuf {
    match id {
        STEP_PYTHON => paths.python_dir(),
        STEP_VENV | STEP_WHEELS => paths.venv_dir(),
        STEP_FLASHSR => paths.flashsr_dir(),
        STEP_WEIGHTS => paths.models_dir(),
        STEP_YTDLP => paths.bin_dir(),
        _ => paths.runtime_dir(),
    }
}

/// Total size of the regular files under `path`, ignoring symlinks (which can point
/// outside the tree and, on a bad day, loop).
pub(crate) fn dir_size(path: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    let mut total = 0;
    for entry in entries.flatten() {
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_dir() {
            total += dir_size(&entry.path());
        } else if kind.is_file() {
            total += entry.metadata().map(|meta| meta.len()).unwrap_or(0);
        }
    }
    total
}

/// The cancel token and the "stopped on purpose" error live in [`crate::process`], which
/// the provisioning steps share with the downloader. Re-exported because every caller of
/// this module already names them here.
pub use crate::process::{is_cancelled, Cancel, Cancelled};

/// The shared cancellation error, worded for the install: "setup cancelled" is what the
/// user asked for and what the window shows, not a bare "cancelled".
fn cancelled() -> anyhow::Error {
    anyhow::Error::new(Cancelled).context("setup cancelled")
}

/// The IO a step needs. Public only so tests can substitute a fake.
pub trait Runner {
    /// Runs `program` with `args` and fails on a non-zero exit.
    fn run(&self, program: &Path, args: &[&str]) -> Result<()>;
    /// Replaces `path` atomically, creating parent directories.
    fn write(&self, path: &Path, bytes: &[u8]) -> Result<()>;
    /// Creates `path` and any missing parents, for steps that shell out to a program
    /// which will not create the directory it writes into (curl, tar).
    fn create_dir_all(&self, path: &Path) -> Result<()>;
    /// The size of a regular file, or `None` when it is absent.
    fn size(&self, path: &Path) -> Option<u64>;
    /// The contents of a regular file, or `None` when it is unreadable.
    fn read(&self, path: &Path) -> Option<Vec<u8>>;
    /// The token a long `run` can be stopped with, when the runner has one.
    fn cancel(&self) -> Option<&Cancel> {
        None
    }
}

/// The real runner: child processes, and files written through a temp file.
#[derive(Default)]
pub(crate) struct SystemRunner {
    cancel: Cancel,
}

impl SystemRunner {
    fn new(cancel: Cancel) -> Self {
        Self { cancel }
    }
}

impl Runner for SystemRunner {
    fn run(&self, program: &Path, args: &[&str]) -> Result<()> {
        crate::process::run_streaming(program, args, Some(&self.cancel), &mut |_| {}).map(|_| ())
    }

    fn write(&self, path: &Path, bytes: &[u8]) -> Result<()> {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let temp = temp_path(path);
        {
            let mut file =
                fs::File::create(&temp).with_context(|| format!("writing {}", temp.display()))?;
            file.write_all(bytes)
                .with_context(|| format!("writing {}", temp.display()))?;
            file.sync_all()
                .with_context(|| format!("syncing {}", temp.display()))?;
        }
        fs::rename(&temp, path).with_context(|| format!("replacing {}", path.display()))
    }

    fn create_dir_all(&self, path: &Path) -> Result<()> {
        fs::create_dir_all(path).with_context(|| format!("creating {}", path.display()))
    }

    fn size(&self, path: &Path) -> Option<u64> {
        file_size(path)
    }

    fn read(&self, path: &Path) -> Option<Vec<u8>> {
        std::fs::read(path).ok()
    }

    fn cancel(&self) -> Option<&Cancel> {
        Some(&self.cancel)
    }
}

/// Runs every step that is not already done, saving progress after each one.
pub fn ensure(paths: &Paths, progress: Progress) -> Result<()> {
    ensure_with(paths, &SystemRunner::default(), progress)
}

/// [`ensure`] with a token that stops the running child and the next step.
pub fn ensure_cancellable(paths: &Paths, progress: Progress, cancel: &Cancel) -> Result<()> {
    ensure_with(paths, &SystemRunner::new(cancel.clone()), progress)
}

/// [`ensure`] with the IO injected; `Runner` is public only so tests can fake it.
#[doc(hidden)]
pub fn ensure_with(paths: &Paths, runner: &dyn Runner, progress: Progress) -> Result<()> {
    let cancel = runner.cancel();
    if cancel.is_some_and(Cancel::is_cancelled) {
        return Err(cancelled());
    }
    let mut state = State::load(paths);
    let mut uv = uv_program(paths, runner);

    for step in plan() {
        if cancel.is_some_and(Cancel::is_cancelled) {
            return Err(cancelled());
        }
        if state.is_done(step.id) && step_outputs_exist(step.id, paths, runner, uv.as_deref(), true)
        {
            continue;
        }
        progress.emit(step.id, 0.0);
        let monitor = Monitor::spawn(&step, paths, progress.clone());
        let result = run_step(step.id, paths, runner, &mut uv);
        monitor.stop();
        result.with_context(|| format!("setup step {:?} failed", step.id))?;
        state.mark_done(step.id);
        state
            .save(paths, runner)
            .with_context(|| format!("recording step {:?}", step.id))?;
        progress.emit(step.id, 1.0);
    }
    Ok(())
}

fn run_step(id: &str, paths: &Paths, runner: &dyn Runner, uv: &mut Option<PathBuf>) -> Result<()> {
    match id {
        STEP_UV => install_uv(paths, runner, uv),
        STEP_PYTHON => install_python(paths, runner, uv),
        STEP_VENV => create_venv(paths, runner, uv),
        STEP_WHEELS => install_wheels(paths, runner, uv),
        STEP_FLASHSR => install_flashsr(paths, runner, uv),
        STEP_WORKER => write_worker(paths, runner),
        STEP_WEIGHTS => download_weights(paths, runner),
        STEP_YTDLP => install_ytdlp(paths, runner),
        other => bail!("no step named {other:?}"),
    }
}

/// Uses uv from `FLACIFY_UV`, then from `PATH`, then our own copy; otherwise downloads it.
/// The 14 MB release archive stays in `runtime/`, against the 3.3 GB of weights.
fn install_uv(paths: &Paths, runner: &dyn Runner, uv: &mut Option<PathBuf>) -> Result<()> {
    if uv.is_some() {
        return Ok(());
    }
    let url = uv_release_url()?;
    let archive = paths
        .runtime_dir()
        .join(if cfg!(windows) { "uv.zip" } else { "uv.tar.gz" });
    let bin_dir = paths.runtime_dir().join("bin");
    let archive_arg = archive.to_string_lossy().into_owned();
    let bin_arg = bin_dir.to_string_lossy().into_owned();

    // curl -o does not create the directory it writes into, and tar -C fails if its
    // destination is missing; on a clean machine neither exists yet.
    runner.create_dir_all(&bin_dir)?;

    runner.run(
        Path::new("curl"),
        &[
            "-fsSL",
            "--retry",
            "3",
            "-o",
            archive_arg.as_str(),
            url.as_str(),
        ],
    )?;
    // The unix archives wrap everything in one directory; the Windows zip does not.
    let mut tar_args = vec!["-xf", archive_arg.as_str(), "-C", bin_arg.as_str()];
    if !cfg!(windows) {
        tar_args.push("--strip-components=1");
    }
    runner.run(Path::new("tar"), &tar_args)?;

    let managed = paths.uv_binary();
    match runner.size(&managed) {
        Some(0) | None => bail!("uv was not extracted to {}", managed.display()),
        Some(_) => {
            *uv = Some(managed);
            Ok(())
        }
    }
}

fn install_python(paths: &Paths, runner: &dyn Runner, uv: &Option<PathBuf>) -> Result<()> {
    let uv = require_uv(uv)?;
    let install_dir = paths.python_dir().to_string_lossy().into_owned();
    runner.run(
        uv,
        &[
            "python",
            "install",
            "3.12",
            "--install-dir",
            install_dir.as_str(),
        ],
    )?;
    runner.write(&paths.python_receipt(), b"python 3.12\n")
}

fn create_venv(paths: &Paths, runner: &dyn Runner, uv: &Option<PathBuf>) -> Result<()> {
    let uv = require_uv(uv)?;
    let venv = paths.venv_dir().to_string_lossy().into_owned();
    runner.run(uv, &["venv", "--python", "3.12", venv.as_str()])?;
    if runner.size(&paths.venv_python()).is_none() {
        bail!(
            "uv venv left no interpreter at {}",
            paths.venv_python().display()
        );
    }
    Ok(())
}

fn install_wheels(paths: &Paths, runner: &dyn Runner, uv: &Option<PathBuf>) -> Result<()> {
    let uv = require_uv(uv)?;
    let python = paths.venv_python().to_string_lossy().into_owned();
    let index = torch_index();

    let torch_args = torch_install_args(&python, index.as_deref());
    let borrowed: Vec<&str> = torch_args.iter().map(String::as_str).collect();
    runner.run(uv, &borrowed)?;

    let mut args: Vec<&str> = vec!["pip", "install", "--python", python.as_str()];
    args.extend(
        PYTHON_WHEELS
            .iter()
            .copied()
            .filter(|wheel| *wheel != "torch"),
    );
    runner.run(uv, &args)?;

    runner.write(&paths.wheels_receipt(), wheels_manifest().as_bytes())
}

/// The torch install command, split out so the upgrade pairing can be tested without a
/// venv, a network or the environment.
///
/// `uv pip install torch` with no version spec keeps whatever torch is already installed,
/// so on a machine that ran the CPU install first the index alone changes nothing (found
/// on 2026-11-15: `--index-url` reported "Would make no changes" and the venv stayed on
/// the CPU wheel). Opting in is a change of wheel, so it has to ask for an upgrade; the
/// unpinned case stays put, which is what keeps a re-run reproducible.
fn torch_install_args(python: &str, index: Option<&str>) -> Vec<String> {
    let mut args: Vec<String> = ["pip", "install", "--python", python]
        .iter()
        .map(|arg| (*arg).to_string())
        .collect();
    if index.is_some() {
        args.push("--upgrade".to_string());
    }
    args.push("torch".to_string());
    if let Some(url) = index {
        args.push("--index-url".to_string());
        args.push(url.to_string());
    }
    args
}

/// Unpack the inference source tree. See `FLASHSR_SCRIPT` for why this is a download
/// and not a pip install.
fn install_flashsr(paths: &Paths, runner: &dyn Runner, _uv: &Option<PathBuf>) -> Result<()> {
    let destinations = paths.flashsr_dir();
    runner.run(
        &paths.venv_python(),
        &[
            "-c",
            FLASHSR_SCRIPT,
            FLASHSR_ARCHIVE,
            destinations.to_string_lossy().as_ref(),
            "FlashSR/FlashSR.py",
        ],
    )
}

fn write_worker(paths: &Paths, runner: &dyn Runner) -> Result<()> {
    runner.write(&paths.worker_script(), WORKER_PY.as_bytes())?;
    runner.write(&paths.flashsr_script(), FLASHSR_ONNX_PY.as_bytes())?;
    runner.write(&paths.worker_receipt(), worker_manifest().as_bytes())
}

fn download_weights(paths: &Paths, runner: &dyn Runner) -> Result<()> {
    let models = paths.models_dir();
    let missing: Vec<(&str, u64)> = WEIGHT_FILES
        .iter()
        .copied()
        .filter(|(name, bytes)| !has_size(runner, &models.join(name), *bytes))
        .collect();

    if !missing.is_empty() {
        let python = paths.venv_python();
        let models_arg = models.to_string_lossy().into_owned();
        let specs: Vec<String> = missing
            .iter()
            .map(|(name, bytes)| format!("{name}={bytes}"))
            .collect();
        let mut args: Vec<&str> = vec!["-c", WEIGHTS_SCRIPT, WEIGHTS_REPO, models_arg.as_str()];
        args.extend(specs.iter().map(String::as_str));
        runner.run(&python, &args)?;
    }

    for (name, bytes) in WEIGHT_FILES {
        let path = models.join(name);
        match runner.size(&path) {
            Some(found) if found == bytes => {}
            Some(found) => bail!("{} is {found} bytes, expected {bytes}", path.display()),
            None => bail!("{} was not downloaded", path.display()),
        }
    }
    Ok(())
}

/// The yt-dlp release asset for this platform, from its `latest` release.
///
/// The standalone build, not the pip package: a URL then works with no Python and no
/// venv, so the 24 MB binary is the whole cost of "fetch a link". The names are the ones
/// the yt-dlp release actually publishes per platform and architecture.
fn ytdlp_release_url() -> Result<String> {
    let asset =
        ytdlp_asset().context("yt-dlp publishes no build for this platform; install it on PATH")?;
    Ok(format!(
        "https://github.com/yt-dlp/yt-dlp/releases/latest/download/{asset}"
    ))
}

fn ytdlp_asset() -> Option<&'static str> {
    if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        Some("yt-dlp.exe")
    } else if cfg!(all(target_os = "windows", target_arch = "aarch64")) {
        Some("yt-dlp_arm64.exe")
    } else if cfg!(all(target_os = "windows", target_arch = "x86")) {
        Some("yt-dlp_x86.exe")
    } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        // The plain `yt-dlp_macos` is the universal build; there is no separate arm one.
        Some("yt-dlp_macos")
    } else if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
        Some("yt-dlp_macos")
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Some("yt-dlp_linux")
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        Some("yt-dlp_linux_aarch64")
    } else if cfg!(all(target_os = "linux", target_arch = "arm")) {
        Some("yt-dlp_linux_armv7l")
    } else {
        None
    }
}

/// Download the standalone yt-dlp into the app's own `bin/`.
///
/// curl writes straight to the destination and `chmod` makes it runnable on unix; Windows
/// has no execute bit. The step is skipped when a binary is already there, because the
/// step's own existence check is what decides whether it runs at all.
fn install_ytdlp(paths: &Paths, runner: &dyn Runner) -> Result<()> {
    let url = ytdlp_release_url()?;
    let target = paths.managed_ytdlp();
    runner.create_dir_all(&paths.bin_dir())?;
    let target_arg = target.to_string_lossy().into_owned();
    runner.run(
        Path::new("curl"),
        &[
            "-fsSL",
            "--retry",
            "3",
            "-o",
            target_arg.as_str(),
            url.as_str(),
        ],
    )?;
    // `is_none_or` is 1.82; the workspace MSRV is 1.80.
    if !matches!(runner.size(&target), Some(size) if size > 0) {
        bail!("yt-dlp was not downloaded to {}", target.display());
    }
    if !cfg!(windows) {
        runner.run(Path::new("chmod"), &["755", target_arg.as_str()])?;
    }
    Ok(())
}

/// Where a usable `uv` comes from, before anything runs.
enum UvSource {
    /// `FLACIFY_UV`, trusted as given.
    Override(PathBuf),
    /// A binary on `PATH`, checked with `--version` before use.
    Path(PathBuf),
    /// The copy this module installed earlier.
    Managed(PathBuf),
}

fn uv_source(paths: &Paths) -> Option<UvSource> {
    if let Some(program) = env::var_os(UV_ENV) {
        return Some(UvSource::Override(PathBuf::from(program)));
    }
    if let Some(found) = find_on_path("uv") {
        return Some(UvSource::Path(found));
    }
    let managed = paths.uv_binary();
    if file_size(&managed).is_some_and(|size| size > 0) {
        return Some(UvSource::Managed(managed));
    }
    None
}

/// Resolves uv for `ensure`, probing a `PATH` hit once. `None` means download it.
fn uv_program(paths: &Paths, runner: &dyn Runner) -> Option<PathBuf> {
    match uv_source(paths)? {
        UvSource::Path(program) => runner.run(&program, &["--version"]).ok().map(|()| program),
        UvSource::Override(program) | UvSource::Managed(program) => Some(program),
    }
}

/// Resolves uv for `is_ready`, which must not execute anything.
fn uv_on_disk(paths: &Paths) -> Option<PathBuf> {
    match uv_source(paths)? {
        UvSource::Override(program) | UvSource::Path(program) | UvSource::Managed(program) => {
            Some(program)
        }
    }
}

fn require_uv(uv: &Option<PathBuf>) -> Result<&Path> {
    uv.as_deref().context("uv is not installed")
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    crate::process::find_on_path(name)
}

/// The release archive for this platform.
///
/// Windows publishes a `.zip` and the others a `.tar.gz`; both are unpacked with `tar`,
/// which on Windows 10 and later is bsdtar and reads zip archives. That keeps one code
/// path instead of a PowerShell branch.
fn uv_release_url() -> Result<String> {
    let triple = uv_triple()
        .context("no uv release archive for this platform; install uv on PATH or set FLACIFY_UV")?;
    let extension = if cfg!(windows) { "zip" } else { "tar.gz" };
    Ok(format!(
        "https://github.com/astral-sh/uv/releases/latest/download/uv-{triple}.{extension}"
    ))
}

fn uv_triple() -> Option<&'static str> {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Some("aarch64-apple-darwin")
    } else if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
        Some("x86_64-apple-darwin")
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        Some("aarch64-unknown-linux-gnu")
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Some("x86_64-unknown-linux-gnu")
    } else if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        Some("x86_64-pc-windows-msvc")
    } else if cfg!(all(target_os = "windows", target_arch = "aarch64")) {
        Some("aarch64-pc-windows-msvc")
    } else {
        None
    }
}

/// Whether a step's outputs are still on disk.
///
/// `as_installed` says whether the receipt must match today's environment as well as the
/// files being present. `ensure` wants that, because changing the torch index has to
/// re-run the step; [`Paths::is_ready`] does not, because the receipt records what was
/// installed, and otherwise the app would refuse to play a working install just because
/// `FLACIFY_TORCH_INDEX` was not exported again at launch.
fn step_outputs_exist(
    id: &str,
    paths: &Paths,
    runner: &dyn Runner,
    uv: Option<&Path>,
    as_installed: bool,
) -> bool {
    match id {
        STEP_UV => uv.is_some(),
        STEP_PYTHON => runner.size(&paths.python_receipt()).is_some(),
        STEP_VENV => runner.size(&paths.venv_python()).is_some(),
        STEP_WHEELS => {
            if as_installed {
                runner.read(&paths.wheels_receipt()).as_deref()
                    == Some(wheels_manifest().as_bytes())
            } else {
                wheels_installed(runner, paths)
            }
        }
        STEP_FLASHSR => runner.size(&paths.flashsr_sources()).is_some(),
        STEP_WORKER => {
            runner.size(&paths.worker_script()).is_some()
                && runner.size(&paths.flashsr_script()).is_some()
                && runner.read(&paths.worker_receipt()).as_deref()
                    == Some(worker_manifest().as_bytes())
        }
        STEP_WEIGHTS => WEIGHT_FILES
            .iter()
            .all(|(name, bytes)| has_size(runner, &paths.models_dir().join(name), *bytes)),
        STEP_YTDLP => runner
            .size(&paths.managed_ytdlp())
            .is_some_and(|size| size > 0),
        _ => false,
    }
}

fn has_size(runner: &dyn Runner, path: &Path, expected: u64) -> bool {
    runner.size(path) == Some(expected)
}

/// What `state.json` records: the finished steps, and the format version.
struct State {
    done: Vec<String>,
}

impl State {
    /// An unreadable or unknown-version file starts from empty; every step is idempotent.
    fn load(paths: &Paths) -> State {
        fs::read_to_string(paths.state_file())
            .ok()
            .and_then(|text| parse_state(&text))
            .unwrap_or(State { done: Vec::new() })
    }

    fn is_done(&self, id: &str) -> bool {
        self.done.iter().any(|done| done == id)
    }

    fn mark_done(&mut self, id: &str) {
        if !self.is_done(id) {
            self.done.push(id.to_string());
        }
    }

    fn save(&self, paths: &Paths, runner: &dyn Runner) -> Result<()> {
        runner.write(&paths.state_file(), self.to_json().as_bytes())
    }

    fn to_json(&self) -> String {
        let ids: Vec<String> = self.done.iter().map(|id| format!("\"{id}\"")).collect();
        format!(
            "{{\"version\": {STATE_VERSION}, \"done\": [{}]}}\n",
            ids.join(", ")
        )
    }
}

/// Reads the one shape [`State::to_json`] writes: a version and an array of step ids.
fn parse_state(text: &str) -> Option<State> {
    let version: u32 = number_after(text, "\"version\"")?;
    if version != STATE_VERSION {
        return None;
    }
    Some(State {
        done: strings_after(text, "\"done\"")?,
    })
}

fn number_after(text: &str, key: &str) -> Option<u32> {
    let rest = &text[text.find(key)? + key.len()..];
    let digits: String = rest
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

fn strings_after(text: &str, key: &str) -> Option<Vec<String>> {
    let rest = &text[text.find(key)? + key.len()..];
    let open = rest.find('[')?;
    let close = rest[open..].find(']')? + open;
    let mut inner = &rest[open + 1..close];
    let mut found = Vec::new();
    while let Some(start) = inner.find('"') {
        let after = &inner[start + 1..];
        let Some(end) = after.find('"') else {
            break;
        };
        found.push(after[..end].to_string());
        inner = &after[end + 1..];
    }
    Some(found)
}

fn temp_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    path.with_file_name(name)
}

fn file_size(path: &Path) -> Option<u64> {
    fs::metadata(path)
        .ok()
        .filter(fs::Metadata::is_file)
        .map(|metadata| metadata.len())
}

#[cfg(target_os = "macos")]
fn platform_data_dir() -> Result<PathBuf> {
    Ok(home_dir()?
        .join("Library")
        .join("Application Support")
        .join("flacify"))
}

#[cfg(target_os = "windows")]
fn platform_data_dir() -> Result<PathBuf> {
    let base = env::var_os("LOCALAPPDATA").context("LOCALAPPDATA is not set")?;
    Ok(PathBuf::from(base).join("flacify"))
}

#[cfg(all(unix, not(target_os = "macos")))]
fn platform_data_dir() -> Result<PathBuf> {
    let base = env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .filter(|dir| !dir.as_os_str().is_empty());
    let base = match base {
        Some(dir) => dir,
        None => home_dir()?.join(".local").join("share"),
    };
    Ok(base.join("flacify"))
}

#[cfg(not(any(unix, target_os = "windows")))]
fn platform_data_dir() -> Result<PathBuf> {
    bail!("no data directory convention for this platform; set {HOME_ENV}")
}

#[cfg(not(target_os = "windows"))]
fn home_dir() -> Result<PathBuf> {
    let home = env::var_os("HOME").context("HOME is not set; set FLACIFY_HOME")?;
    Ok(PathBuf::from(home))
}

#[cfg(test)]
mod tests {
    use super::{
        dir_size, receipt_wheels, torch_install_args, wheels_manifest, CUDA_TORCH_INDEX,
        PYTHON_WHEELS,
    };
    use std::env;
    use std::fs;
    use std::path::Path;

    #[test]
    fn given_a_receipt_from_another_index_when_read_then_the_same_wheels_are_listed() {
        let receipt = wheels_manifest();
        assert!(receipt.starts_with("torch-index "), "{receipt}");
        assert_eq!(receipt_wheels(&receipt), PYTHON_WHEELS);
    }

    #[test]
    fn given_a_receipt_without_the_index_line_when_read_then_the_wheels_still_match() {
        let receipt = format!("{}\n", PYTHON_WHEELS.join("\n"));
        assert_eq!(receipt_wheels(&receipt), PYTHON_WHEELS);
    }

    #[test]
    fn given_a_receipt_missing_a_wheel_when_read_then_the_list_is_short() {
        let receipt = format!("torch-index \n{}\n", PYTHON_WHEELS[..3].join("\n"));
        assert_ne!(receipt_wheels(&receipt), PYTHON_WHEELS);
    }

    #[test]
    fn given_no_torch_index_when_the_torch_command_is_built_then_it_does_not_upgrade() {
        let args = torch_install_args("/venv/python", None);
        assert_eq!(
            args,
            vec!["pip", "install", "--python", "/venv/python", "torch"]
        );
    }

    #[test]
    fn given_a_torch_index_when_the_torch_command_is_built_then_it_upgrades_from_that_index() {
        let args = torch_install_args("/venv/python", Some(CUDA_TORCH_INDEX));
        assert_eq!(
            args,
            vec![
                "pip",
                "install",
                "--python",
                "/venv/python",
                "--upgrade",
                "torch",
                "--index-url",
                CUDA_TORCH_INDEX,
            ]
        );
    }

    #[test]
    fn given_nested_files_when_measured_then_every_regular_byte_is_counted() {
        let root = env::temp_dir().join(format!("flacify-dirsize-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("a/b")).expect("creating the tree");
        fs::write(root.join("a/one"), b"12345").expect("writing");
        fs::write(root.join("a/b/two"), b"123").expect("writing");
        assert_eq!(dir_size(&root), 8);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn given_a_missing_directory_when_measured_then_it_is_zero() {
        assert_eq!(dir_size(Path::new("this/path/does/not/exist")), 0);
    }
}
