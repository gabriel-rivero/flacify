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

const HOME_ENV: &str = "FLACIFY_HOME";
const UV_ENV: &str = "FLACIFY_UV";

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

/// Where the torch wheel comes from.
///
/// PyPI's torch is CPU-only on Windows and Linux; a CUDA build lives on PyTorch's own
/// index and is about 2.5 GB larger. Opting in is per machine:
///
/// ```text
/// FLACIFY_TORCH_INDEX=https://download.pytorch.org/whl/cu124
/// ```
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

    /// The sidecar entry point.
    pub fn worker_script(&self) -> PathBuf {
        self.runtime_dir().join("worker.py")
    }

    /// True when every step is recorded as done and its output is still on disk.
    pub fn is_ready(&self) -> bool {
        let state = State::load(self);
        let uv = uv_on_disk(self);
        plan().iter().all(|step| {
            state.is_done(step.id)
                && step_outputs_exist(step.id, self, &SystemRunner, uv.as_deref())
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
}

/// The provisioning steps in order. Pure: callers sum `bytes` to show the total first.
pub fn plan() -> Vec<Step> {
    vec![
        Step {
            id: STEP_UV,
            label: "uv package manager",
            bytes: UV_BYTES,
        },
        Step {
            id: STEP_PYTHON,
            label: "Python 3.12",
            bytes: PYTHON_BYTES,
        },
        Step {
            id: STEP_VENV,
            label: "virtual environment",
            bytes: VENV_BYTES,
        },
        Step {
            id: STEP_WHEELS,
            label: "PyTorch and inference wheels",
            bytes: WHEELS_BYTES,
        },
        Step {
            id: STEP_FLASHSR,
            label: "FlashSR inference sources",
            bytes: 40_000_000,
        },
        Step {
            id: STEP_WORKER,
            label: "sidecar sources",
            bytes: 0,
        },
        Step {
            id: STEP_WEIGHTS,
            label: "FlashSR weights",
            bytes: WEIGHTS_BYTES,
        },
    ]
}

type ProgressSink = Box<dyn Fn(&str, f32) + Send>;

/// Reports a step id and that step's progress in `0.0..=1.0`.
#[derive(Default)]
pub struct Progress {
    report: Option<ProgressSink>,
}

impl Progress {
    pub fn new(report: impl Fn(&str, f32) + Send + 'static) -> Progress {
        Progress {
            report: Some(Box::new(report)),
        }
    }

    fn emit(&self, step: &str, fraction: f32) {
        if let Some(report) = &self.report {
            report(step, fraction);
        }
    }
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
}

/// The real runner: child processes, and files written through a temp file.
pub(crate) struct SystemRunner;

impl Runner for SystemRunner {
    fn run(&self, program: &Path, args: &[&str]) -> Result<()> {
        let command = command_line(program, args);
        let output = std::process::Command::new(program)
            .args(args)
            .output()
            .with_context(|| format!("running {command}"))?;
        if output.status.success() {
            return Ok(());
        }
        let detail = last_lines(&String::from_utf8_lossy(&output.stderr), 20);
        if detail.trim().is_empty() {
            bail!("{command} failed ({})", output.status);
        }
        bail!("{command} failed: {detail}")
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
}

/// Runs every step that is not already done, saving progress after each one.
pub fn ensure(paths: &Paths, progress: Progress) -> Result<()> {
    ensure_with(paths, &SystemRunner, progress)
}

/// [`ensure`] with the IO injected; `Runner` is public only so tests can fake it.
#[doc(hidden)]
pub fn ensure_with(paths: &Paths, runner: &dyn Runner, progress: Progress) -> Result<()> {
    let mut state = State::load(paths);
    let mut uv = uv_program(paths, runner);

    for step in plan() {
        if state.is_done(step.id) && step_outputs_exist(step.id, paths, runner, uv.as_deref()) {
            continue;
        }
        progress.emit(step.id, 0.0);
        run_step(step.id, paths, runner, &mut uv)
            .with_context(|| format!("setup step {:?} failed", step.id))?;
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

    // torch goes first and alone, because --index-url applies to the whole command and
    // PyTorch's index does not carry the rest of these wheels.
    let mut torch_args: Vec<&str> = vec!["pip", "install", "--python", python.as_str(), "torch"];
    if let Some(url) = index.as_deref() {
        torch_args.push("--index-url");
        torch_args.push(url);
    }
    runner.run(uv, &torch_args)?;

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
    let path = env::var_os("PATH")?;
    let file_name = if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    };
    env::split_paths(&path)
        .map(|dir| dir.join(&file_name))
        .find(|candidate| is_executable(candidate))
}

fn is_executable(path: &Path) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    // Written this way so both platform branches are one expression: the `if`/`return`
    // shape left a `bool` literal behind on Windows and clippy 1.100 called it.
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    let executable = {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o111 != 0
    };
    #[cfg(not(unix))]
    let executable = true;
    executable
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

fn step_outputs_exist(id: &str, paths: &Paths, runner: &dyn Runner, uv: Option<&Path>) -> bool {
    match id {
        STEP_UV => uv.is_some(),
        STEP_PYTHON => runner.size(&paths.python_receipt()).is_some(),
        STEP_VENV => runner.size(&paths.venv_python()).is_some(),
        STEP_WHEELS => {
            runner.read(&paths.wheels_receipt()).as_deref() == Some(wheels_manifest().as_bytes())
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

fn command_line(program: &Path, args: &[&str]) -> String {
    format!("{} {}", program.display(), args.join(" "))
}

fn last_lines(text: &str, count: usize) -> String {
    let mut lines: Vec<&str> = text.lines().collect();
    let start = lines.len().saturating_sub(count);
    lines.drain(..start);
    lines.join("\n")
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
