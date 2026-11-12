//! State machine tests for the first-run provisioning in `src/setup.rs`.
//!
//! No network, no uv and no Python: the commands are recorded by a fake runner, and the
//! files it would have produced are created for real under a unique `FLACIFY_HOME`.
//! `setup` lives in the binary, so the module is mounted here by path, as `decode` is.

use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::Result;

#[path = "../src/setup.rs"]
mod setup;

use setup::{ensure, ensure_with, plan, Paths, Progress, Runner, SystemRunner};

/// The sizes the three weight files have upstream, pinned here independently of the module.
const WEIGHT_SIZES: [(&str, u64); 3] = [
    ("student_ldm.pth", 1_033_200_174),
    ("sr_vocoder.pth", 627_861_067),
    ("vae.pth", 1_655_458_345),
];

/// `FLACIFY_HOME` and `FLACIFY_UV` are process-wide, so the tests take turns.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// A unique data directory for one test, with uv redirected away from `PATH`.
struct Home {
    root: PathBuf,
    previous_home: Option<OsString>,
    previous_uv: Option<OsString>,
    _guard: MutexGuard<'static, ()>,
}

impl Home {
    fn new(name: &str) -> Home {
        let guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let root = env::temp_dir().join(format!("flacify-setup-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("creating the temp home");

        let previous_home = env::var_os("FLACIFY_HOME");
        let previous_uv = env::var_os("FLACIFY_UV");
        env::set_var("FLACIFY_HOME", &root);
        env::set_var("FLACIFY_UV", root.join("uv-override"));

        Home {
            root,
            previous_home,
            previous_uv,
            _guard: guard,
        }
    }

    fn paths(&self) -> Paths {
        Paths::discover().expect("discovering the temp home")
    }
}

impl Drop for Home {
    fn drop(&mut self) {
        match &self.previous_home {
            Some(value) => env::set_var("FLACIFY_HOME", value),
            None => env::remove_var("FLACIFY_HOME"),
        }
        match &self.previous_uv {
            Some(value) => env::set_var("FLACIFY_UV", value),
            None => env::remove_var("FLACIFY_UV"),
        }
        let _ = fs::remove_dir_all(&self.root);
        // `_guard` drops last, after the variables are restored.
    }
}

/// Records commands and creates what a successful one would leave behind. File IO is
/// real, so `state.json` and the embedded sources land on disk as they do in the app.
struct FakeRunner {
    models: PathBuf,
    venv_python: PathBuf,
    is_venv_created: bool,
    commands: RefCell<Vec<String>>,
    downloaded: RefCell<HashMap<PathBuf, u64>>,
    fail_on: Option<&'static str>,
}

impl FakeRunner {
    fn new(paths: &Paths) -> FakeRunner {
        FakeRunner {
            models: paths.models_dir(),
            venv_python: paths.venv_python(),
            is_venv_created: true,
            commands: RefCell::new(Vec::new()),
            downloaded: RefCell::new(HashMap::new()),
            fail_on: None,
        }
    }

    fn failing(mut self, needle: &'static str) -> FakeRunner {
        self.fail_on = Some(needle);
        self
    }

    /// A `uv venv` that exits successfully but writes no interpreter.
    fn without_venv(mut self) -> FakeRunner {
        self.is_venv_created = false;
        self
    }

    fn take_commands(&self) -> Vec<String> {
        self.commands.borrow_mut().drain(..).collect()
    }
}

impl Runner for FakeRunner {
    fn run(&self, program: &Path, args: &[&str]) -> Result<()> {
        let command = format!("{} {}", program.display(), args.join(" "));
        self.commands.borrow_mut().push(command.clone());
        if let Some(needle) = self.fail_on {
            if command.contains(needle) {
                anyhow::bail!("{command} failed (fake)");
            }
        }
        if command.contains("FlashSR_weights") {
            for (name, bytes) in args.iter().filter_map(|arg| arg.split_once('=')) {
                if name.ends_with(".pth") {
                    let bytes: u64 = bytes.parse().expect("a size argument");
                    self.downloaded
                        .borrow_mut()
                        .insert(self.models.join(name), bytes);
                }
            }
        }
        if args.first() == Some(&"venv") && self.is_venv_created {
            SystemRunner.write(&self.venv_python, b"#!/bin/sh\n")?;
        }
        Ok(())
    }

    fn write(&self, path: &Path, bytes: &[u8]) -> Result<()> {
        SystemRunner.write(path, bytes)
    }

    fn size(&self, path: &Path) -> Option<u64> {
        self.downloaded
            .borrow()
            .get(path)
            .copied()
            .or_else(|| SystemRunner.size(path))
    }
}

fn write_file(path: &Path, bytes: &[u8]) {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).expect("creating a parent directory");
    }
    fs::write(path, bytes).expect("writing a file");
}

/// A file of `len` bytes without the blocks to back it (APFS and ext4 both spare them).
fn sparse_file(path: &Path, len: u64) {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).expect("creating a parent directory");
    }
    let file = fs::File::create(path).expect("creating a sparse file");
    file.set_len(len).expect("setting a sparse length");
}

fn write_state(paths: &Paths, done: &[&str]) {
    let ids: Vec<String> = done.iter().map(|id| format!("\"{id}\"")).collect();
    write_file(
        &paths.state_file(),
        format!("{{\"version\": 1, \"done\": [{}]}}\n", ids.join(", ")).as_bytes(),
    );
}

/// Every output the six steps produce, as the app would leave them.
fn create_complete_install(paths: &Paths) {
    write_file(&paths.python_receipt(), b"python 3.12\n");
    write_file(&paths.venv_python(), b"#!/bin/sh\n");
    write_file(&paths.wheels_receipt(), b"wheels\n");
    write_file(&paths.flashsr_receipt(), b"flashsr\n");
    write_file(&paths.worker_script(), b"print('worker')\n");
    write_file(&paths.flashsr_script(), b"print('flashsr')\n");
    for (name, bytes) in WEIGHT_SIZES {
        sparse_file(&paths.models_dir().join(name), bytes);
    }
}

fn position(commands: &[String], needle: &str) -> usize {
    commands
        .iter()
        .position(|command| command.contains(needle))
        .unwrap_or_else(|| panic!("no {needle:?} in {commands:#?}"))
}

#[test]
fn given_an_empty_home_when_ensure_runs_then_every_step_executes_in_order() {
    let home = Home::new("fresh");
    let paths = home.paths();
    assert_eq!(paths.root(), home.root.as_path());
    let runner = FakeRunner::new(&paths);
    let events: Arc<Mutex<Vec<(String, f32)>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&events);
    let progress = Progress::new(move |step: &str, fraction: f32| {
        sink.lock()
            .expect("the progress sink")
            .push((step.to_string(), fraction));
    });

    ensure_with(&paths, &runner, progress).expect("a fresh install");

    let commands = runner.take_commands();
    assert_eq!(
        commands.len(),
        5,
        "one command per networked step: {commands:#?}"
    );
    assert!(position(&commands, "python install") < position(&commands, "venv --python"));
    assert!(position(&commands, "venv --python") < position(&commands, "pip install"));
    assert!(position(&commands, "pip install") < position(&commands, "FlashSR_weights"));

    let state = fs::read_to_string(paths.state_file()).expect("reading the state file");
    let ids: Vec<&str> = plan().iter().map(|step| step.id).collect();
    let positions: Vec<usize> = ids
        .iter()
        .map(|id| {
            state
                .find(&format!("\"{id}\""))
                .unwrap_or_else(|| panic!("{state} lacks {id}"))
        })
        .collect();
    assert!(
        positions.windows(2).all(|pair| pair[0] < pair[1]),
        "the state records the steps out of order: {state}"
    );

    assert_eq!(
        fs::read_to_string(paths.worker_script()).expect("the worker source"),
        include_str!("../../../tools/worker.py")
    );
    assert_eq!(
        fs::read_to_string(paths.flashsr_script()).expect("the FlashSR wrapper"),
        include_str!("../../../tools/flashsr_onnx.py")
    );

    let events = events.lock().expect("the progress sink");
    assert_eq!(events.len(), 2 * ids.len(), "{events:?}");
    for id in ids {
        let reported: Vec<f32> = events
            .iter()
            .filter(|(step, _)| step == id)
            .map(|(_, fraction)| *fraction)
            .collect();
        assert_eq!(reported, vec![0.0, 1.0], "{id} reported {reported:?}");
    }
}

#[test]
fn given_a_finished_home_when_ensure_runs_again_then_no_step_executes() {
    let home = Home::new("second-run");
    let paths = home.paths();
    let runner = FakeRunner::new(&paths);
    ensure_with(&paths, &runner, Progress::default()).expect("the first install");
    runner.take_commands();

    ensure_with(&paths, &runner, Progress::default()).expect("the second install");

    assert!(
        runner.take_commands().is_empty(),
        "a finished home ran a step again"
    );
}

#[test]
fn given_a_state_missing_some_ids_when_ensure_runs_then_only_those_steps_execute() {
    let home = Home::new("partial");
    let paths = home.paths();
    write_state(&paths, &["uv", "python"]);
    write_file(&paths.python_receipt(), b"python 3.12\n");
    let runner = FakeRunner::new(&paths);

    ensure_with(&paths, &runner, Progress::default()).expect("resuming a partial install");

    let commands = runner.take_commands();
    assert!(
        !commands
            .iter()
            .any(|command| command.contains("python install")),
        "{commands:#?}"
    );
    assert!(commands
        .iter()
        .any(|command| command.contains("venv --python")));
    assert!(commands
        .iter()
        .any(|command| command.contains("pip install")));
    assert!(commands
        .iter()
        .any(|command| command.contains("FlashSR_weights")));

    let state = fs::read_to_string(paths.state_file()).expect("reading the state file");
    for step in plan() {
        assert!(state.contains(&format!("\"{}\"", step.id)), "{state}");
    }
}

#[test]
fn given_a_failing_step_when_ensure_runs_then_that_step_is_not_recorded() {
    let home = Home::new("failure");
    let paths = home.paths();
    let runner = FakeRunner::new(&paths).failing("pip install");

    let error = ensure_with(&paths, &runner, Progress::default()).expect_err("the pip failure");
    let message = format!("{error:#}");
    assert!(
        message.contains("wheels"),
        "the step id is missing: {message}"
    );
    assert!(
        message.contains("pip install"),
        "the command is missing: {message}"
    );

    let state = fs::read_to_string(paths.state_file()).expect("reading the state file");
    assert!(state.contains("\"uv\""), "{state}");
    assert!(state.contains("\"python\""), "{state}");
    assert!(state.contains("\"venv\""), "{state}");
    assert!(!state.contains("\"wheels\""), "{state}");
    assert!(!state.contains("\"worker\""), "{state}");
    assert!(!state.contains("\"weights\""), "{state}");
}

#[test]
fn given_all_weights_present_when_the_weights_step_runs_then_nothing_downloads() {
    let home = Home::new("weights-present");
    let paths = home.paths();
    create_complete_install(&paths);
    write_state(
        &paths,
        &["uv", "python", "venv", "wheels", "flashsr", "worker"],
    );
    let runner = FakeRunner::new(&paths);

    ensure_with(&paths, &runner, Progress::default()).expect("recording a finished download");

    assert!(
        runner.take_commands().is_empty(),
        "the weights were downloaded again"
    );
    let state = fs::read_to_string(paths.state_file()).expect("reading the state file");
    assert!(state.contains("\"weights\""), "{state}");
}

#[test]
fn given_a_foreign_or_unreadable_state_when_ensure_runs_then_the_steps_execute_again() {
    let home = Home::new("foreign-state");
    let paths = home.paths();
    create_complete_install(&paths);
    let ids: Vec<&str> = plan().iter().map(|step| step.id).collect();
    let done: Vec<String> = ids.iter().map(|id| format!("\"{id}\"")).collect();

    write_file(
        &paths.state_file(),
        format!("{{\"version\": 99, \"done\": [{}]}}\n", done.join(", ")).as_bytes(),
    );
    let runner = FakeRunner::new(&paths);
    ensure_with(&paths, &runner, Progress::default()).expect("a run over a newer state file");
    let commands = runner.take_commands();
    assert!(
        commands
            .iter()
            .any(|command| command.contains("python install")),
        "a state from another version was trusted: {commands:#?}"
    );

    write_file(&paths.state_file(), b"not json at all\n");
    let runner = FakeRunner::new(&paths);
    ensure_with(&paths, &runner, Progress::default()).expect("a run over a garbage state file");
    let commands = runner.take_commands();
    assert!(
        commands
            .iter()
            .any(|command| command.contains("python install")),
        "a garbage state file was trusted: {commands:#?}"
    );

    let state = fs::read_to_string(paths.state_file()).expect("reading the state file");
    assert!(state.contains("\"version\": 1"), "{state}");
}

#[test]
fn given_a_venv_without_an_interpreter_when_ensure_runs_then_the_error_names_the_path() {
    let home = Home::new("no-interpreter");
    let paths = home.paths();
    let runner = FakeRunner::new(&paths).without_venv();

    let error =
        ensure_with(&paths, &runner, Progress::default()).expect_err("the missing interpreter");
    let message = format!("{error:#}");

    assert!(
        message.contains("venv"),
        "the step id is missing: {message}"
    );
    assert!(
        message.contains("bin/python"),
        "the path is missing: {message}"
    );
}

#[test]
fn given_a_done_state_when_an_output_is_missing_then_only_that_step_runs() {
    let home = Home::new("missing-output");
    let paths = home.paths();
    let ids: Vec<&str> = plan().iter().map(|step| step.id).collect();
    create_complete_install(&paths);
    write_state(&paths, &ids);
    fs::remove_file(paths.python_receipt()).expect("removing the interpreter receipt");
    let runner = FakeRunner::new(&paths);

    ensure_with(&paths, &runner, Progress::default()).expect("repairing the install");

    let commands = runner.take_commands();
    assert_eq!(commands.len(), 1, "{commands:#?}");
    assert!(commands[0].contains("python install"), "{commands:#?}");
    assert!(
        paths.python_receipt().exists(),
        "the receipt was not restored"
    );
}

#[test]
fn given_a_done_state_when_a_file_is_missing_then_is_ready_is_false() {
    let home = Home::new("ready");
    let paths = home.paths();
    let ids: Vec<&str> = plan().iter().map(|step| step.id).collect();
    create_complete_install(&paths);
    write_state(&paths, &ids);

    assert!(paths.is_ready(), "a complete install should be ready");

    fs::remove_file(paths.worker_script()).expect("removing the worker source");
    assert!(!paths.is_ready(), "a missing worker source is not ready");

    write_file(&paths.worker_script(), b"print('worker')\n");
    let vocoder = paths.models_dir().join("sr_vocoder.pth");
    sparse_file(&vocoder, WEIGHT_SIZES[1].1 - 1);
    assert!(!paths.is_ready(), "a truncated weight file is not ready");
}

#[test]
fn given_the_plan_when_summed_then_it_reports_the_size_the_screen_shows() {
    let steps = plan();
    let ids: Vec<&str> = steps.iter().map(|step| step.id).collect();
    assert_eq!(
        ids,
        ["uv", "python", "venv", "wheels", "flashsr", "worker", "weights"]
    );
    assert!(steps.iter().all(|step| !step.label.is_empty()));

    let weights = steps
        .iter()
        .find(|step| step.id == "weights")
        .expect("a weights step");
    let weight_total: u64 = WEIGHT_SIZES.iter().map(|(_, bytes)| bytes).sum();
    assert_eq!(weights.bytes, weight_total);

    let worker = steps
        .iter()
        .find(|step| step.id == "worker")
        .expect("a worker step");
    assert_eq!(worker.bytes, 0, "the sources are embedded, not downloaded");

    let total: u64 = steps.iter().map(|step| step.bytes).sum();
    assert_eq!(
        total,
        20_000_000 + 30_000_000 + 1_000_000 + 2_500_000_000 + 3_316_519_586,
        "the setup screen prints this total"
    );
}

#[cfg(unix)]
#[test]
fn given_a_command_that_fails_when_run_then_the_error_carries_its_output_or_exit() {
    let with_output = SystemRunner
        .run(Path::new("sh"), &["-c", "echo boom >&2; exit 3"])
        .expect_err("a command that exits 3");
    let message = format!("{with_output:#}");
    assert!(
        message.contains("sh -c"),
        "the command is missing: {message}"
    );
    assert!(message.contains("boom"), "the output is missing: {message}");

    let without_output = SystemRunner
        .run(Path::new("sh"), &["-c", "exit 4"])
        .expect_err("a command that exits 4");
    let message = format!("{without_output:#}");
    assert!(
        message.contains("exit status"),
        "the status is missing: {message}"
    );
}

#[test]
fn given_a_missing_uv_when_ensure_runs_then_the_error_names_the_step() {
    let home = Home::new("no-uv");
    let paths = home.paths();

    // The real runner, with FLACIFY_UV pointing at a path that does not exist: the uv
    // step trusts the override and the python step fails at exec, before any network.
    let error = ensure(&paths, Progress::default()).expect_err("the missing uv binary");
    let message = format!("{error:#}");

    assert!(
        message.contains("python"),
        "the step id is missing: {message}"
    );
    assert!(
        message.contains("uv-override"),
        "the path is missing: {message}"
    );
}
