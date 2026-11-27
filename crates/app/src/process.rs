//! Running another program: one child at a time, its output read as it arrives.
//!
//! Two callers need this. `setup` shells out to `uv`, `pip`, `curl` and `tar` and wants
//! only "did it work, and what did it say when it did not"; the YouTube source shells out
//! to `yt-dlp` and reads its progress lines to know how far a download has got. Both need
//! the same three things, so they live here rather than twice: a child whose pipes cannot
//! deadlock the wait, a [`Cancel`] token that kills it, and an error that carries the tail
//! of what it printed. Nothing here knows what the program is for.

use std::env;
use std::fs;
use std::io::{BufRead, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{bail, Context, Result};

/// The one child a run owns, shared with its watchdog. `Child` is not `Sync`, so the
/// handle is; killing and waiting both go through it, which keeps the kill off `PATH` and
/// off a pid that may have been recycled.
type SharedChild = Arc<Mutex<Child>>;

/// How often the reload of a cancelled child is checked, and how long a killed child is
/// waited for before its pipes are abandoned.
const POLL: Duration = Duration::from_millis(20);
const REAP: Duration = Duration::from_secs(5);

/// How many lines of a failed child's output an error carries. Enough to name the cause,
/// short enough that a `pip` traceback does not bury it.
const ERROR_LINES: usize = 20;

/// A token a long child can be stopped with. Cloning one shares the same flag, which is
/// how a window's cancel button reaches a thread that is blocked reading a pipe.
#[derive(Clone, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// The work was stopped on purpose. Not a failure: the unfinished step can run again.
#[derive(Debug)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("cancelled")
    }
}

impl std::error::Error for Cancelled {}

/// Whether an error means the caller cancelled rather than the program failing.
pub fn is_cancelled(error: &anyhow::Error) -> bool {
    error.downcast_ref::<Cancelled>().is_some()
}

/// What a finished child left behind.
#[derive(Debug)]
pub struct Finished {
    /// Every line it wrote to stdout, in order, without the newlines.
    pub stdout: Vec<String>,
    /// Its stderr, whole, for a caller that wants more than the tail an error carries.
    // The setup test mounts this module for its runner alone, never reads a warning.
    #[cfg_attr(test, allow(dead_code))]
    pub stderr: String,
}

/// Runs `program` with `args` to completion, handing each line of its stdout to `on_line`
/// as it arrives and failing on a non-zero exit.
///
/// stderr is drained on its own thread, so a child that prints past the pipe buffer cannot
/// deadlock the read. stdout is read here, on the caller's thread, which is what lets
/// `on_line` borrow from the caller. A watchdog kills the child when `cancel` flips, so a
/// child that prints nothing and never exits still ends.
pub fn run_streaming(
    program: &Path,
    args: &[&str],
    cancel: Option<&Cancel>,
    on_line: &mut dyn FnMut(&str),
) -> Result<Finished> {
    let command = command_line(program, args);
    if cancel.is_some_and(Cancel::is_cancelled) {
        return Err(anyhow::Error::new(Cancelled));
    }
    let mut child = std::process::Command::new(program)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("running {command}"))?;
    let mut pipe = std::io::BufReader::new(
        child
            .stdout
            .take()
            .with_context(|| format!("{command} has no stdout"))?,
    );
    let stderr = child.stderr.take().map(drain);
    // The watchdog kills through this handle rather than by running `kill`, which would
    // need `PATH` and is exactly what a test emptying `PATH` takes away.
    let shared = Arc::new(Mutex::new(child));
    let mut watching = cancel.map(|token| Watchdog::new(Arc::clone(&shared), token));

    let mut lines = Vec::new();
    let mut raw = Vec::new();
    let mut read_failed = None;
    loop {
        raw.clear();
        // `read_until`, not `read_line`: one byte of non-UTF-8 (a path in a non-UTF-8
        // encoding is a realistic yt-dlp output) would make `read_line` return
        // `InvalidData` and abort a run that would otherwise succeed.
        match pipe.read_until(b'\n', &mut raw) {
            Ok(0) => break,
            Ok(_) => {
                let text = String::from_utf8_lossy(&raw)
                    .trim_end_matches(['\r', '\n'])
                    .to_string();
                on_line(&text);
                lines.push(text);
            }
            Err(error) => {
                read_failed = Some(error);
                break;
            }
        }
    }
    // Signalled on every path, not just the happy one: a reader error must not leave the
    // watchdog to time out on its own.
    if let Some(watching) = watching.as_mut() {
        watching.stop();
    }
    let status = match read_failed {
        // The watchdog kills a child whose pipe we are trying to read; a reset is that.
        Some(error) => {
            reap(&shared);
            let _ = finish(stderr);
            if cancel.is_some_and(Cancel::is_cancelled) {
                return Err(anyhow::Error::new(Cancelled));
            }
            return Err(error).with_context(|| format!("reading {command}'s output"));
        }
        None => wait_for(&shared, cancel)?,
    };
    let stderr = finish(stderr);
    if status.success() {
        return Ok(Finished {
            stdout: lines,
            stderr,
        });
    }
    let detail = last_lines(&stderr, ERROR_LINES);
    if detail.trim().is_empty() {
        bail!("{command} failed ({status})");
    }
    bail!("{command} failed: {detail}")
}

/// Waits for the child, but reports cancellation rather than a status when that is why it
/// died: a killed child's exit code says nothing about the program.
fn wait_for(shared: &SharedChild, cancel: Option<&Cancel>) -> Result<ExitStatus> {
    loop {
        if cancel.is_some_and(Cancel::is_cancelled) {
            reap(shared);
            return Err(anyhow::Error::new(Cancelled));
        }
        let mut child = shared.lock().expect("the child handle");
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) => {
                drop(child);
                std::thread::sleep(POLL);
            }
            Err(error) => return Err(error).context("waiting for the child"),
        }
    }
}

/// Kills the child and waits a bounded time for it, so a cancel cannot leak a process.
fn reap(shared: &SharedChild) {
    let deadline = std::time::Instant::now() + REAP;
    {
        let mut child = shared.lock().expect("the child handle");
        let _ = child.kill();
    }
    while std::time::Instant::now() < deadline {
        let mut child = shared.lock().expect("the child handle");
        match child.try_wait() {
            Ok(Some(_)) | Err(_) => return,
            Ok(None) => {
                drop(child);
                std::thread::sleep(POLL);
            }
        }
    }
}

/// A thread that kills the child the moment the token flips. Without it a child that
/// prints nothing would block the caller's read forever and the cancel would go unseen.
struct Watchdog {
    done: std::sync::mpsc::Sender<()>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Watchdog {
    fn new(child: SharedChild, cancel: &Cancel) -> Self {
        // The handle, not the pid: `Child::kill` is the syscall, so it needs no `PATH` and
        // cannot hit a recycled pid. A reaped child makes it a harmless `InvalidInput`.
        let token = cancel.clone();
        let (done, waiting) = std::sync::mpsc::channel::<()>();
        let thread = std::thread::spawn(move || loop {
            if waiting.recv_timeout(POLL).is_ok() {
                return;
            }
            if token.is_cancelled() {
                if let Ok(mut child) = child.lock() {
                    let _ = child.kill();
                }
                return;
            }
        });
        Self {
            done,
            thread: Some(thread),
        }
    }

    /// Signal the thread and wait for it, when the caller is on the happy path and can
    /// afford to block. Takes the handle out so `Drop` has nothing left to join.
    fn stop(&mut self) {
        let _ = self.done.send(());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for Watchdog {
    /// The early returns above drop the watchdog without stopping it. A `Sender` dropped
    /// on its own makes `recv_timeout` return `Err` at once, which would leave the thread
    /// spinning for the life of the process, so the signal has to be sent here too. The
    /// thread is not joined: it exits on its own next poll, and blocking a `Drop` on a
    /// path that was already reporting an error would be the wrong trade.
    fn drop(&mut self) {
        let _ = self.done.send(());
    }
}

/// Read a child's pipe to the end on its own thread, so the child never blocks on a full
/// pipe while the parent polls for exit or cancellation.
fn drain(mut stream: impl Read + Send + 'static) -> std::thread::JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let mut buffer = Vec::new();
        let _ = stream.read_to_end(&mut buffer);
        buffer
    })
}

/// Joins a draining thread and decodes what it read, without failing the run over it.
fn finish(reader: Option<std::thread::JoinHandle<Vec<u8>>>) -> String {
    let bytes = reader
        .and_then(|thread| thread.join().ok())
        .unwrap_or_default();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// The command as it was run, for an error a person can copy into a terminal.
fn command_line(program: &Path, args: &[&str]) -> String {
    format!("{} {}", program.display(), args.join(" "))
}

/// The last `count` lines of `text`, which is what a failed child's message carries.
fn last_lines(text: &str, count: usize) -> String {
    let mut lines: Vec<&str> = text.lines().collect();
    let start = lines.len().saturating_sub(count);
    lines.drain(..start);
    lines.join("\n")
}

/// Where `name` is on `PATH`, honouring the platform's shim extensions.
///
/// Windows needs `PATHEXT`, not just `.exe`: a `yt-dlp` installed by pipx or scoop is a
/// `.cmd` or `.bat` shim, and testing only for `.exe` reports a present tool as missing.
pub fn find_on_path(name: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    let names = candidate_names(name);
    env::split_paths(&path)
        .flat_map(|dir| names.iter().map(move |candidate| dir.join(candidate)))
        .find(|candidate| is_executable(candidate))
}

/// The file names to look for: the bare one on unix, and one per `PATHEXT` suffix on
/// Windows, which keeps the bare name so an extensionless binary still resolves.
fn candidate_names(name: &str) -> Vec<String> {
    if !cfg!(windows) {
        return vec![name.to_string()];
    }
    let mut names = vec![name.to_string()];
    let extensions = env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_string());
    for extension in extensions.split(';').filter(|part| !part.is_empty()) {
        names.push(format!("{name}{extension}"));
        names.push(format!("{name}{extension}").to_lowercase());
    }
    names
}

/// Whether `path` is a file this platform would run.
pub fn is_executable(path: &Path) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute, because a sibling test in another file empties `PATH` to prove a step
    /// re-downloads `uv`; these tests must not depend on finding a shell.
    fn shell() -> PathBuf {
        let path = Path::new("/bin/sh");
        if path.exists() {
            return path.to_path_buf();
        }
        Path::new("/usr/bin/sh").to_path_buf()
    }

    // A POSIX shell: this test needs `/bin/sh -c`, so it does not run on Windows.
    #[cfg(unix)]
    #[test]
    fn given_a_child_that_prints_when_run_then_every_line_reaches_the_sink() {
        let mut seen = Vec::new();
        let finished = run_streaming(
            &shell(),
            &["-c", "printf 'one\\ntwo\\n'"],
            None,
            &mut |line| seen.push(line.to_string()),
        )
        .expect("sh prints two lines");
        assert_eq!(seen, vec!["one".to_string(), "two".to_string()]);
        assert_eq!(finished.stdout, seen);
    }

    // A POSIX shell: this test needs `/bin/sh -c`, so it does not run on Windows.
    #[cfg(unix)]
    #[test]
    fn given_a_child_that_fails_when_run_then_the_error_carries_its_last_lines() {
        let error = run_streaming(
            &shell(),
            &["-c", "echo 'the reason' >&2; exit 3"],
            None,
            &mut |_| {},
        )
        .expect_err("a non-zero exit is a failure");
        assert!(format!("{error:#}").contains("the reason"), "{error:#}");
    }

    // A POSIX shell: this test needs `/bin/sh -c`, so it does not run on Windows.
    #[cfg(unix)]
    #[test]
    fn given_a_cancelled_token_when_run_then_no_child_is_spawned() {
        let cancel = Cancel::default();
        cancel.cancel();
        let error = run_streaming(&shell(), &["-c", "true"], Some(&cancel), &mut |_| {})
            .expect_err("a cancelled run does not start");
        assert!(is_cancelled(&error), "{error:#}");
    }

    // A POSIX shell: this test needs `/bin/sh -c`, so it does not run on Windows.
    #[cfg(unix)]
    #[test]
    fn given_a_child_that_never_prints_when_cancelled_then_it_is_killed_not_hung() {
        let cancel = Cancel::default();
        let token = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            token.cancel();
        });
        let started = std::time::Instant::now();
        let error = run_streaming(&shell(), &["-c", "sleep 30"], Some(&cancel), &mut |_| {})
            .expect_err("a cancelled child does not return success");
        // `sleep 30` would end on its own, so the error alone proves nothing: the
        // watchdog firing is what makes this fast, and that is what is asserted.
        assert!(is_cancelled(&error), "{error:#}");
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "took {:?}, so the child was not killed",
            started.elapsed()
        );
    }

    #[test]
    fn given_a_program_that_is_not_there_when_run_then_the_error_names_the_command() {
        let error = run_streaming(
            Path::new("/nonexistent/not-a-real-program"),
            &[],
            None,
            &mut |_| {},
        )
        .expect_err("no such program");
        assert!(
            format!("{error:#}").contains("not-a-real-program"),
            "{error:#}"
        );
    }

    #[test]
    fn given_a_binary_on_the_path_when_looked_up_then_its_full_path_comes_back() {
        // Self-contained: a sibling test file empties `PATH` on purpose, so this one sets
        // its own and restores the previous value before it asserts anything.
        let dir = std::env::temp_dir().join(format!("flacify-path-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("a temp dir");
        let program = dir.join(if cfg!(windows) { "probe.exe" } else { "probe" });
        fs::write(&program, b"#!/bin/sh\n").expect("a program");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = fs::metadata(&program).expect("metadata").permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&program, permissions).expect("chmod");
        }
        let previous = env::var_os("PATH");
        env::set_var("PATH", &dir);

        let found = find_on_path("probe");
        let missing = find_on_path("not-a-real-program-anywhere");

        env::remove_var("PATH");
        if let Some(previous) = previous {
            env::set_var("PATH", previous);
        }
        let _ = fs::remove_dir_all(&dir);
        assert_eq!(found, Some(program));
        assert!(missing.is_none());
    }

    #[test]
    fn given_only_the_last_lines_wanted_then_the_earlier_ones_are_dropped() {
        assert_eq!(last_lines("a\nb\nc", 2), "b\nc");
        assert_eq!(last_lines("a", 5), "a");
        assert_eq!(last_lines("", 2), "");
    }
}
