//! `flacify batch`: a folder rendered one file at a time.
//!
//! A plan is the folder's audio files in path order, each with what the cache held for
//! it. The runner walks that list, renders one file at a time, and records a failure
//! instead of returning it, so one bad file does not cost the rest of the folder.
//!
//! Nothing here opens a device or decodes a whole track: a file goes through the source
//! tier and the segmented renderer, the same path a play uses to fill an entry. An
//! interrupted render leaves its segments, so a rerun continues where the last one died
//! without a resume mechanism of its own.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use flacify_infer::WorkerSession;

use crate::cache::{self, Cache, EntrySpec, Fields, State, CACHE_VERSION};
use crate::render;
use crate::source;

/// The one file in an entry that is not audio. `cache.rs` owns the format; this module
/// names it only to read an entry without going through `Cache`, which would rewrite its
/// `last_used` (see [`entry_state`]).
const MANIFEST: &str = "manifest.json";

/// The containers this build's symphonia decodes, by extension. A file with one of these
/// stays a candidate even when it cannot be decoded, so a truncated or corrupt file is
/// reported by the render that fails on it instead of vanishing from the walk.
const AUDIO_EXTENSIONS: &[&str] = &["aac", "flac", "m4a", "mp3", "mp4", "oga", "ogg", "wav"];

/// What one plan is rendered with: the cache it is looked up in and the settings that
/// name a render. `model_id` identifies the installed weights, so a plan made after a
/// weights update sees the old entries as misses.
pub struct Settings<'a> {
    pub cache: &'a Cache,
    pub model_id: &'a str,
    pub splice: bool,
}

/// What the cache holds for one file's render key, as the plan found it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EntryState {
    /// Nothing has been rendered for this source and these settings.
    Missing,
    /// An entry another cache version wrote: it is not a hit, and nothing resumes it.
    OtherVersion(String),
    /// A render that stopped before its last segment. The next one resumes it.
    Partial,
    /// The whole track, ready.
    Complete,
    /// A manifest that cannot be read or parsed. Rendering the file again is the only way
    /// out, and the render is where the failure gets reported.
    Unreadable(String),
}

impl EntryState {
    /// Why a file in this state is, or is not, rendered: one line for the dry run.
    pub fn reason(&self) -> String {
        match self {
            Self::Missing => "no entry".to_string(),
            Self::OtherVersion(version) => format!("an entry from {version}"),
            Self::Partial => "a partial entry, resuming".to_string(),
            Self::Complete => "cached".to_string(),
            Self::Unreadable(error) => format!("an entry that will not parse: {error}"),
        }
    }
}

/// What the plan does with one file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    Render,
    Skip,
}

/// The skip rule, in one place: only a complete entry means there is nothing to do.
/// Unknown-version, partial and unreadable entries all render; the segmented renderer
/// resumes what a killed one left and fails loudly on what it cannot repair.
pub fn decide(state: &EntryState) -> Decision {
    match state {
        EntryState::Complete => Decision::Skip,
        EntryState::Missing
        | EntryState::OtherVersion(_)
        | EntryState::Partial
        | EntryState::Unreadable(_) => Decision::Render,
    }
}

/// One file the plan will visit, with the snapshot of the cache it was decided from.
#[derive(Debug)]
pub struct PlannedFile {
    pub path: PathBuf,
    /// The render settings and source hash the key was made from, so the render writes
    /// the entry the plan checked instead of hashing the file a second time.
    pub fields: Fields,
    pub entry: EntryState,
}

impl PlannedFile {
    pub fn decision(&self) -> Decision {
        decide(&self.entry)
    }
}

/// A path the plan could not process: a folder it could not read, or a file it could not
/// hash. The runner reports these as failures and renders the rest.
#[derive(Debug)]
pub struct Problem {
    pub path: PathBuf,
    pub error: String,
}

/// The folder's files in path order, ready for [`run`].
#[derive(Debug)]
pub struct Plan {
    pub files: Vec<PlannedFile>,
    pub problems: Vec<Problem>,
}

impl Plan {
    /// Whether anything here needs the model. A folder already rendered needs none.
    pub fn has_render(&self) -> bool {
        self.files
            .iter()
            .any(|file| file.decision() == Decision::Render)
    }

    /// The dry run's text: every file with its decision, then what could not be read.
    pub fn describe(&self) -> String {
        let rendering = self
            .files
            .iter()
            .filter(|file| file.decision() == Decision::Render)
            .count();
        let mut text = String::new();
        let _ = writeln!(
            text,
            "{} files: {rendering} to render, {} skipped",
            self.files.len(),
            self.files.len() - rendering
        );
        for file in &self.files {
            let mark = match file.decision() {
                Decision::Render => "render",
                Decision::Skip => "skip  ",
            };
            let _ = writeln!(
                text,
                "  {mark}  {}  ({})",
                file.path.display(),
                file.entry.reason()
            );
        }
        for problem in &self.problems {
            let _ = writeln!(
                text,
                "  failed  {}  ({})",
                problem.path.display(),
                problem.error
            );
        }
        text
    }
}

/// Plan `dir`: walk it, keep what symphonia can decode, and ask the cache about each file.
pub fn plan(dir: &Path, settings: &Settings) -> Result<Plan> {
    anyhow::ensure!(dir.is_dir(), "{} is not a folder", dir.display());
    let mut paths = Vec::new();
    let mut problems = Vec::new();
    walk(dir, &mut paths, &mut problems);
    paths.sort();

    let mut planned = Vec::new();
    for path in paths {
        let source_hash = match cache::source_hash(&path) {
            Ok(source_hash) => source_hash,
            Err(error) => {
                problems.push(Problem {
                    path,
                    error: format!("{error:#}"),
                });
                continue;
            }
        };
        let fields = Fields {
            source_hash,
            window: render::WINDOW,
            overlap: render::OVERLAP,
            model_id: settings.model_id.to_string(),
            splice: settings.splice,
        };
        let entry = entry_state(settings.cache, &fields.key());
        planned.push(PlannedFile {
            path,
            fields,
            entry,
        });
    }
    Ok(Plan {
        files: planned,
        problems,
    })
}

/// Collect the audio files under `dir`, unordered: the caller sorts.
///
/// Symlinks are not followed, so a linked folder cannot make the walk loop. A subfolder
/// that cannot be read becomes a problem and the walk continues without it.
fn walk(dir: &Path, paths: &mut Vec<PathBuf>, problems: &mut Vec<Problem>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) => {
            problems.push(Problem {
                path: dir.to_path_buf(),
                error: format!("reading the folder: {error}"),
            });
            return;
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                problems.push(Problem {
                    path: dir.to_path_buf(),
                    error: format!("reading a folder entry: {error}"),
                });
                continue;
            }
        };
        let path = entry.path();
        let kind = match entry.file_type() {
            Ok(kind) => kind,
            Err(error) => {
                problems.push(Problem {
                    path,
                    error: format!("reading the file type: {error}"),
                });
                continue;
            }
        };
        if kind.is_dir() {
            walk(&path, paths, problems);
        } else if kind.is_file() && is_audio(&path) {
            paths.push(path);
        }
    }
}

/// Whether a file is worth decoding: a container we know by extension, or anything
/// symphonia can probe. The probe catches a renamed file; the extension catches a corrupt
/// one, which the render then reports by name.
fn is_audio(path: &Path) -> bool {
    if has_audio_extension(path) {
        return true;
    }
    opens(path)
}

fn has_audio_extension(path: &Path) -> bool {
    path.extension()
        .and_then(std::ffi::OsStr::to_str)
        .is_some_and(|extension| {
            AUDIO_EXTENSIONS
                .iter()
                .any(|known| extension.eq_ignore_ascii_case(known))
        })
}

/// Whether symphonia can open the file and it holds the audio its size claims.
///
/// The probe logs its own failure at ERROR, which would put a line about a file the walk
/// is deliberately ignoring on the user's terminal, so it is asked with no subscriber
/// attached: the answer is a boolean, not a report.
///
/// Opening is not enough. Symphonia decoded a 90 MB disk image as 2.5 s of 16 kHz
/// stereo, and its own frame count was absent, so the length is proven by decoding: a
/// file must hold at least as much audio as its bytes allow at the highest bitrate we
/// support. The decode stops the moment that much audio has arrived.
fn opens(path: &Path) -> bool {
    let bytes = std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
    let wanted = bytes as f64 / MAX_BYTES_PER_SECOND;
    let silent = tracing::subscriber::NoSubscriber::default();
    tracing::subscriber::with_default(silent, || match crate::decode::AudioStream::open(path) {
        Ok(mut stream) => stream.holds_at_least_seconds(wanted),
        Err(_) => false,
    })
}

/// The most a second of audio can plausibly need: 5 Mbit/s covers 24-bit 96 kHz stereo
/// with room to spare.
const MAX_BYTES_PER_SECOND: f64 = 5_000_000.0 / 8.0;

/// What the cache holds at `key`, read from the entry's manifest and nothing else.
///
/// This does not go through [`Cache::segments`]: that rewrites the entry's `last_used`,
/// which a dry run must not do, and it reports a partial entry exactly like a missing one
/// when the difference is what tells the user the next render resumes.
fn entry_state(cache: &Cache, key: &str) -> EntryState {
    let manifest = cache.entry_dir(key).join(MANIFEST);
    let text = match std::fs::read_to_string(&manifest) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return EntryState::Missing,
        Err(error) => {
            return EntryState::Unreadable(format!("reading {}: {error}", manifest.display()))
        }
    };
    let meta: cache::Meta = match serde_json::from_str(&text) {
        Ok(meta) => meta,
        Err(error) => {
            return EntryState::Unreadable(format!("parsing {}: {error}", manifest.display()))
        }
    };
    if meta.version != CACHE_VERSION {
        return EntryState::OtherVersion(meta.version);
    }
    match meta.state {
        State::Complete => EntryState::Complete,
        State::Partial => EntryState::Partial,
    }
}

/// What the renderer did with one file.
pub enum Outcome {
    /// The cache now holds the whole track.
    Rendered,
    /// There is nothing to render for this file, for example because it is full-band.
    Skipped(&'static str),
}

/// One file a run rendered, and how long its render took.
#[derive(Debug)]
pub struct Rendered {
    pub path: PathBuf,
    pub elapsed: Duration,
}

/// One file a run left alone, and why.
#[derive(Debug)]
pub struct Skipped {
    pub path: PathBuf,
    pub reason: String,
}

/// One file a run could not render, with the error it stopped on.
#[derive(Debug)]
pub struct Failed {
    pub path: PathBuf,
    pub elapsed: Duration,
    pub error: String,
}

/// What a run did, file by file, and how long it took.
#[derive(Debug, Default)]
pub struct Report {
    pub rendered: Vec<Rendered>,
    pub skipped: Vec<Skipped>,
    pub failed: Vec<Failed>,
    pub elapsed: Duration,
}

impl Report {
    /// The process exit code: one when any file failed, zero otherwise.
    pub fn exit_code(&self) -> i32 {
        if self.failed.is_empty() {
            0
        } else {
            1
        }
    }

    /// Every path this run accounted for.
    pub fn total(&self) -> usize {
        self.rendered.len() + self.skipped.len() + self.failed.len()
    }

    /// The markdown a rerun needs: what failed and why, what was skipped, what rendered.
    pub fn to_markdown(&self) -> String {
        let mut text = String::new();
        let _ = writeln!(text, "# flacify batch");
        let _ = writeln!(
            text,
            "\n{} files in {}: {} rendered, {} skipped, {} failed",
            self.total(),
            elapsed_label(self.elapsed),
            self.rendered.len(),
            self.skipped.len(),
            self.failed.len()
        );
        if !self.failed.is_empty() {
            let _ = writeln!(text, "\n## failed\n");
            for failed in &self.failed {
                let _ = writeln!(
                    text,
                    "- `{}` ({}): {}",
                    failed.path.display(),
                    elapsed_label(failed.elapsed),
                    failed.error
                );
            }
        }
        if !self.skipped.is_empty() {
            let _ = writeln!(text, "\n## skipped\n");
            for skipped in &self.skipped {
                let _ = writeln!(text, "- `{}`: {}", skipped.path.display(), skipped.reason);
            }
        }
        if !self.rendered.is_empty() {
            let _ = writeln!(text, "\n## rendered\n");
            for rendered in &self.rendered {
                let _ = writeln!(
                    text,
                    "- `{}` in {}",
                    rendered.path.display(),
                    elapsed_label(rendered.elapsed)
                );
            }
        }
        text
    }

    /// Write the report beside the cache, so the next run can be read alongside it.
    pub fn write(&self, cache_root: &Path) -> Result<PathBuf> {
        let path = report_path(cache_root);
        std::fs::create_dir_all(cache_root)
            .with_context(|| format!("creating {}", cache_root.display()))?;
        std::fs::write(&path, self.to_markdown())
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(path)
    }
}

/// Where the last run's report lives: beside the cache's entries, never inside one.
pub fn report_path(cache_root: &Path) -> PathBuf {
    cache_root.join("batch-last.md")
}

/// Walk the plan and render one file at a time.
///
/// Every file is attempted: a renderer error becomes a line in the report and the walk
/// goes on. The caller reads [`Report::exit_code`] to fail the process.
pub fn run(plan: Plan, mut render_one: impl FnMut(&PlannedFile) -> Result<Outcome>) -> Report {
    let started = Instant::now();
    let mut report = Report::default();
    for problem in plan.problems {
        report.failed.push(Failed {
            path: problem.path,
            elapsed: Duration::ZERO,
            error: problem.error,
        });
    }
    for file in &plan.files {
        match file.decision() {
            Decision::Skip => report.skipped.push(Skipped {
                path: file.path.clone(),
                reason: file.entry.reason(),
            }),
            Decision::Render => {
                let at = Instant::now();
                let outcome = render_one(file);
                let elapsed = at.elapsed();
                match outcome {
                    Ok(Outcome::Rendered) => report.rendered.push(Rendered {
                        path: file.path.clone(),
                        elapsed,
                    }),
                    Ok(Outcome::Skipped(reason)) => report.skipped.push(Skipped {
                        path: file.path.clone(),
                        reason: reason.to_string(),
                    }),
                    Err(error) => report.failed.push(Failed {
                        path: file.path.clone(),
                        elapsed,
                        error: format!("{error:#}"),
                    }),
                }
            }
        }
    }
    report.elapsed = started.elapsed();
    report
}

/// Render one planned file into the cache, with no device and no whole-track buffer.
///
/// The worker is the caller's, so a batch loads the model once. The source tier and the
/// segmented renderer are the ones a play uses, which is what makes a batch and a play
/// produce the same entry and share the same resume.
pub fn render_file(
    file: &PlannedFile,
    worker: Arc<WorkerSession>,
    cache: &Cache,
) -> Result<Outcome> {
    let key = file.fields.key();
    // Two files with the same bytes share a key, and another batch may have finished this
    // file since the plan was made; either way the entry is already what the render would
    // write, and `Cache::begin` refuses to reopen it.
    if cache.segments(&key)?.is_some() {
        return Ok(Outcome::Skipped("cached: these bytes are already rendered"));
    }

    let segment_frames = cache::frames_per_segment(render::WINDOW, render::OVERLAP)?;
    let tier = source::Tier::new(cache.root(), segment_frames)?;
    // The plan already hashed the file; the tier would read it again for its own name.
    let source = tier.build_or_reuse(&file.path, &file.fields.source_hash)?;
    let analysis = render::analyse_streaming(&source)
        .with_context(|| format!("analysing {}", file.path.display()))?;
    if analysis.looks_full_band() {
        return Ok(Outcome::Skipped("already full-band; nothing to generate"));
    }

    let frames = source.meta().frames;
    let mut writer = cache.begin(
        &key,
        &EntrySpec {
            frames,
            cutoff_hz: analysis.cutoff_hz,
            fields: file.fields.clone(),
        },
    )?;
    let seed = render::seed_streaming(&source)?;
    let rendered = render::render_segments(
        &source,
        render::Job {
            session: worker,
            cutoff_hz: Some(analysis.cutoff_hz),
            seed,
            splice_low_band: file.fields.splice,
        },
        &mut writer,
        &AtomicBool::new(false),
        &mut |_| {},
        |_, _| Ok(()),
    );
    if rendered.is_err() {
        // Keep the segments that landed: the next run resumes them instead of starting
        // the file again.
        writer.abandon();
    }
    rendered.map(|_| Outcome::Rendered)
}

/// A duration as a person reads it: seconds below a minute, minutes and seconds above.
fn elapsed_label(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs_f64();
    if seconds < 60.0 {
        format!("{seconds:.1} s")
    } else {
        format!("{}m {:02.0}s", (seconds / 60.0) as u64, seconds % 60.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn given_a_short_file_when_asked_for_more_audio_than_it_holds_then_it_says_no() {
        // A file cannot contain more audio than it has; this is the check that refuses a
        // container which probed as audio and decoded a few seconds out of a large file.
        let root = temp_root("decode-holds-short");
        std::fs::create_dir_all(&root).expect("creating the root");
        let path = root.join("short.wav");
        wav::write(&path, 48_000, 2, &vec![0.1f32; 4_800]).expect("writing the wav");
        let mut stream = crate::decode::AudioStream::open(&path).expect("opening the wav");
        assert!(stream.holds_at_least_seconds(0.05));
        let mut stream = crate::decode::AudioStream::open(&path).expect("opening the wav");
        assert!(!stream.holds_at_least_seconds(10.0));
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_large_file_of_noise_when_probed_then_it_is_not_audio_at_its_own_size() {
        // The shape the walk met on a real disk: a container that opens and decodes a
        // short stream out of far more bytes than any audio at 5 Mbit/s could fill.
        let root = temp_root("walk-large-non-audio");
        std::fs::create_dir_all(&root).expect("creating the root");
        let path = root.join("image.dmg");
        std::fs::write(&path, vec![0x5au8; 20_000_000]).expect("writing the image");
        assert!(!is_audio(&path));
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_the_walk_over_a_non_audio_file_when_planned_then_it_is_left_out() {
        let root = temp_root("plan-disk-image");
        std::fs::create_dir_all(&root).expect("creating the root");
        // A file that symphonia will happily "decode": 200 kB of zeros, which its probe
        // reads as a short stream, next to a real WAV.
        std::fs::write(root.join("big.dmg"), vec![0u8; 200_000]).expect("writing the image");
        let source = root.join("real.wav");
        wav::write(&source, 48_000, 2, &vec![0.25f32; 4_800]).expect("writing the wav");
        let cache = Cache::new(root.join("cache"));

        let plan = plan(&root, &settings(&cache)).expect("planning");
        let names: Vec<String> = plan
            .files
            .iter()
            .map(|file| file.path.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, vec!["real.wav".to_string()]);
        std::fs::remove_dir_all(root).ok();
    }
    use crate::limit::LimitReport;
    use crate::wav;

    /// A unique directory in the temp directory, deleted by the test that made it.
    fn temp_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "flacify-batch-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("a clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("creating the test directory");
        root
    }

    /// A stereo tone at 48 kHz: real audio, so hashing and probing behave as on a track.
    fn write_tone(path: &Path, hertz: f32, seconds: f32) {
        let frames = (seconds * 48_000.0) as usize;
        let mut planar = vec![0.0f32; frames * 2];
        for frame in 0..frames {
            let value = 0.5 * (std::f32::consts::TAU * hertz * frame as f32 / 48_000.0).sin();
            planar[frame] = value;
            planar[frames + frame] = value;
        }
        wav::write(path, 48_000, 2, &planar).expect("writing a wav");
    }

    /// A stereo track with energy spread below 21 kHz, so the analysis finds a cutoff the
    /// pipeline can use. A single tone reads as a 24 kHz cutoff, which no preprocessor
    /// accepts, and that is a property of the analysis rather than of the batch.
    fn write_track(path: &Path, seconds: f32, phase: f32) {
        let frames = (seconds * 48_000.0) as usize;
        let mut planar = vec![0.0f32; frames * 2];
        for frame in 0..frames {
            let time = frame as f32 / 48_000.0;
            let mut value = 0.0;
            for harmonic in 1..=30 {
                let hertz = 500.0 * harmonic as f32;
                value += 0.02 * (std::f32::consts::TAU * hertz * time + phase).sin();
            }
            planar[frame] = value;
            planar[frames + frame] = value;
        }
        wav::write(path, 48_000, 2, &planar).expect("writing a wav");
    }

    fn settings(cache: &Cache) -> Settings<'_> {
        Settings {
            cache,
            model_id: "fake",
            splice: false,
        }
    }

    /// The fields a plan makes for one file, so a test can store or open its entry.
    fn fields_for(path: &Path) -> Fields {
        Fields {
            source_hash: cache::source_hash(path).expect("hashing the source"),
            window: render::WINDOW,
            overlap: render::OVERLAP,
            model_id: "fake".to_string(),
            splice: false,
        }
    }

    /// A planned file that never touches the disk: for the runner's own tests.
    fn planned(path: &str, entry: EntryState) -> PlannedFile {
        PlannedFile {
            path: PathBuf::from(path),
            fields: Fields {
                source_hash: path.to_string(),
                window: render::WINDOW,
                overlap: render::OVERLAP,
                model_id: "fake".to_string(),
                splice: false,
            },
            entry,
        }
    }

    /// The fake worker `cargo` builds beside this test binary; `render.rs` looks it up the
    /// same way for its own tests.
    fn fake_worker() -> PathBuf {
        let exe = std::env::current_exe().expect("the test binary's path");
        let name = if cfg!(windows) {
            "fake-flashsr-worker.exe"
        } else {
            "fake-flashsr-worker"
        };
        for dir in exe.ancestors().skip(1) {
            for candidate in [dir.join(name), dir.join("deps").join(name)] {
                if candidate.is_file() {
                    return candidate;
                }
            }
        }
        panic!(
            "{} was not found above {}; run `cargo test --workspace --all-targets`",
            name,
            exe.display()
        );
    }

    /// One worker for the whole batch, on the fake sidecar: no Python and no weights.
    fn fake_session() -> Arc<WorkerSession> {
        use flacify_infer::python::{
            WorkerConfig, DEFAULT_EXCHANGE_TIMEOUT, DEFAULT_HANDSHAKE_TIMEOUT,
        };

        WorkerSession::with_args(
            WorkerConfig {
                program: fake_worker(),
                worker_script: PathBuf::from("tools/worker.py"),
                model_dir: PathBuf::from("unused"),
                flashsr_dir: None,
                device: "fake".to_string(),
                timeout: DEFAULT_EXCHANGE_TIMEOUT,
                handshake_timeout: DEFAULT_HANDSHAKE_TIMEOUT,
            },
            vec!["--gain".to_string(), "0.5".to_string()],
        )
    }

    #[test]
    fn given_a_folder_of_twenty_audio_files_when_planned_then_each_file_is_a_render_in_path_order()
    {
        let root = temp_root("plan-20");
        let music = root.join("music");
        std::fs::create_dir_all(music.join("disc2")).expect("creating a subfolder");
        for index in 0..18 {
            write_tone(
                &music.join(format!("{index:02}.wav")),
                300.0 + 10.0 * index as f32,
                0.1,
            );
        }
        write_tone(&music.join("disc2/18.wav"), 900.0, 0.1);
        write_tone(&music.join("disc2/19.wav"), 910.0, 0.1);
        std::fs::write(music.join("notes.txt"), "not audio").expect("writing a note");
        let cache = Cache::new(root.join("cache"));

        let plan = plan(&music, &settings(&cache)).expect("a plan");

        assert_eq!(plan.files.len(), 20);
        assert!(plan.problems.is_empty(), "{:?}", plan.problems);
        let paths: Vec<PathBuf> = plan.files.iter().map(|file| file.path.clone()).collect();
        let mut sorted = paths.clone();
        sorted.sort();
        assert_eq!(paths, sorted, "the plan is in path order");
        assert_eq!(paths[0], music.join("00.wav"));
        assert_eq!(paths[19], music.join("disc2/19.wav"));
        assert!(plan
            .files
            .iter()
            .all(|file| file.decision() == Decision::Render));
        assert!(plan.has_render());
        assert!(
            plan.describe()
                .contains("20 files: 20 to render, 0 skipped"),
            "{}",
            plan.describe()
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_complete_entry_when_planned_then_that_file_is_skipped_as_cached() {
        let root = temp_root("plan-cached");
        let music = root.join("music");
        std::fs::create_dir_all(&music).expect("creating the folder");
        let cached = music.join("a.wav");
        write_tone(&cached, 440.0, 0.1);
        let cache = Cache::new(root.join("cache"));
        let fields = fields_for(&cached);
        let key = fields.key();
        cache
            .store(
                &key,
                &render::Render {
                    planar: vec![0.0f32; 4_800 * 2],
                    limit: LimitReport {
                        peak_before: 0.0,
                        peak_after: 0.0,
                        limited_samples: 0,
                    },
                    device: "fake".to_string(),
                    gain_db: 0.0,
                    lufs_input: -14.0,
                    lufs_output: -14.0,
                },
                12_000,
                &fields,
            )
            .expect("storing the entry");

        let plan = plan(&music, &settings(&cache)).expect("a plan");

        assert_eq!(plan.files.len(), 1);
        assert_eq!(plan.files[0].entry, EntryState::Complete);
        assert_eq!(plan.files[0].decision(), Decision::Skip);
        assert!(!plan.has_render());
        let described = plan.describe();
        assert!(described.contains("0 to render, 1 skipped"), "{described}");
        assert!(described.contains("(cached)"), "{described}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_partial_entry_when_planned_then_the_file_renders_and_says_it_resumes() {
        let root = temp_root("plan-partial");
        let music = root.join("music");
        std::fs::create_dir_all(&music).expect("creating the folder");
        let file = music.join("a.wav");
        write_tone(&file, 440.0, 0.1);
        let cache = Cache::new(root.join("cache"));
        let fields = fields_for(&file);
        cache
            .begin(
                &fields.key(),
                &EntrySpec {
                    frames: 4_800,
                    cutoff_hz: 12_000,
                    fields,
                },
            )
            .expect("starting a partial entry");

        let plan = plan(&music, &settings(&cache)).expect("a plan");

        assert_eq!(plan.files[0].entry, EntryState::Partial);
        assert_eq!(plan.files[0].decision(), Decision::Render);
        assert!(
            plan.describe().contains("a partial entry, resuming"),
            "{}",
            plan.describe()
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_an_entry_from_another_cache_version_when_planned_then_the_file_renders() {
        let root = temp_root("plan-version");
        let music = root.join("music");
        std::fs::create_dir_all(&music).expect("creating the folder");
        let file = music.join("a.wav");
        write_tone(&file, 440.0, 0.1);
        let cache = Cache::new(root.join("cache"));
        let fields = fields_for(&file);
        let dir = cache.entry_dir(&fields.key());
        std::fs::create_dir_all(&dir).expect("creating the entry");
        std::fs::write(
            dir.join(MANIFEST),
            r#"{"version":"flacify-cache-1","frames":4800,"sample_rate":48000,
                "cutoff_hz":12000,"peak_before":0.0,"peak_after":0.0,
                "limited_samples":0,"device":"old","state":"complete"}"#,
        )
        .expect("writing the manifest");

        let plan = plan(&music, &settings(&cache)).expect("a plan");

        assert_eq!(
            plan.files[0].entry,
            EntryState::OtherVersion("flacify-cache-1".to_string())
        );
        assert_eq!(plan.files[0].decision(), Decision::Render);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_corrupt_manifest_when_planned_then_the_walk_continues_and_both_files_render() {
        let root = temp_root("plan-corrupt");
        let music = root.join("music");
        std::fs::create_dir_all(&music).expect("creating the folder");
        let broken = music.join("a.wav");
        write_tone(&broken, 440.0, 0.1);
        write_tone(&music.join("b.wav"), 880.0, 0.1);
        let cache = Cache::new(root.join("cache"));
        let dir = cache.entry_dir(&fields_for(&broken).key());
        std::fs::create_dir_all(&dir).expect("creating the entry");
        std::fs::write(dir.join(MANIFEST), "{ not a manifest").expect("writing a broken manifest");

        let plan = plan(&music, &settings(&cache)).expect("a plan");

        assert_eq!(
            plan.files.len(),
            2,
            "the unreadable entry did not stop the walk"
        );
        assert!(matches!(plan.files[0].entry, EntryState::Unreadable(_)));
        assert_eq!(plan.files[0].decision(), Decision::Render);
        assert_eq!(plan.files[1].entry, EntryState::Missing);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_non_audio_and_renamed_audio_when_planned_then_only_the_decodable_files_are_planned() {
        let root = temp_root("plan-extensions");
        let music = root.join("music");
        std::fs::create_dir_all(&music).expect("creating the folder");
        write_tone(&music.join("upper.WAV"), 440.0, 0.1);
        write_tone(&music.join("renamed.dat"), 440.0, 0.1);
        std::fs::write(music.join("cover.jpg"), [0xffu8, 0xd8, 0xff]).expect("writing an image");
        std::fs::write(music.join("notes.txt"), "not audio").expect("writing a note");
        let cache = Cache::new(root.join("cache"));

        let plan = plan(&music, &settings(&cache)).expect("a plan");

        let names: Vec<String> = plan
            .files
            .iter()
            .map(|file| {
                file.path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        assert_eq!(names, vec!["renamed.dat", "upper.WAV"]);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_file_when_planned_then_it_is_refused_as_a_folder() {
        let root = temp_root("plan-file");
        let file = root.join("a.wav");
        write_tone(&file, 440.0, 0.1);
        let cache = Cache::new(root.join("cache"));

        let error = plan(&file, &settings(&cache)).expect_err("a file is not a folder");

        assert!(
            format!("{error:#}").contains("is not a folder"),
            "{error:#}"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_plan_with_a_failing_file_when_run_then_the_rest_are_attempted_and_the_exit_code_is_one(
    ) {
        let plan = Plan {
            files: vec![
                planned("a.wav", EntryState::Missing),
                planned("b.wav", EntryState::Missing),
                planned("c.wav", EntryState::Complete),
                planned("d.wav", EntryState::Partial),
            ],
            problems: vec![Problem {
                path: PathBuf::from("locked"),
                error: "reading the folder: permission denied".to_string(),
            }],
        };
        let mut visited = Vec::new();

        let report = run(plan, |file| {
            visited.push(file.path.clone());
            if file.path.ends_with("b.wav") {
                anyhow::bail!("the decoder refused it");
            }
            Ok(Outcome::Rendered)
        });

        assert_eq!(
            visited,
            vec![
                PathBuf::from("a.wav"),
                PathBuf::from("b.wav"),
                PathBuf::from("d.wav")
            ],
            "the cached file is not rendered and the walk does not stop at the failure"
        );
        assert_eq!(report.rendered.len(), 2);
        assert_eq!(report.skipped.len(), 1);
        assert_eq!(report.skipped[0].reason, "cached");
        assert_eq!(report.failed.len(), 2);
        assert_eq!(report.failed[0].path, PathBuf::from("locked"));
        assert_eq!(report.failed[1].error, "the decoder refused it");
        assert_eq!(report.total(), 5);
        assert_eq!(report.exit_code(), 1);
    }

    #[test]
    fn given_a_file_with_nothing_to_render_when_run_then_it_is_skipped_and_the_exit_code_is_zero() {
        let plan = Plan {
            files: vec![planned("a.wav", EntryState::Missing)],
            problems: Vec::new(),
        };

        let report = run(plan, |_| {
            Ok(Outcome::Skipped("already full-band; nothing to generate"))
        });

        assert!(report.rendered.is_empty());
        assert_eq!(report.failed.len(), 0);
        assert_eq!(
            report.skipped[0].reason,
            "already full-band; nothing to generate"
        );
        assert_eq!(report.exit_code(), 0);
    }

    #[test]
    fn given_a_report_when_written_then_it_lands_beside_the_cache_with_failures_first() {
        let root = temp_root("report");
        let cache = root.join("cache");
        let report = Report {
            rendered: vec![Rendered {
                path: PathBuf::from("ok.wav"),
                elapsed: Duration::from_millis(1_500),
            }],
            skipped: vec![Skipped {
                path: PathBuf::from("done.wav"),
                reason: "cached".to_string(),
            }],
            failed: vec![Failed {
                path: PathBuf::from("bad.wav"),
                elapsed: Duration::from_millis(400),
                error: "the decoder refused it".to_string(),
            }],
            elapsed: Duration::from_secs(90),
        };

        let path = report.write(&cache).expect("writing the report");

        assert_eq!(path, cache.join("batch-last.md"));
        let text = std::fs::read_to_string(&path).expect("reading the report");
        assert!(
            text.contains("3 files in 1m 30s: 1 rendered, 1 skipped, 1 failed"),
            "{text}"
        );
        assert!(
            text.contains("`bad.wav` (0.4 s): the decoder refused it"),
            "{text}"
        );
        assert!(text.contains("`done.wav`: cached"), "{text}");
        assert!(text.contains("`ok.wav` in 1.5 s"), "{text}");
        let failed = text.find("## failed").expect("a failed section");
        let skipped = text.find("## skipped").expect("a skipped section");
        let rendered = text.find("## rendered").expect("a rendered section");
        assert!(failed < skipped && skipped < rendered, "{text}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_two_files_and_a_fake_worker_when_batched_twice_then_the_second_run_renders_nothing() {
        let root = temp_root("render-twice");
        let music = root.join("music");
        std::fs::create_dir_all(&music).expect("creating the folder");
        write_track(&music.join("a.wav"), 1.0, 0.0);
        write_track(&music.join("b.wav"), 1.0, 0.9);
        let cache = Cache::new(root.join("cache"));
        let worker = fake_session();

        let first = plan(&music, &settings(&cache)).expect("a plan");
        let report = run(first, |file| render_file(file, Arc::clone(&worker), &cache));
        assert_eq!(
            report.failed.len(),
            0,
            "rendering failed: {:?}",
            report.failed
        );
        assert_eq!(report.rendered.len(), 2);
        assert_eq!(report.exit_code(), 0);
        for path in [music.join("a.wav"), music.join("b.wav")] {
            let entry = cache
                .segments(&fields_for(&path).key())
                .expect("looking the entry up")
                .expect("a complete entry the batch wrote");
            assert!(entry.meta().frames > 0);
        }

        let second = plan(&music, &settings(&cache)).expect("a second plan");
        assert!(second
            .files
            .iter()
            .all(|file| file.decision() == Decision::Skip));
        let report = run(second, |file| {
            render_file(file, Arc::clone(&worker), &cache)
        });
        assert_eq!(
            report.rendered.len(),
            0,
            "a completed file is not rendered again"
        );
        assert_eq!(report.skipped.len(), 2);
        assert_eq!(report.exit_code(), 0);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_two_files_with_the_same_bytes_when_batched_then_one_render_serves_both() {
        let root = temp_root("render-duplicate");
        let music = root.join("music");
        std::fs::create_dir_all(&music).expect("creating the folder");
        write_track(&music.join("a.wav"), 1.0, 0.0);
        write_track(&music.join("b.wav"), 1.0, 0.0);
        let cache = Cache::new(root.join("cache"));
        let worker = fake_session();

        let plan = plan(&music, &settings(&cache)).expect("a plan");
        assert_eq!(
            plan.files.len(),
            2,
            "both files are planned before either renders"
        );
        let report = run(plan, |file| render_file(file, Arc::clone(&worker), &cache));

        assert_eq!(report.rendered.len(), 1);
        assert_eq!(report.failed.len(), 0, "{:?}", report.failed);
        assert_eq!(report.skipped.len(), 1);
        assert!(report.skipped[0].reason.contains("already rendered"));
        assert_eq!(report.exit_code(), 0);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_full_band_file_when_batched_then_it_is_skipped_and_no_entry_is_written() {
        let root = temp_root("render-full-band");
        let music = root.join("music");
        std::fs::create_dir_all(&music).expect("creating the folder");
        let file = music.join("hd.wav");
        write_tone(&file, 22_000.0, 2.0);
        let cache = Cache::new(root.join("cache"));

        let plan = plan(&music, &settings(&cache)).expect("a plan");
        assert_eq!(plan.files[0].decision(), Decision::Render);
        let report = run(plan, |file| render_file(file, fake_session(), &cache));

        assert!(report.rendered.is_empty(), "{:?}", report.failed);
        assert_eq!(report.skipped.len(), 1);
        assert!(report.skipped[0].reason.contains("full-band"));
        assert!(cache
            .segments(&fields_for(&file).key())
            .expect("looking the entry up")
            .is_none());
        std::fs::remove_dir_all(root).ok();
    }

    #[cfg(unix)]
    #[test]
    fn given_an_unreadable_file_and_folder_when_planned_then_both_are_problems_and_the_rest_plan() {
        use std::os::unix::fs::PermissionsExt;

        let root = temp_root("unreadable");
        let music = root.join("music");
        let locked_dir = music.join("locked");
        std::fs::create_dir_all(&locked_dir).expect("creating the folders");
        write_tone(&music.join("ok.wav"), 440.0, 0.1);
        let blocked = music.join("blocked.wav");
        write_tone(&blocked, 440.0, 0.1);
        std::fs::set_permissions(&blocked, std::fs::Permissions::from_mode(0o000))
            .expect("locking the file");
        std::fs::set_permissions(&locked_dir, std::fs::Permissions::from_mode(0o000))
            .expect("locking the folder");

        // root reads through 0o000: the test would prove nothing, so it says so and stops.
        if std::fs::read(&blocked).is_ok() {
            let _ = std::fs::set_permissions(&blocked, std::fs::Permissions::from_mode(0o644));
            let _ = std::fs::set_permissions(&locked_dir, std::fs::Permissions::from_mode(0o755));
            std::fs::remove_dir_all(root).ok();
            return;
        }

        let cache = Cache::new(root.join("cache"));
        let plan = plan(&music, &settings(&cache)).expect("a plan");

        assert_eq!(plan.files.len(), 1);
        assert_eq!(plan.files[0].path, music.join("ok.wav"));
        assert_eq!(plan.problems.len(), 2);
        assert!(plan.problems.iter().any(|problem| problem.path == blocked));
        assert!(plan
            .problems
            .iter()
            .any(|problem| problem.path == locked_dir));

        let report = run(plan, |_| Ok(Outcome::Rendered));
        assert_eq!(report.rendered.len(), 1);
        assert_eq!(report.failed.len(), 2);
        assert_eq!(report.exit_code(), 1);

        let _ = std::fs::set_permissions(&blocked, std::fs::Permissions::from_mode(0o644));
        let _ = std::fs::set_permissions(&locked_dir, std::fs::Permissions::from_mode(0o755));
        std::fs::remove_dir_all(root).ok();
    }
}
