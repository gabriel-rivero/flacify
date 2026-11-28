//! `flacify diagnose`: one text file with the facts an issue needs.
//!
//! It records versions, the data directory, whether the install is ready, which torch
//! index was chosen, the cache's size against its cap, the device inventory and the model
//! files. Nothing is uploaded; the user decides where the file goes.
//!
//! The counters are per process, so a fresh `flacify diagnose` reports zeros: what makes
//! them useful is the same line in a pasted report when the app itself wrote it.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::cache::{human_bytes, Cache, Usage, CACHE_VERSION, DEFAULT_CAP_BYTES};
use crate::device;
use crate::process;
use crate::setup::{self, Paths};
use crate::youtube;

/// Write the report and return its path.
pub fn run(out: Option<PathBuf>) -> Result<PathBuf> {
    let paths = Paths::discover()?;
    let out = out.unwrap_or_else(|| PathBuf::from("flacify-diagnose.txt"));
    std::fs::write(&out, report(&paths)).with_context(|| format!("writing {}", out.display()))?;
    Ok(out)
}

pub fn report(paths: &Paths) -> String {
    report_devices(paths, device::list())
}

/// The report with the device inventory supplied.
///
/// Enumerating real audio devices is a host call that Windows runners without an audio
/// device do not survive, so the unit tests hand in a list instead of touching cpal.
pub fn report_devices(paths: &Paths, devices: Result<device::DeviceList>) -> String {
    let mut text = String::new();
    let _ = writeln!(text, "flacify  {}", env!("CARGO_PKG_VERSION"));
    let _ = writeln!(
        text,
        "os       {} {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    let _ = writeln!(text, "home     {}", paths.root().display());
    let _ = writeln!(text, "ready    {}", paths.is_ready());
    let installed = std::fs::read_to_string(paths.wheels_receipt())
        .ok()
        .and_then(|receipt| {
            receipt.lines().find_map(|line| {
                line.strip_prefix("torch-index ")
                    .map(|index| index.trim().to_string())
            })
        })
        .filter(|index| !index.is_empty())
        .unwrap_or_else(|| "(none)".to_string());
    let torch = std::env::var("FLACIFY_TORCH_INDEX").unwrap_or_else(|_| "(default)".to_string());
    let _ = writeln!(text, "torch    installed {installed}, env {torch}");
    let _ = writeln!(text, "cache    {CACHE_VERSION}");
    let usage = Cache::new(paths.cache_dir()).usage();
    write_usage(&mut text, &usage);
    let _ = writeln!(text, "batch    {}", batch_report(&paths.cache_dir()));

    // Only presence is reported, not a version: `--version` would be a process spawn in
    // the one function the tests call, and a bug report can run it itself. yt-dlp is the
    // tool whose version moves under us, so naming where it is found matters most.
    let ytdlp = youtube::find_ytdlp()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "missing (only a URL needs it)".to_string());
    let _ = writeln!(text, "yt-dlp   {ytdlp}");
    // ffmpeg decides whether a fetch asks for a FLAC transcode or the container as served,
    // so a bug report about a URL needs it as much as yt-dlp's own path.
    let ffmpeg = process::find_on_path("ffmpeg")
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "missing (a fetch keeps the source's AAC then)".to_string());
    let _ = writeln!(text, "ffmpeg   {ffmpeg}");
    let _ = writeln!(
        text,
        "downloads {} bytes",
        setup::dir_size(&paths.downloads_dir())
    );

    let _ = writeln!(text, "\nsteps");
    for step in setup::plan() {
        let _ = writeln!(
            text,
            "  {:<10} {:>13} bytes  {}",
            step.id, step.bytes, step.label
        );
    }

    let _ = writeln!(text, "\nmodels");
    for name in ["student_ldm.pth", "sr_vocoder.pth", "vae.pth"] {
        let path = paths.models_dir().join(name);
        match std::fs::metadata(&path) {
            Ok(meta) => {
                let _ = writeln!(text, "  {name} {} bytes", meta.len());
            }
            Err(_) => {
                let _ = writeln!(text, "  {name} missing");
            }
        }
    }

    let _ = writeln!(text, "\ndevices");
    match devices {
        Ok(list) => {
            for line in device::describe(&list).lines() {
                let _ = writeln!(text, "  {line}");
            }
        }
        Err(error) => {
            let _ = writeln!(text, "  unavailable: {error:#}");
        }
    }
    let counters = device::counters();
    let _ = writeln!(
        text,
        "  counters: {} stream errors, {} buffers over {} frames",
        counters.stream_errors,
        counters.buffer_overflows,
        device::MAX_FRAMES
    );
    text
}

/// The disk-pressure block: every kind of thing the cache holds, and the cap it is measured
/// against, so a pasted report answers "is this a full disk" without a second question.
fn write_usage(text: &mut String, usage: &Usage) {
    for (kind, entries, bytes) in [
        ("rendered", usage.rendered.entries, usage.rendered.bytes),
        ("partial", usage.partial.entries, usage.partial.bytes),
        ("source", usage.source.entries, usage.source.bytes),
        ("unknown", usage.unknown.entries, usage.unknown.bytes),
        ("stale", usage.stale.entries, usage.stale.bytes),
    ] {
        let label = if entries == 1 { "entry" } else { "entries" };
        let _ = writeln!(
            text,
            "  {kind:<9} {entries:>4} {label:<7} {:>11}",
            human_bytes(bytes)
        );
    }
    let _ = writeln!(
        text,
        "  {:<9} {:>14} {:>11}",
        "total",
        format!("{} entries", usage.entries()),
        human_bytes(usage.total())
    );
    let _ = writeln!(text, "  cap       {}", human_bytes(DEFAULT_CAP_BYTES));
    if usage.unaccounted_bytes > 0 || usage.overstated_entries > 0 {
        let _ = writeln!(
            text,
            "  manifests {} bytes unaccounted, {} overstated",
            usage.unaccounted_bytes, usage.overstated_entries
        );
    }
}

/// Where the last `flacify batch` run left its failures and skips, or that there is none.
fn batch_report(cache_root: &Path) -> String {
    let path = crate::batch::report_path(cache_root);
    if path.is_file() {
        format!("last run {}", path.display())
    } else {
        "no run yet".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{Meta, Segment, State};
    use std::path::Path;

    /// A manifest this build can read, with one segment and a `bytes` size the test picks.
    fn write_entry(cache_dir: &Path, key: &str, bytes: usize, state: State) {
        let dir = cache_dir.join(key);
        std::fs::create_dir_all(&dir).expect("creating the entry");
        let meta = Meta {
            version: CACHE_VERSION.to_string(),
            frames: 420,
            sample_rate: 48_000,
            frames_per_segment: 420,
            cutoff_hz: 12_000,
            source_hash: "source".into(),
            window: 420,
            overlap: 60,
            splice: false,
            peak_before: 0.5,
            peak_after: 0.5,
            limited_samples: 0,
            gain_db: Some(0.0),
            device: "cpu".into(),
            state,
            rendered_frames: 420,
            segments: vec![Segment {
                index: 0,
                frames: 420,
                bytes: bytes as u64,
            }],
            last_used: 10,
            created_at: 10,
        };
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_string(&meta).expect("serialising"),
        )
        .expect("writing the manifest");
        std::fs::write(dir.join("seg-000000.flac"), vec![0u8; bytes]).expect("writing a segment");
    }

    /// The accounting line for one kind, which is the only place a report says how much of
    /// that kind the cache holds.
    fn line_for<'a>(text: &'a str, kind: &str) -> &'a str {
        text.lines()
            .find(|line| line.trim_start().starts_with(kind))
            .unwrap_or_else(|| panic!("no {kind} line in {text}"))
    }

    #[test]
    fn given_an_empty_home_when_reported_then_it_says_not_ready_and_names_missing_models() {
        let root = std::env::temp_dir().join(format!("flacify-diagnose-{}", std::process::id()));
        std::fs::create_dir_all(&root).expect("creating the temp home");
        let paths = Paths::at(root.clone());

        let text = report_devices(&paths, Ok(device::DeviceList::default()));
        assert!(text.contains(env!("CARGO_PKG_VERSION")), "{text}");
        assert!(text.contains("ready    false"), "{text}");
        assert!(text.contains("vae.pth missing"), "{text}");
        assert!(line_for(&text, "rendered").contains("0 entries"), "{text}");
        assert!(line_for(&text, "total").contains("0 entries"), "{text}");
        assert!(text.contains("20.0 GB"), "the cap is stated: {text}");
        assert!(text.contains("batch    no run yet"), "{text}");
        assert!(text.contains("devices"), "{text}");
        assert!(text.contains("counters: 0 stream errors"), "{text}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_batch_report_beside_the_cache_when_reported_then_it_names_the_file() {
        let root =
            std::env::temp_dir().join(format!("flacify-diagnose-batch-{}", std::process::id()));
        let cache = root.join("cache");
        std::fs::create_dir_all(&cache).expect("creating the cache");
        let paths = Paths::at(root.clone());
        let written = crate::batch::report_path(&cache);
        std::fs::write(&written, "# flacify batch\n").expect("writing the report");

        let text = report_devices(&paths, Ok(device::DeviceList::default()));
        assert!(
            text.contains(&format!("batch    last run {}", written.display())),
            "{text}"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_complete_entry_a_partial_one_and_a_stale_directory_when_reported_then_each_kind_is_counted(
    ) {
        let root = std::env::temp_dir().join(format!("flacify-cache-sum-{}", std::process::id()));
        let paths = Paths::at(root.clone());
        let cache_dir = paths.cache_dir();
        write_entry(&cache_dir, "done", 2_000_000, State::Complete);
        write_entry(&cache_dir, "half", 1_000_000, State::Partial);
        std::fs::create_dir_all(cache_dir.join("left-behind.tmp")).expect("creating a stale dir");

        let text = report_devices(&paths, Ok(device::DeviceList::default()));
        assert!(line_for(&text, "rendered").contains("1 entry "), "{text}");
        assert!(line_for(&text, "rendered").contains("2.0 MB"), "{text}");
        assert!(line_for(&text, "partial").contains("1 entry "), "{text}");
        assert!(line_for(&text, "stale").contains("1 entry "), "{text}");
        assert!(
            line_for(&text, "total").contains("2 entries"),
            "a stale directory is not an entry: {text}"
        );
        std::fs::remove_dir_all(root).ok();
    }
}
