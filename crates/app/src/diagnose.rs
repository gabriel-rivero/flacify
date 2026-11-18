//! `flacify diagnose`: one text file with the facts an issue needs.
//!
//! It records versions, the data directory, whether the install is ready, which torch
//! index was chosen, the cache size, the device inventory and the model files. Nothing is
//! uploaded; the user decides where the file goes.
//!
//! The counters are per process, so a fresh `flacify diagnose` reports zeros: what makes
//! them useful is the same line in a pasted report when the app itself wrote it.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::cache::CACHE_VERSION;
use crate::device;
use crate::setup::{self, Paths};

/// Write the report and return its path.
pub fn run(out: Option<PathBuf>) -> Result<PathBuf> {
    let paths = Paths::discover()?;
    let out = out.unwrap_or_else(|| PathBuf::from("flacify-diagnose.txt"));
    std::fs::write(&out, report(&paths)).with_context(|| format!("writing {}", out.display()))?;
    Ok(out)
}

pub fn report(paths: &Paths) -> String {
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
    let _ = writeln!(
        text,
        "cache    {CACHE_VERSION}, {}",
        cache_summary(&paths.cache_dir())
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
    match device::list() {
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

fn cache_summary(root: &Path) -> String {
    let Ok(entries) = std::fs::read_dir(root) else {
        return "0 entries".to_string();
    };
    let mut count = 0usize;
    let mut bytes = 0u64;
    for entry in entries.flatten() {
        if entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false) {
            count += 1;
            bytes += setup::dir_size(&entry.path());
        }
    }
    format!("{count} entries, {:.0} MB", bytes as f64 / 1e6)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn given_an_empty_home_when_reported_then_it_says_not_ready_and_names_missing_models() {
        let root = std::env::temp_dir().join(format!("flacify-diagnose-{}", std::process::id()));
        std::fs::create_dir_all(&root).expect("creating the temp home");
        let paths = Paths::at(root.clone());

        let text = report(&paths);
        assert!(text.contains(env!("CARGO_PKG_VERSION")), "{text}");
        assert!(text.contains("ready    false"), "{text}");
        assert!(text.contains("vae.pth missing"), "{text}");
        assert!(text.contains("0 entries"), "{text}");
        assert!(text.contains("devices"), "{text}");
        assert!(text.contains("counters: 0 stream errors"), "{text}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_cache_entry_when_summarised_then_it_counts_the_entry_and_its_bytes() {
        let root = std::env::temp_dir().join(format!("flacify-cache-sum-{}", std::process::id()));
        let entry = root.join("entry");
        std::fs::create_dir_all(&entry).expect("creating the entry");
        std::fs::write(entry.join("manifest.json"), vec![0u8; 2_000_000]).expect("writing");

        let summary = cache_summary(&root);
        assert!(summary.contains("1 entries"), "{summary}");
        assert!(summary.contains("2 MB"), "{summary}");
        std::fs::remove_dir_all(root).ok();
    }
}
