//! `flacify`: set up the model, render a file, or play one with an A/B switch.
//!
//! The app is the whole product minus the model: decode, analysis, the windowing
//! pipeline, the buffer policy and the A/B switch are Rust. The model runs in a Python
//! sidecar the app installs and owns (ADR-010), and [`session`] keeps that sidecar alive
//! for the life of the app so a second file does not load it again.

mod cache;
mod cli;
mod commands;
mod decode;
mod device;
mod diagnose;
mod first_run;
mod flac;
mod frontier;
mod limit;
// `play --live` (U1) and `flacify sink` (K1) are the callers; until they land the mechanism
// has none outside the tests.
#[allow(dead_code)]
mod live;
mod loudness;
mod player;
mod render;
mod render_status;
mod segments;
mod session;
mod setup;
mod source;
mod spectrum;
mod splice;
mod ui;
mod wav;

use anyhow::Result;
use clap::Parser;

use crate::cli::{Cli, Command};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();

    match Cli::parse().command {
        Command::Setup { cuda } => commands::setup::run(cuda),
        Command::Probe { file, rtf } => commands::probe::run(&file, rtf),
        Command::Enhance { file, out, splice } => commands::enhance::run(&file, out, splice),
        Command::Play {
            file,
            smoke,
            splice,
        } => commands::play::run(&file, smoke, splice),
        Command::Devices => commands::devices::run(),
        Command::Batch { dir, dry_run } => commands::batch::run(&dir, dry_run),
        Command::Ab {
            file,
            trials,
            seed,
            export,
        } => commands::ab::run(&file, trials, seed, export.as_deref()),
        Command::Cache { evict, cap_gb } => commands::cache::run(evict, cap_gb),
        Command::Diagnose { out } => {
            let path = diagnose::run(out)?;
            println!("wrote {}", path.display());
            Ok(())
        }
    }
}

/// Counts allocations on the calling thread. Exists so a test can pin the audio
/// callback's no-allocation guarantee instead of restating it in a comment.
#[cfg(test)]
mod alloc_counter {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        /// `Some(bytes)` while a measurement is armed on this thread.
        static ALLOCATED: Cell<Option<usize>> = const { Cell::new(None) };
    }

    pub struct Counting;

    // SAFETY: every method forwards to `System`, which is the allocator these tests would
    // otherwise use; the counter is bookkeeping that never touches the returned pointer.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            add(layout.size());
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            // A realloc moves the allocation, so count the new bytes rather than the
            // difference: only "did it allocate anything" is asked of this number.
            add(new_size);
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    fn add(bytes: usize) {
        // `try_with`: an allocation during thread teardown must not panic in the allocator.
        let _ = ALLOCATED.try_with(|count| {
            if let Some(total) = count.get() {
                count.set(Some(total + bytes));
            }
        });
    }

    /// The bytes `body` allocated on this thread, with its result.
    pub fn measure<T>(body: impl FnOnce() -> T) -> (T, usize) {
        ALLOCATED.with(|total| total.set(Some(0)));
        let value = body();
        let bytes = ALLOCATED.with(|total| total.replace(None)).unwrap_or(0);
        (value, bytes)
    }
}

#[cfg(test)]
#[global_allocator]
static ALLOCATOR: alloc_counter::Counting = alloc_counter::Counting;
