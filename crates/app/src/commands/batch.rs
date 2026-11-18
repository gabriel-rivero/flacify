//! `flacify batch`: render a folder into the cache, one file at a time.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;

use crate::batch::{self, Outcome, Settings};
use crate::session::Session;
use crate::setup;

pub(crate) fn run(dir: &Path, dry_run: bool) -> Result<()> {
    let paths = setup::Paths::discover()?;
    let session = Session::new(&paths, false)?;
    let plan = batch::plan(
        dir,
        &Settings {
            cache: session.cache(),
            model_id: session.model_id(),
            splice: false,
        },
    )?;

    if dry_run {
        print!("{}", plan.describe());
        return Ok(());
    }

    // A folder that is already rendered needs no model at all, so the check waits for a
    // plan that will actually render something.
    if plan.has_render() && !paths.is_ready() {
        anyhow::bail!("the model is not installed yet; run `flacify setup` once (about 6 GB)");
    }

    let total = plan.files.len();
    let worker = session.worker();
    let mut index = 0;
    let report = batch::run(plan, |file| {
        index += 1;
        println!("[{index}/{total}] {}", file.path.display());
        let outcome = batch::render_file(file, Arc::clone(&worker), session.cache())?;
        match &outcome {
            Outcome::Rendered => println!("        done"),
            Outcome::Skipped(reason) => println!("        skipped: {reason}"),
        }
        Ok(outcome)
    });

    let written = report.write(session.cache().root())?;
    println!(
        "{} files in {:.1} s: {} rendered, {} skipped, {} failed",
        report.total(),
        report.elapsed.as_secs_f64(),
        report.rendered.len(),
        report.skipped.len(),
        report.failed.len()
    );
    println!("report {}", written.display());
    for failed in &report.failed {
        eprintln!("  failed  {}: {}", failed.path.display(), failed.error);
    }
    if report.exit_code() != 0 {
        anyhow::bail!(
            "{} of {} files failed; see {}",
            report.failed.len(),
            report.total(),
            written.display()
        );
    }
    Ok(())
}
