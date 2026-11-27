//! `flacify enhance`: render the generated high band and write it to a file.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use anyhow::{Context, Result};
use flacify_core::SAMPLE_RATE;

use crate::render;
use crate::session::Session;
use crate::setup;
use crate::youtube;

pub(crate) fn run(file: &Path, out: Option<PathBuf>, splice: bool) -> Result<()> {
    let paths = setup::Paths::discover()?;
    if !paths.is_ready() {
        anyhow::bail!("the model is not installed yet. Run `flacify setup` once (about 6 GB).");
    }
    // A URL is fetched first; a path comes back as it was.
    let fetched = youtube::resolve(file, &paths.downloads_dir(), None)?;
    let from_url = youtube::parse(&file.to_string_lossy()).is_some();
    let file = fetched.path.as_path();
    let session = Session::new(&paths, splice)?;
    let out = out.unwrap_or_else(|| {
        let mut name = file.file_name().unwrap_or_default().to_os_string();
        name.push(".flacify.wav");
        // A file's render sits beside it. A fetched track's render goes to the working
        // directory instead: the download directory holds the source and its names file,
        // and a render there would be mistaken for a download on the next play.
        if from_url {
            PathBuf::from(name)
        } else {
            file.with_file_name(name)
        }
    });

    let started = Instant::now();
    let mut last_printed = 0u32;
    render::enhance_file(
        file,
        &out,
        session.worker(),
        render::Store {
            cache: session.cache(),
            model_id: session.model_id(),
        },
        splice,
        &AtomicBool::new(false),
        &mut |done, total| {
            if total == 0 {
                return;
            }
            let percent = (done * 100 / total) as u32;
            if percent >= last_printed + 5 {
                last_printed = percent;
                eprint!(
                    "\r  {percent:3}%  {:.0} s of audio",
                    done as f64 / f64::from(SAMPLE_RATE)
                );
            }
        },
    )
    .context("enhancing")?;
    eprintln!();
    println!(
        "wrote {} in {:.1} s",
        out.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
