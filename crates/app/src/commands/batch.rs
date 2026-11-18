//! `flacify batch`: wired by its own item; this module is the landing place.

use std::path::Path;

use anyhow::Result;

pub(crate) fn run(_dir: &Path, _dry_run: bool) -> Result<()> {
    anyhow::bail!("flacify batch is not wired yet")
}
