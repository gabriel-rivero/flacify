//! `flacify ab`: wired by its own item; this module is the landing place.

use std::path::Path;

use anyhow::Result;

pub(crate) fn run(_file: &Path, _trials: u32, _seed: u64, _export: Option<&Path>) -> Result<()> {
    anyhow::bail!("flacify ab is not wired yet")
}
