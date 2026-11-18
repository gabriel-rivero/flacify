//! `flacify cache`: wired by its own item; this module is the landing place.

use anyhow::Result;

pub(crate) fn run(_evict: bool, _cap_gb: Option<f64>) -> Result<()> {
    anyhow::bail!("flacify cache is not wired yet")
}
