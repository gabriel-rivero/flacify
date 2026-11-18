//! `flacify setup`: install the runtime, the wheels and the weights.

use anyhow::Result;

use crate::setup;

pub(crate) fn run(cuda: bool) -> Result<()> {
    if cuda {
        if cfg!(target_os = "macos") {
            anyhow::bail!(
                "macOS gets Metal from PyPI's torch; the CUDA wheel is for Windows and Linux"
            );
        }
        if std::env::var_os("FLACIFY_TORCH_INDEX").is_none() {
            std::env::set_var("FLACIFY_TORCH_INDEX", setup::CUDA_TORCH_INDEX);
        }
    }
    let paths = setup::Paths::discover()?;
    let bytes: u64 = setup::plan().iter().map(|step| step.bytes).sum();
    println!("flacify installs into {}", paths.root().display());
    println!(
        "this is a one-time download of about {:.1} GB:",
        bytes as f64 / 1e9
    );
    for step in setup::plan() {
        println!(
            "  {:<10} {} ({:.0} MB)",
            step.id,
            step.label,
            step.bytes as f64 / 1e6
        );
    }
    let progress = setup::Progress::new(|step, fraction| {
        if fraction >= 1.0 {
            println!("  {step}: done");
        } else {
            println!("  {step}: ...");
        }
    });
    setup::ensure(&paths, progress)?;
    println!("ready; try `flacify play <file>`");
    Ok(())
}
