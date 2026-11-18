//! `flacify probe`: the detected band limit and the playback verdict for a file.

use std::path::Path;

use anyhow::Result;
use flacify_core::{plan, Probe, SAMPLE_RATE};

use crate::render;

pub(crate) fn run(path: &Path, rtf: f64) -> Result<()> {
    let loaded = render::load_in_memory(path)?;
    let (track, analysis) = (&loaded.track, &loaded.analysis);
    let seconds = track.frames as f64 / f64::from(SAMPLE_RATE);
    let plan = plan(&Probe::from_rtf(rtf), 5.12, Some(seconds));

    println!("file      {}", path.display());
    println!(
        "format    {}, {} frames ({seconds:.2} s)",
        loaded.format_label(),
        track.frames
    );
    println!(
        "cutoff    {} Hz  (energy above 21 kHz: {:.4}% of the spectrum)",
        analysis.cutoff_hz,
        analysis.high_band_ratio * 100.0
    );
    if analysis.looks_full_band() {
        println!("verdict   already full-band; there is no top octave to infer");
        return Ok(());
    }
    println!(
        "verdict   the top {:.1} kHz would be inferred, not recovered",
        (f64::from(SAMPLE_RATE) / 2.0 - f64::from(analysis.cutoff_hz)) / 1000.0
    );
    println!("verdict   {}", mode_label(&plan));

    if let Some(warning) = &plan.warning {
        println!("\n{}", warning.headline);
        println!("{}", warning.detail);
    }
    Ok(())
}

fn mode_label(plan: &flacify_core::Plan) -> String {
    match plan.mode {
        flacify_core::Mode::Realtime => format!(
            "realtime: {:.1} s of prebuffer, first audio after {:.1} s",
            plan.prebuffer_secs, plan.time_to_first_audio_secs
        ),
        flacify_core::Mode::Marginal => format!(
            "marginal: {:.1} s of prebuffer, first audio after {:.1} s",
            plan.prebuffer_secs, plan.time_to_first_audio_secs
        ),
        flacify_core::Mode::PreRender => format!(
            "pre-render: {:.1} s of audio rendered before first sound, {:.0} s of wait",
            plan.prebuffer_secs,
            plan.estimated_wait_secs.unwrap_or_default()
        ),
    }
}
