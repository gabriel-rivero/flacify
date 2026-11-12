//! The Rust preprocessing against golden output from the reference Python.
//!
//! `tools/fixtures/gen.py` writes a stereo window, the signal after the zero-phase
//! Chebyshev stage, and the signal after the resample round trip. Three files per case,
//! because when the port goes wrong the first thing worth knowing is which stage moved.

use std::path::{Path, PathBuf};

use flacify_core::dsp::cutoff::analyse;
use flacify_core::Preprocessor;
use serde::Deserialize;

#[derive(Deserialize)]
struct Manifest {
    window: usize,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    true_cutoff: Option<u32>,
    detected_cutoff: u32,
    channels: usize,
    frames: usize,
    rms_out: f64,
}

fn fixtures() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/fixtures/out");
    if dir.join("manifest.json").exists() {
        Some(dir)
    } else {
        eprintln!(
            "skipping fixture tests: {} is missing. Run `python3 tools/fixtures/gen.py`.",
            dir.display()
        );
        None
    }
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn planar(data: &[f32], channels: usize, frames: usize) -> Vec<&[f32]> {
    (0..channels)
        .map(|c| &data[c * frames..(c + 1) * frames])
        .collect()
}

fn max_abs_diff(a: &[f64], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - f64::from(*y)).abs())
        .fold(0.0f64, f64::max)
}

#[test]
fn given_the_reference_fixtures_when_preprocessed_then_each_stage_matches() {
    let Some(dir) = fixtures() else { return };
    let manifest: Manifest =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap())
            .expect("manifest");
    assert!(!manifest.cases.is_empty());

    for case in &manifest.cases {
        let input = read_f32(&dir.join(format!("{}.in.f32", case.name)));
        let expected_filtered = read_f32(&dir.join(format!("{}.filtered.f32", case.name)));
        let expected_out = read_f32(&dir.join(format!("{}.out.f32", case.name)));
        assert_eq!(input.len(), case.channels * case.frames, "{}", case.name);

        let channels = planar(&input, case.channels, case.frames);

        let analysis = analyse(channels[0]).unwrap();
        assert_eq!(
            analysis.cutoff_hz, case.detected_cutoff,
            "{}: detected cutoff",
            case.name
        );
        let expect_full_band = case.true_cutoff.is_none();
        assert_eq!(
            analysis.looks_full_band(),
            expect_full_band,
            "{}: full-band verdict, ratio {:.3e}",
            case.name,
            analysis.high_band_ratio
        );

        let preprocessor = Preprocessor::new(case.detected_cutoff).unwrap();
        for (index, channel) in channels.iter().enumerate() {
            let filtered = preprocessor.filtered(channel).unwrap();
            let offset = index * case.frames;
            let expected = &expected_filtered[offset..offset + case.frames];
            let error = max_abs_diff(&filtered, expected);
            report(&case.name, "filtered", index, error, case.rms_out);
            assert!(
                error < 1e-5,
                "{} channel {index}: filtered stage off by {error}",
                case.name
            );

            let out = preprocessor.band_limited(&filtered);
            let expected = &expected_out[offset..offset + case.frames];
            let error = max_abs_diff(&out, expected);
            report(&case.name, "out", index, error, case.rms_out);
            assert!(
                error < 1e-5,
                "{} channel {index}: round trip off by {error}",
                case.name
            );
        }

        assert_eq!(case.frames, manifest.window, "{}: window length", case.name);
    }
}

/// One line per stage, in dB relative to the fixture's RMS, so a passing run still says
/// how much headroom the tolerance has.
fn report(name: &str, stage: &str, channel: usize, error: f64, rms: f64) {
    let db = if error > 0.0 {
        20.0 * (error / rms).log10()
    } else {
        f64::NEG_INFINITY
    };
    println!("{name:16} {stage:8} ch{channel} max error {error:.3e} ({db:.1} dB)");
}
