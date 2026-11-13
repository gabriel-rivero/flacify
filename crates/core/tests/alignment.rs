//! Does a sample stay where it was?
//!
//! The A/B switch is only honest if frame `k` of the generated signal is frame `k` of the
//! original. These tests pin that at the two places it can break without a model: the
//! pipeline's windowing and the preprocessing round trip.

use flacify_core::preprocess::Preprocessor;
use flacify_core::upscaler::mocks::Identity;
use flacify_core::{Block, Pipeline};

const WINDOW: usize = 245_760;

/// Where the largest excursion sits, in frames.
fn peak_index(samples: &[f32]) -> usize {
    samples
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .map(|(index, _)| index)
        .unwrap()
}

fn impulse_track(frames: usize, positions: &[usize]) -> Vec<f32> {
    let mut data = vec![0.0f32; frames];
    for position in positions {
        data[*position] = 0.9;
    }
    data
}

#[test]
fn given_an_impulse_when_it_passes_the_pipeline_unchanged_then_it_stays_at_its_index() {
    for position in [0usize, 1_000, 197_760, 400_000, 500_000] {
        let frames = 600_000;
        let track = impulse_track(frames, &[position]);
        let mut pipeline = Pipeline::new(Identity, 1, WINDOW, 48_000, 7).expect("a pipeline");
        let mut out = Vec::with_capacity(frames);
        let chunk = 48_000;
        let mut start = 0;
        while start < frames {
            let end = (start + chunk).min(frames);
            let block = Block::from_planar(1, track[start..end].to_vec()).expect("a block");
            for emitted in pipeline.push(&block).expect("a push") {
                out.extend_from_slice(emitted.channel(0));
            }
            start = end;
        }
        for emitted in pipeline.finish().expect("a finish") {
            out.extend_from_slice(emitted.channel(0));
        }

        assert_eq!(out.len(), frames, "length must be preserved exactly");
        assert_eq!(
            peak_index(&out),
            position,
            "an impulse at {position} came out elsewhere"
        );
    }
}

#[test]
fn given_an_impulse_when_it_passes_the_preprocessor_then_it_does_not_move() {
    // A cutoff measured on real music, so the resample round trip runs at a ratio that
    // does not divide evenly: 48 kHz down to 30.7 kHz and back.
    let pre = Preprocessor::new(15_351).expect("a preprocessor");
    for position in [10_000usize, 100_000, 200_000] {
        let mut channel = vec![0.0f32; WINDOW];
        channel[position] = 0.9;
        let processed = pre.process(&channel).expect("a process");
        assert_eq!(processed.len(), channel.len(), "length must be preserved");

        // The round trip delays by a known amount; compensate exactly as the engine does.
        let mut compensation: Vec<f32> = processed.iter().map(|s| *s as f32).collect();
        pre.compensate(&mut compensation, 1, channel.len());

        let peak = peak_index(&compensation);
        assert!(
            peak.abs_diff(position) <= 4,
            "a burst at {position} peaked at {peak} after compensating {} samples",
            pre.delay_samples()
        );
    }
}
