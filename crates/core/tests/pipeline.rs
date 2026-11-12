//! Invariants the pipeline has to hold whatever the window, overlap or file length.

use flacify_core::pipeline::{curve_fall, curve_rise};
use flacify_core::upscaler::mocks::{Delay, FailAfter, Gain, Identity, Noise};
use flacify_core::{Block, Error, Pipeline};

fn ramp(channels: usize, frames: usize) -> Block {
    let data: Vec<f32> = (0..channels * frames)
        .map(|i| (((i * 7919) % 1000) as f32 / 500.0) - 1.0)
        .collect();
    Block::from_planar(channels, data).expect("planar")
}

fn run<U: flacify_core::Upscaler>(
    mut pipeline: Pipeline<U>,
    input: &Block,
    chunk: usize,
) -> Vec<Block> {
    let mut out = Vec::new();
    for start in (0..input.frames()).step_by(chunk.max(1)) {
        let end = (start + chunk).min(input.frames());
        out.extend(pipeline.push(&input.slice(start, end)).expect("push"));
    }
    out.extend(pipeline.finish().expect("finish"));
    out
}

fn total(blocks: &[Block]) -> usize {
    blocks.iter().map(Block::frames).sum()
}

fn flatten(blocks: &[Block], channels: usize, frames: usize) -> Block {
    let mut data = Vec::with_capacity(channels * frames);
    for channel in 0..channels {
        for block in blocks {
            data.extend_from_slice(block.channel(channel));
        }
    }
    Block::from_planar(channels, data).expect("planar")
}

#[test]
fn given_any_length_and_overlap_when_finished_then_the_output_length_equals_the_input() {
    let window = 1024;
    for overlap in [0, 1, 7, 512] {
        for frames in [1, 2, 511, 1023, 1024, 1025, 2048, 2049, 5000] {
            for chunk in [1, 333, 1024, 4096] {
                let input = ramp(2, frames);
                let pipeline = Pipeline::new(Identity, 2, window, overlap, 0).expect("pipeline");
                let out = run(pipeline, &input, chunk);
                assert_eq!(
                    total(&out),
                    frames,
                    "window={window} overlap={overlap} frames={frames} chunk={chunk}"
                );
            }
        }
    }
}

#[test]
fn given_an_identity_upscaler_when_rendered_then_the_signal_survives() {
    let window = 1024;
    for overlap in [0, 3, 512] {
        let input = ramp(2, 5000);
        let pipeline = Pipeline::new(Identity, 2, window, overlap, 0).expect("pipeline");
        let out = flatten(&run(pipeline, &input, 777), 2, input.frames());
        for channel in 0..2 {
            for (a, b) in input.channel(channel).iter().zip(out.channel(channel)) {
                assert!(
                    (a - b).abs() <= 4.0 * f32::EPSILON * a.abs().max(1.0),
                    "{a} != {b}"
                );
            }
        }
    }
}

#[test]
fn given_the_crossfade_ramps_when_summed_then_they_partition_unity() {
    for n in [1, 2, 3, 100, 1024] {
        let rise = curve_rise(n);
        let fall = curve_fall(n);
        for i in 0..n {
            assert!((rise[i] + fall[i] - 1.0).abs() < 1e-6, "n={n} i={i}");
        }
        assert!((rise[0] - 0.0).abs() < 1e-6 || n == 1);
        assert!((fall[0] - 1.0).abs() < 1e-6 || n == 1);
    }
}

#[test]
fn given_the_same_seed_when_rendered_twice_then_the_output_is_bit_identical() {
    let input = ramp(1, 3000);
    let render = |seed: u64| {
        let pipeline = Pipeline::new(Noise(0.1), 1, 1024, 64, seed).expect("pipeline");
        flatten(&run(pipeline, &input, 500), 1, input.frames())
    };
    assert_eq!(render(7), render(7));
    assert_ne!(render(7), render(8));
}

#[test]
fn given_a_gain_upscaler_when_rendered_then_the_gain_is_applied_everywhere() {
    let input = ramp(1, 4096);
    let pipeline = Pipeline::new(Gain(0.5), 1, 1024, 128, 0).expect("pipeline");
    let out = flatten(&run(pipeline, &input, 1024), 1, input.frames());
    for (a, b) in input.channel(0).iter().zip(out.channel(0)) {
        assert!((a * 0.5 - b).abs() < 1e-6, "{b}");
    }
}

#[test]
fn given_a_delaying_upscaler_when_rendered_then_the_signal_shifts_by_that_much() {
    // Inside one window, output[n] == input[n - delay].
    let delay = 10;
    let input = ramp(1, 1024);
    let pipeline = Pipeline::new(Delay(delay), 1, 1024, 0, 0).expect("pipeline");
    let out = flatten(&run(pipeline, &input, 4096), 1, input.frames());
    for n in delay..input.frames() {
        let a = input.channel(0)[n - delay];
        let b = out.channel(0)[n];
        assert!((a - b).abs() < 1e-6, "n={n}: {a} != {b}");
    }
}

#[test]
fn given_a_failing_upscaler_when_rendered_then_the_error_is_reported_not_hidden() {
    let input = ramp(1, 4096);
    let mut pipeline = Pipeline::new(FailAfter::new(1), 1, 1024, 0, 0).expect("pipeline");
    pipeline.push(&input.slice(0, 1024)).expect("first window");
    match pipeline.push(&input.slice(1024, 2048)) {
        Err(Error::Inference(message)) => assert!(message.contains("mock failure")),
        other => panic!("expected an inference error, got {other:?}"),
    }
}

#[test]
fn given_a_finished_pipeline_when_pushed_again_then_it_refuses() {
    let mut pipeline = Pipeline::new(Identity, 1, 256, 64, 0).expect("pipeline");
    pipeline.push(&ramp(1, 300)).expect("push");
    pipeline.finish().expect("finish");
    assert_eq!(pipeline.finish().expect("finish twice"), Vec::new());
    assert_eq!(pipeline.push(&ramp(1, 10)), Err(Error::Finished));
}

#[test]
fn given_a_window_too_small_when_built_then_it_is_rejected() {
    assert!(Pipeline::new(Identity, 1, 1, 0, 0).is_err());
    assert!(Pipeline::new(Identity, 0, 1024, 0, 0).is_err());
}

#[test]
fn given_an_overlap_larger_than_half_when_built_then_it_is_clamped() {
    let pipeline = Pipeline::new(Identity, 1, 1024, 100_000, 0).expect("pipeline");
    assert_eq!(pipeline.overlap(), 511);
    assert_eq!(pipeline.step(), 513);
}

#[test]
fn given_no_input_when_finished_then_nothing_comes_out() {
    let mut pipeline = Pipeline::new(Identity, 1, 1024, 64, 0).expect("pipeline");
    assert_eq!(pipeline.finish().expect("finish"), Vec::new());
}
