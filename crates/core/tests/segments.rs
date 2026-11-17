//! A track rendered as independent ranges equals one whole-track render, sample for
//! sample: the seed schedule, the crossfade tail and the end of the track all carry over.

use flacify_core::pipeline::Tail;
use flacify_core::upscaler::mocks::{Delay, Gain, Identity, Noise};
use flacify_core::{Block, Error, Pipeline, Upscaler};

const CHANNELS: usize = 2;
const WINDOW: usize = 256;
const OVERLAP: usize = 64;
const STEP: usize = WINDOW - OVERLAP;
const SEED: u64 = 0x51ED_2701;
/// Deliberately not a divisor of the window, the step or any segment length.
const PUSH_CHUNK: usize = 777;

fn ramp(frames: usize) -> Block {
    let mut data = Vec::with_capacity(CHANNELS * frames);
    for channel in 0..CHANNELS {
        for frame in 0..frames {
            let n = (channel as f32 * 0.37) + frame as f32 * 0.618_034;
            data.push((n.fract()) * 2.0 - 1.0);
        }
    }
    Block::from_planar(CHANNELS, data).expect("planar")
}

fn flatten(blocks: &[Block]) -> Block {
    let frames: usize = blocks.iter().map(Block::frames).sum();
    let mut data = Vec::with_capacity(CHANNELS * frames);
    for channel in 0..CHANNELS {
        for block in blocks {
            data.extend_from_slice(block.channel(channel));
        }
    }
    Block::from_planar(CHANNELS, data).expect("planar")
}

fn push_range<U: Upscaler>(
    pipeline: &mut Pipeline<U>,
    input: &Block,
    start: usize,
    end: usize,
) -> Vec<Block> {
    let mut out = Vec::new();
    for at in (start..end).step_by(PUSH_CHUNK) {
        let to = (at + PUSH_CHUNK).min(end);
        out.extend(pipeline.push(&input.slice(at, to)).expect("push"));
    }
    out
}

/// One render of the whole track, in arbitrary push chunks.
fn render_whole<U: Upscaler>(engine: U, input: &Block) -> Vec<Block> {
    let mut pipeline = Pipeline::new(engine, CHANNELS, WINDOW, OVERLAP, SEED).expect("pipeline");
    let mut out = push_range(&mut pipeline, input, 0, input.frames());
    out.extend(pipeline.finish().expect("finish"));
    out
}

/// Render the track as ranges of `steps_per_segment` windows, carrying the tail from one
/// range to the next. `whole` supplies the window count and each window's emitted length.
fn render_segmented<U: Upscaler>(
    engine: &mut impl FnMut() -> U,
    input: &Block,
    whole: &[Block],
    steps_per_segment: usize,
) -> Vec<Block> {
    let total_windows = whole.len();
    let mut segments = Vec::new();
    let mut tail: Option<Tail> = None;
    let mut at_window = 0;
    loop {
        let end_window = (at_window + steps_per_segment).min(total_windows);
        let mut pipeline =
            Pipeline::new(engine(), CHANNELS, WINDOW, OVERLAP, SEED).expect("pipeline");
        pipeline
            .start_at(at_window as u64, tail.take())
            .expect("start_at");
        let start = at_window * STEP;
        // A range that is not the last one also supplies the lookahead its last window
        // reads; the last one runs to the end of the track and finishes.
        let is_last = end_window == total_windows;
        let end = if is_last {
            input.frames()
        } else {
            end_window * STEP + OVERLAP
        };
        let mut out = push_range(&mut pipeline, input, start, end);
        if is_last {
            out.extend(pipeline.finish().expect("finish"));
            segments.extend(out);
            break;
        }
        let emitted: usize = out.iter().map(Block::frames).sum();
        let expected: usize = whole[at_window..end_window].iter().map(Block::frames).sum();
        assert_eq!(
            emitted, expected,
            "{steps_per_segment}-step range at window {at_window}: must emit its steps, not its tail"
        );
        tail = Some(pipeline.end_range().expect("end_range"));
        segments.extend(out);
        at_window = end_window;
    }
    segments
}

fn assert_same_samples(expected: &Block, got: &Block, frames: usize, label: &str) {
    assert_eq!(expected.frames(), frames, "{label}: whole-track frames");
    assert_eq!(got.frames(), frames, "{label}: segmented frames");
    for channel in 0..CHANNELS {
        for (frame, (a, b)) in expected
            .channel(channel)
            .iter()
            .zip(got.channel(channel))
            .enumerate()
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{label}: channel {channel} frame {frame}"
            );
        }
    }
}

fn assert_segments_match<U: Upscaler>(name: &str, mut engine: impl FnMut() -> U, input: &Block) {
    let whole = render_whole(engine(), input);
    let whole_block = flatten(&whole);
    for steps_per_segment in [1, 7, 13] {
        let segments = render_segmented(&mut engine, input, &whole, steps_per_segment);
        let segment_block = flatten(&segments);
        assert_same_samples(
            &whole_block,
            &segment_block,
            input.frames(),
            &format!("{name}, {steps_per_segment}-step segments"),
        );
    }
}

#[test]
fn given_a_track_rendered_in_segments_when_compared_with_one_render_then_every_sample_matches() {
    // An exact multiple of the step, then three lengths that end inside the last window:
    // below the overlap, exactly at it, and above it.
    for frames in [
        13 * STEP,
        13 * STEP + 40,
        13 * STEP + OVERLAP,
        13 * STEP + 150,
    ] {
        let input = ramp(frames);
        assert_segments_match("identity", || Identity, &input);
        assert_segments_match("gain", || Gain(0.5), &input);
        assert_segments_match("delay", || Delay(10), &input);
        assert_segments_match("noise", || Noise(0.1), &input);
    }
}

#[test]
fn given_a_range_started_cold_when_compared_with_the_whole_track_then_only_the_crossfade_differs() {
    let input = ramp(13 * STEP + 150);
    let whole = flatten(&render_whole(Noise(0.1), &input));

    let mut pipeline =
        Pipeline::new(Noise(0.1), CHANNELS, WINDOW, OVERLAP, SEED).expect("pipeline");
    pipeline.start_at(7, None).expect("start_at");
    let mut cold = push_range(&mut pipeline, &input, 7 * STEP, input.frames());
    cold.extend(pipeline.finish().expect("finish"));
    let cold = flatten(&cold);

    let start = 7 * STEP;
    assert_eq!(cold.frames(), input.frames() - start);
    for channel in 0..CHANNELS {
        // Without the tail the first overlap frames blend with silence, so they differ.
        assert_ne!(
            &cold.channel(channel)[..OVERLAP],
            &whole.channel(channel)[start..start + OVERLAP],
            "channel {channel}: the cold range should differ inside the crossfade"
        );
        for (frame, (a, b)) in cold.channel(channel)[OVERLAP..]
            .iter()
            .zip(&whole.channel(channel)[start + OVERLAP..])
            .enumerate()
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "channel {channel} frame {}: the seed schedule must match",
                start + OVERLAP + frame
            );
        }
    }
}

#[test]
fn given_an_intermediate_range_when_ended_then_it_hands_the_tail_on_instead_of_ending_the_track() {
    let input = ramp(13 * STEP);
    let mut pipeline = Pipeline::new(Identity, CHANNELS, WINDOW, OVERLAP, SEED).expect("pipeline");
    pipeline.start_at(0, None).expect("start_at");
    let out = push_range(&mut pipeline, &input, 0, 3 * STEP + OVERLAP);

    let emitted: usize = out.iter().map(Block::frames).sum();
    assert_eq!(
        emitted,
        3 * STEP,
        "an intermediate range emits exactly three steps, not the held tail"
    );
    let tail = pipeline.end_range().expect("end_range");
    assert_eq!(tail.samples().frames(), OVERLAP);
    assert_eq!(tail.next_window(), 3);
    assert_eq!(pipeline.end_range(), Err(Error::Finished));
    assert_eq!(pipeline.push(&input.slice(0, WINDOW)), Err(Error::Finished));
}

#[test]
fn given_a_range_that_stops_mid_window_when_ended_then_it_is_rejected() {
    let input = ramp(13 * STEP);
    let mut pipeline = Pipeline::new(Identity, CHANNELS, WINDOW, OVERLAP, SEED).expect("pipeline");
    pipeline.start_at(0, None).expect("start_at");
    pipeline.push(&input.slice(0, 3 * STEP)).expect("push");

    assert_eq!(
        pipeline.end_range(),
        Err(Error::RangeNotAligned {
            buffered: STEP,
            expected: OVERLAP,
        })
    );
}

#[test]
fn given_a_range_that_rendered_no_window_when_ended_then_it_is_rejected() {
    let input = ramp(13 * STEP);
    let mut pipeline = Pipeline::new(Identity, CHANNELS, WINDOW, OVERLAP, SEED).expect("pipeline");
    pipeline.start_at(0, None).expect("start_at");
    pipeline.push(&input.slice(0, OVERLAP)).expect("push");

    assert_eq!(pipeline.end_range(), Err(Error::EmptyRange));
}

#[test]
fn given_a_pipeline_mid_range_when_started_again_then_it_is_rejected() {
    let input = ramp(13 * STEP);

    // A held tail.
    let mut with_tail = Pipeline::new(Identity, CHANNELS, WINDOW, OVERLAP, SEED).expect("pipeline");
    with_tail.start_at(0, None).expect("start_at");
    with_tail.push(&input.slice(0, WINDOW)).expect("push");
    assert_eq!(with_tail.start_at(1, None), Err(Error::MidRange));

    // Buffered input only.
    let mut with_input =
        Pipeline::new(Identity, CHANNELS, WINDOW, OVERLAP, SEED).expect("pipeline");
    with_input.start_at(0, None).expect("start_at");
    with_input.push(&input.slice(0, 10)).expect("push");
    assert_eq!(with_input.start_at(0, None), Err(Error::MidRange));
}

#[test]
fn given_a_tail_from_another_window_when_starting_then_it_is_rejected() {
    let input = ramp(13 * STEP);
    let mut pipeline = Pipeline::new(Identity, CHANNELS, WINDOW, OVERLAP, SEED).expect("pipeline");
    pipeline.start_at(0, None).expect("start_at");
    push_range(&mut pipeline, &input, 0, 3 * STEP + OVERLAP);
    let tail = pipeline.end_range().expect("end_range");

    let mut next = Pipeline::new(Identity, CHANNELS, WINDOW, OVERLAP, SEED).expect("pipeline");
    assert_eq!(
        next.start_at(4, Some(tail)),
        Err(Error::TailFromWindow {
            start_window: 4,
            tail_window: 3,
        })
    );
}

#[test]
fn given_a_tail_from_another_window_size_when_starting_then_it_is_rejected() {
    // Window 512, overlap 128: a tail twice as long as the one a 256/64 pipeline holds.
    let mut wide = Pipeline::new(Identity, CHANNELS, 512, 128, SEED).expect("pipeline");
    wide.push(&ramp(512)).expect("push");
    let tail = wide.end_range().expect("end_range");
    assert_eq!(tail.next_window(), 1);

    let mut pipeline = Pipeline::new(Identity, CHANNELS, WINDOW, OVERLAP, SEED).expect("pipeline");
    assert_eq!(
        pipeline.start_at(1, Some(tail)),
        Err(Error::TailShape {
            expected_channels: CHANNELS,
            expected_frames: OVERLAP,
            got_channels: CHANNELS,
            got_frames: 128,
        })
    );
}
