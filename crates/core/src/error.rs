use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error, PartialEq)]
pub enum Error {
    #[error("a block needs at least one channel")]
    NoChannels,

    #[error("planar data of {samples} samples does not divide into {channels} channels")]
    NotPlanar { samples: usize, channels: usize },

    #[error("window {window} is too small; it must be at least 2 frames")]
    WindowTooSmall { window: usize },

    #[error("window {window} is not a multiple of the {granule}-sample granule")]
    BadWindow { window: usize, granule: usize },

    #[error("input has {got} channels, pipeline expects {expected}")]
    ChannelMismatch { got: usize, expected: usize },

    #[error("upscaler returned {got_channels}x{got_frames}, expected {expected_channels}x{expected_frames}")]
    UpscalerShape {
        expected_channels: usize,
        expected_frames: usize,
        got_channels: usize,
        got_frames: usize,
    },

    #[error("inference failed: {0}")]
    Inference(String),

    #[error("signal of {len} samples is too short to filter (needs more than {min})")]
    SignalTooShort { len: usize, min: usize },

    #[error("filter design failed: {0}")]
    Filter(String),

    #[error("pipeline is finished and accepts no more input")]
    Finished,

    #[error("pipeline is mid-range; end the range or finish the track before starting another")]
    MidRange,

    #[error(
        "range did not end on a window boundary: {buffered} samples buffered, expected {expected}"
    )]
    RangeNotAligned { buffered: usize, expected: usize },

    #[error("range rendered no window, so it has no tail to carry")]
    EmptyRange,

    #[error(
        "tail belongs after window {tail_window}, but this range starts at window {start_window}"
    )]
    TailFromWindow { start_window: u64, tail_window: u64 },

    #[error("tail is {got_channels}x{got_frames}, expected {expected_channels}x{expected_frames}")]
    TailShape {
        expected_channels: usize,
        expected_frames: usize,
        got_channels: usize,
        got_frames: usize,
    },
}
