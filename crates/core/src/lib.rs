//! The parts of flacify that are testable without a GPU, an audio device or a model.
//!
//! The pipeline turns an audio stream into the same stream, enhanced, one window
//! at a time. Everything here is pure: it takes blocks and returns blocks. I/O,
//! decoding, playback and the ONNX session live in `flacify-app` and `flacify-infer`.

pub mod block;
pub mod dsp;
pub mod error;
pub mod pipeline;
pub mod policy;
pub mod preprocess;
pub mod upscaler;

pub use block::Block;
pub use error::{Error, Result};
pub use pipeline::Pipeline;
pub use policy::{plan, Mode, Plan, Probe, Warning};
pub use preprocess::Preprocessor;
pub use upscaler::Upscaler;

/// The model only accepts window lengths that are a multiple of 0.64 s at 48 kHz.
pub const GRANULE: usize = 30_720;
/// Native sample rate of the model and of the whole pipeline.
pub const SAMPLE_RATE: u32 = 48_000;
