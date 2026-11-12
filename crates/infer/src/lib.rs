//! The seam between the pipeline and a model, and the deterministic noise it is fed.
//!
//! FlashSR's sampler starts from noise. Upstream draws it with `torch.randn`, which is
//! neither seeded nor reproducible, so two runs on the same file do not produce the same
//! audio. This crate replaces that with noise derived from the window seed the pipeline
//! hands out (ADR-006), so a given file always renders the same way.

pub mod engine;
pub mod model;
pub mod noise;
pub mod python;

pub use engine::Engine;
pub use model::{FixedModel, IdentityModel, Model};
pub use noise::fill_normal;
