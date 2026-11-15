//! What the background render is doing, in a form the window can read.
//!
//! The [`Frontier`](crate::frontier::Frontier) already knows how many seconds exist; this
//! is the part the frontier cannot express, because a renderer that failed and a renderer
//! that is merely slow both leave it empty. Without this a failed render looks like a slow
//! one: original playback, no generated curve, and nothing on stderr unless `RUST_LOG` was
//! raised past the default `ERROR` filter.
//!
//! The audio callback never touches this; only the render thread writes and only the
//! window reads, so a plain atomic and one arc-swapped string are enough.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwapOption;

#[derive(Default)]
pub struct RenderStatus {
    /// The renderer returned, successfully or not. No further seconds will appear.
    done: AtomicBool,
    /// The error that stopped it, if it stopped with one.
    failure: ArcSwapOption<String>,
}

impl RenderStatus {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// The renderer finished the whole track.
    pub fn finish(&self) {
        self.done.store(true, Ordering::Release);
    }

    /// The renderer stopped early; the message is what the window shows.
    pub fn fail(&self, error: impl Into<String>) {
        self.failure.store(Some(Arc::new(error.into())));
        self.finish();
    }

    pub fn is_done(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }

    pub fn failure(&self) -> Option<Arc<String>> {
        self.failure.load_full()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn given_a_new_status_when_read_then_it_is_still_running_and_has_no_error() {
        let status = RenderStatus::new();
        assert!(!status.is_done());
        assert!(status.failure().is_none());
    }

    #[test]
    fn given_a_failure_when_read_then_the_message_is_kept_and_the_render_is_done() {
        let status = RenderStatus::new();
        status.fail("the worker exited");
        assert!(status.is_done());
        assert_eq!(
            status.failure().as_deref().map(String::as_str),
            Some("the worker exited")
        );
    }

    #[test]
    fn given_a_finished_render_when_read_then_it_is_done_without_an_error() {
        let status = RenderStatus::new();
        status.finish();
        assert!(status.is_done());
        assert!(status.failure().is_none());
    }
}
