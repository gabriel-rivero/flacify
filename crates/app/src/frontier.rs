//! The render frontier: audio that exists, published one second at a time.
//!
//! The renderer runs ahead of playback and publishes finished seconds as immutable
//! chunks; the audio callback reads whatever has been published. Nothing is ever
//! mutated after publication, so the callback never takes a lock, never allocates and
//! never touches a frame the renderer is still writing.
//!
//! `arc-swap` gives that handoff in safe Rust: the writer replaces an empty slot with
//! a shared slice, the reader loads the slot and indexes it. The **read** side is what
//! matters — `load()` is a single atomic pointer read on the uncontended path.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwapOption;

/// One second at 48 kHz. Big enough that the callback resolves one slot per second of
/// audio, small enough that the frontier noticeably moves.
pub const CHUNK_FRAMES: usize = 48_000;

pub struct Frontier {
    /// Interleaved stereo, `CHUNK_FRAMES * 2` samples except the last chunk.
    slots: Vec<ArcSwapOption<Vec<f32>>>,
    published: AtomicUsize,
    frames: usize,
}

impl Frontier {
    pub fn new(frames: usize) -> Arc<Self> {
        let chunks = frames.div_ceil(CHUNK_FRAMES).max(1);
        Arc::new(Self {
            slots: (0..chunks).map(|_| ArcSwapOption::empty()).collect(),
            published: AtomicUsize::new(0),
            frames,
        })
    }

    /// Publish the chunk that starts at `index * CHUNK_FRAMES`. The samples are
    /// interleaved stereo at 48 kHz, exactly what the model produced.
    pub fn publish(&self, index: usize, interleaved: Vec<f32>) {
        if let Some(slot) = self.slots.get(index) {
            slot.store(Some(Arc::new(interleaved)));
            self.published.store(index + 1, Ordering::Release);
        }
    }

    /// How many chunks have been published, and therefore how many frames are readable.
    pub fn ready_frames(&self) -> usize {
        (self.published.load(Ordering::Acquire) * CHUNK_FRAMES).min(self.frames)
    }

    pub fn is_complete(&self) -> bool {
        self.ready_frames() >= self.frames
    }

    pub fn frames(&self) -> usize {
        self.frames
    }

    /// The slot for `frame`, or `None` when that second has not been rendered.
    fn slot(&self, frame: usize) -> Option<&ArcSwapOption<Vec<f32>>> {
        if frame >= self.frames {
            return None;
        }
        self.slots.get(frame / CHUNK_FRAMES)
    }

    /// Interleaved stereo for the frame range `[start, end)`, as far as it exists.
    ///
    /// Returns the guard and the slice, so a caller filling a buffer can hold it for the
    /// whole range instead of loading per sample.
    pub fn chunk_at(&self, frame: usize) -> (usize, arc_swap::Guard<Option<Arc<Vec<f32>>>>, usize) {
        let index = frame / CHUNK_FRAMES;
        let guard = match self.slot(frame) {
            Some(slot) => slot.load(),
            None => arc_swap::Guard::from(None),
        };
        (index, guard, (index + 1) * CHUNK_FRAMES)
    }

    /// One stereo frame. The window uses this to take a spectrum of generated audio;
    /// the audio callback does not (it reads whole chunks).
    pub fn frame(&self, frame: usize) -> Option<(f32, f32)> {
        let (_, guard, _) = self.chunk_at(frame);
        let chunk = guard.as_ref()?;
        let offset = (frame % CHUNK_FRAMES) * 2;
        Some((chunk[offset], chunk[offset + 1]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn given_nothing_published_when_read_then_the_frontier_is_empty() {
        let frontier = Frontier::new(4 * CHUNK_FRAMES);
        assert_eq!(frontier.ready_frames(), 0);
        assert!(frontier.frame(0).is_none());
        assert!(!frontier.is_complete());
    }

    #[test]
    fn given_a_published_chunk_when_read_then_its_frames_are_there_and_the_rest_is_not() {
        let frontier = Frontier::new(2 * CHUNK_FRAMES);
        let samples: Vec<f32> = (0..CHUNK_FRAMES).flat_map(|_| [0.25f32, -0.25]).collect();
        frontier.publish(0, samples);

        assert_eq!(frontier.ready_frames(), CHUNK_FRAMES);
        assert_eq!(frontier.frame(0), Some((0.25, -0.25)));
        assert_eq!(frontier.frame(CHUNK_FRAMES - 1), Some((0.25, -0.25)));
        assert!(frontier.frame(CHUNK_FRAMES).is_none());
    }

    #[test]
    fn given_the_last_chunk_when_published_then_ready_frames_is_capped_at_the_track() {
        let frames = CHUNK_FRAMES + 100;
        let frontier = Frontier::new(frames);
        frontier.publish(0, vec![0.0; CHUNK_FRAMES * 2]);
        assert_eq!(frontier.ready_frames(), CHUNK_FRAMES);
        assert!(!frontier.is_complete());
        frontier.publish(1, vec![0.0; 100 * 2]);
        assert_eq!(frontier.ready_frames(), frames);
        assert!(frontier.is_complete());
    }

    #[test]
    fn given_a_chunk_index_past_the_end_when_published_then_it_is_ignored() {
        let frontier = Frontier::new(CHUNK_FRAMES);
        frontier.publish(7, vec![0.0; 8]);
        assert_eq!(frontier.ready_frames(), 0);
    }
}
