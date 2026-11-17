//! Segmented playback: the audio callback reads decoded FLAC segments, not a track.
//!
//! One reader answers for both A/B sides of a complete cache entry: the source tier's dry
//! audio and the entry's rendered audio, on the same segment grid. A worker thread decodes
//! the segment at the playhead and the one after it, keeps at most three resident (the
//! current, the next and the previous) and publishes each through `ArcSwap`, so the callback
//! loads an `Arc` and indexes it: no allocation, no lock, no file I/O.
//!
//! A segment that is not decoded yet is not an error. The rendered side falls back to the
//! dry one; a playhead past both (just after a long seek) holds still and writes silence
//! rather than repeating stale frames or blocking the device.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, Thread};
use std::time::Duration;

use anyhow::{Context, Result};
use arc_swap::{ArcSwapOption, Guard};

/// The most segments the reader keeps decoded: the one at the playhead, the next one and
/// the one before it.
pub const RESIDENT_LIMIT: usize = 3;

/// How long it waits before retrying a segment that failed to decode, so a corrupt file
/// cannot spin the worker at the fill rate.
const RETRY_FAILED: Duration = Duration::from_secs(2);

/// One side of playback as whole segments on disk: the source tier's dry audio, or a
/// complete cache entry's rendered audio.
pub trait SegmentSource: Send + Sync {
    /// Frames in the whole track.
    fn frames(&self) -> usize;

    /// Frames in every segment but the last.
    fn frames_per_segment(&self) -> usize;

    /// Planar 48 kHz stereo for segment `index`, whole, in its own frame range.
    fn read_segment(&self, index: usize) -> Result<Vec<f32>>;

    /// Frames in segment `index`, or `None` past the last one.
    fn segment_frames(&self, index: usize) -> Option<usize> {
        let start = index.checked_mul(self.frames_per_segment())?;
        (start < self.frames()).then(|| self.frames_per_segment().min(self.frames() - start))
    }
}

impl SegmentSource for crate::source::Entry {
    fn frames(&self) -> usize {
        self.meta().frames
    }

    fn frames_per_segment(&self) -> usize {
        self.meta().frames_per_segment
    }

    /// `read_range` decodes only the segments a range touches, so a range that is exactly
    /// one segment decodes exactly one.
    fn read_segment(&self, index: usize) -> Result<Vec<f32>> {
        let frames = self
            .segment_frames(index)
            .with_context(|| format!("the source tier has no segment {index}"))?;
        self.read_range(index * self.frames_per_segment(), frames)
    }
}

impl SegmentSource for crate::cache::Segments {
    fn frames(&self) -> usize {
        self.meta().frames
    }

    fn frames_per_segment(&self) -> usize {
        self.meta().frames_per_segment
    }

    fn read_segment(&self, index: usize) -> Result<Vec<f32>> {
        crate::cache::Segments::read_segment(self, index)
    }
}

/// One decoded segment: both sides of the same frame range, planar stereo.
pub struct Resident {
    index: usize,
    start: usize,
    frames: usize,
    dry: Arc<[f32]>,
    wet: Option<Arc<[f32]>>,
}

impl Resident {
    /// `dry` is the source's own audio and `wet` the rendered side when it has been
    /// decoded, both `2 * frames` planar samples.
    pub fn new(
        index: usize,
        start: usize,
        frames: usize,
        dry: Arc<[f32]>,
        wet: Option<Arc<[f32]>>,
    ) -> Self {
        debug_assert_eq!(dry.len(), frames * 2);
        debug_assert!(match &wet {
            Some(wet) => wet.len() == frames * 2,
            None => true,
        });
        Self {
            index,
            start,
            frames,
            dry,
            wet,
        }
    }

    /// Position in the track.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Frame the segment starts at.
    pub fn start(&self) -> usize {
        self.start
    }

    /// Frames in the segment.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// One sample of the source's own audio, `offset` frames into the segment.
    pub fn dry(&self, offset: usize, channel: usize) -> f32 {
        self.dry[channel * self.frames + offset]
    }

    /// One sample of the rendered audio, or the source's own when this segment has no
    /// rendered side yet: the passthrough the frontier path already falls back to.
    pub fn wet(&self, offset: usize, channel: usize) -> f32 {
        match &self.wet {
            Some(wet) => wet[channel * self.frames + offset],
            None => self.dry(offset, channel),
        }
    }

    /// Whether the rendered side of this segment has been decoded.
    pub fn has_wet(&self) -> bool {
        self.wet.is_some()
    }
}

/// Reads segments on demand and publishes the decoded ones to the audio callback.
pub struct Reader {
    dry: Arc<dyn SegmentSource>,
    wet: Option<Arc<dyn SegmentSource>>,
    frames: usize,
    frames_per_segment: usize,
    slots: [ArcSwapOption<Resident>; RESIDENT_LIMIT],
    /// The segment index playback last asked for, which the worker follows.
    requested: AtomicUsize,
    /// Set when the last owner drops the reader; the worker returns at its next check.
    stopping: AtomicBool,
    /// The worker itself, so a request can wake it without a lock. `unpark` before `park`
    /// leaves a token, so a request that lands between a load and the park is not lost.
    worker: OnceLock<Thread>,
}

impl Reader {
    /// Start reading `dry` (the source tier) and, when a complete cache entry backs the
    /// track, `wet` (the rendered side). Both must cover the same frames in segments of
    /// the same length, because a fill reads both at one index.
    pub fn open(
        dry: Arc<dyn SegmentSource>,
        wet: Option<Arc<dyn SegmentSource>>,
    ) -> Result<Arc<Self>> {
        let frames = dry.frames();
        anyhow::ensure!(frames > 0, "a segmented track needs at least one frame");
        let frames_per_segment = dry.frames_per_segment();
        anyhow::ensure!(
            frames_per_segment > 0,
            "a segmented track needs a segment length"
        );
        if let Some(wet) = wet.as_ref() {
            anyhow::ensure!(
                wet.frames() == frames,
                "the rendered side holds {} frames, the source {}",
                wet.frames(),
                frames
            );
            anyhow::ensure!(
                wet.frames_per_segment() == frames_per_segment,
                "the rendered side uses {}-frame segments, the source {frames_per_segment}",
                wet.frames_per_segment()
            );
        }

        let reader = Arc::new(Self::assemble(dry, wet));
        // The first segments are decoded before the device starts, so the first fill has
        // audio instead of the silence a not-yet-resident segment produces.
        reader
            .fill_window(0)
            .context("decoding the first segment")?;
        let weak = Arc::downgrade(&reader);
        let handle = thread::Builder::new()
            .name("flacify-segments".to_string())
            .spawn(move || work(weak))
            .context("starting the segment reader")?;
        let _ = reader.worker.set(handle.thread().clone());
        Ok(reader)
    }

    fn assemble(dry: Arc<dyn SegmentSource>, wet: Option<Arc<dyn SegmentSource>>) -> Self {
        let frames = dry.frames();
        let frames_per_segment = dry.frames_per_segment();
        Self {
            dry,
            wet,
            frames,
            frames_per_segment,
            slots: std::array::from_fn(|_| ArcSwapOption::empty()),
            requested: AtomicUsize::new(0),
            stopping: AtomicBool::new(false),
            worker: OnceLock::new(),
        }
    }

    /// Frames in the whole track.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Frames in every segment but the last.
    pub fn frames_per_segment(&self) -> usize {
        self.frames_per_segment
    }

    /// Whether a complete cache entry backs the rendered side.
    pub fn has_rendered(&self) -> bool {
        self.wet.is_some()
    }

    /// Ask for the segment covering `frame`. Called from the audio callback: one atomic
    /// exchange and at most one thread wake, never a lock.
    pub fn request(&self, frame: usize) {
        let index = (frame / self.frames_per_segment).min(self.last_index());
        if self.requested.swap(index, Ordering::AcqRel) == index {
            return;
        }
        if let Some(worker) = self.worker.get() {
            worker.unpark();
        }
    }

    /// The resident segment covering `frame`, or an empty slot when the worker has not
    /// caught up. One atomic load per slot, no allocation: the callback's read path.
    pub fn segment_at(&self, frame: usize) -> Guard<Option<Arc<Resident>>> {
        self.segment(frame / self.frames_per_segment())
    }

    fn last_index(&self) -> usize {
        self.frames.saturating_sub(1) / self.frames_per_segment
    }

    fn segment(&self, index: usize) -> Guard<Option<Arc<Resident>>> {
        for slot in &self.slots {
            let guard = slot.load();
            if guard
                .as_ref()
                .is_some_and(|segment| segment.index() == index)
            {
                return guard;
            }
        }
        Guard::from(None)
    }

    /// The segment `index` is resident with everything playback can use from it: the dry
    /// side always, and the rendered side when a cache entry backs the track.
    fn is_ready(&self, index: usize) -> bool {
        let guard = self.segment(index);
        guard
            .as_ref()
            .is_some_and(|segment| self.wet.is_none() || segment.has_wet())
    }

    fn empty_slot(&self) -> Option<&ArcSwapOption<Resident>> {
        self.slots.iter().find(|slot| slot.load_full().is_none())
    }

    /// Drop every resident segment outside `[first, last]`, before anything is decoded, so
    /// the resident set can never grow past [`RESIDENT_LIMIT`].
    fn drop_outside(&self, first: usize, last: usize) {
        for slot in &self.slots {
            let stale = slot
                .load_full()
                .is_some_and(|segment| segment.index() < first || segment.index() > last);
            if stale {
                slot.store(None);
            }
        }
    }

    /// Bring the resident set to what segment `index` needs: that segment and the next one.
    /// The previous one is kept when it is already resident and never decoded on a move,
    /// so steady playback holds three and a seek holds two.
    fn fill_window(&self, index: usize) -> Result<()> {
        let index = index.min(self.last_index());
        let first = index.saturating_sub(1);
        let last = (index + 1).min(self.last_index());
        self.drop_outside(first, last);
        self.ensure(index)?;
        if index < last {
            self.ensure(index + 1)?;
        }
        Ok(())
    }

    /// Decode segment `index` unless it is already resident, publishing the dry side
    /// first so the callback can fall back to it while the rendered side is being read.
    fn ensure(&self, index: usize) -> Result<()> {
        if self.is_ready(index) {
            return Ok(());
        }
        let frames = self
            .dry
            .segment_frames(index)
            .with_context(|| format!("the track has no segment {index}"))?;
        let start = index * self.frames_per_segment;
        let dry: Arc<[f32]> = self.dry.read_segment(index)?.into();
        anyhow::ensure!(
            dry.len() == frames * 2,
            "segment {index} of the source holds {} samples, {frames} frames need {}",
            dry.len(),
            frames * 2
        );
        let slot = self
            .empty_slot()
            .context("no free slot for a segment inside the resident window")?;
        slot.store(Some(Arc::new(Resident::new(
            index,
            start,
            frames,
            Arc::clone(&dry),
            None,
        ))));

        let Some(wet_source) = self.wet.as_ref() else {
            return Ok(());
        };
        let wet: Arc<[f32]> = wet_source.read_segment(index)?.into();
        anyhow::ensure!(
            wet.len() == frames * 2,
            "segment {index} of the render holds {} samples, {frames} frames need {}",
            wet.len(),
            frames * 2
        );
        slot.store(Some(Arc::new(Resident::new(
            index,
            start,
            frames,
            dry,
            Some(wet),
        ))));
        Ok(())
    }
}

impl Drop for Reader {
    /// Wake the worker so it sees the flag at once; it holds only a weak reference, so it
    /// exits instead of keeping the reader alive.
    fn drop(&mut self) {
        self.stopping.store(true, Ordering::Release);
        if let Some(worker) = self.worker.get() {
            worker.unpark();
        }
    }
}

/// The prefetch worker: follow the requested segment, then sleep until the next request.
fn work(reader: std::sync::Weak<Reader>) {
    loop {
        let Some(reader) = reader.upgrade() else {
            return;
        };
        if reader.stopping.load(Ordering::Acquire) {
            return;
        }
        let index = reader.requested.load(Ordering::Acquire);
        let retry_in = match reader.fill_window(index) {
            Ok(()) => None,
            Err(error) => {
                tracing::warn!(%error, index, "prefetching a segment failed");
                Some(RETRY_FAILED)
            }
        };
        // Released before parking: the reader must be droppable while this thread sleeps.
        drop(reader);
        match retry_in {
            Some(timeout) => thread::park_timeout(timeout),
            None => thread::park(),
        }
    }
}

#[cfg(test)]
impl Resident {
    /// Sample data held, both planes and both sides.
    fn bytes(&self) -> usize {
        let wet = self.wet.as_ref().map_or(0, |wet| wet.len());
        (self.dry.len() + wet) * std::mem::size_of::<f32>()
    }
}

#[cfg(test)]
impl Reader {
    /// Block until the segment covering `frame` is decoded, or `timeout` passes.
    ///
    /// Test-only: the audio callback never waits, and the device does not either, because
    /// [`Reader::open`] decodes the first segments before it returns.
    pub(crate) fn wait_until_ready(&self, frame: usize, timeout: Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if self.is_ready(frame / self.frames_per_segment()) {
                return true;
            }
            if std::time::Instant::now() >= deadline {
                return false;
            }
            thread::sleep(Duration::from_millis(1));
        }
    }

    /// The segment indexes resident right now, ascending.
    pub(crate) fn resident_indexes(&self) -> Vec<usize> {
        let mut indexes: Vec<usize> = self
            .slots
            .iter()
            .filter_map(|slot| slot.load_full().map(|segment| segment.index()))
            .collect();
        indexes.sort_unstable();
        indexes
    }

    /// Bytes of decoded sample data resident right now, both sides.
    pub(crate) fn resident_bytes(&self) -> usize {
        self.slots
            .iter()
            .filter_map(|slot| slot.load_full())
            .map(|segment| segment.bytes())
            .sum()
    }
}

#[cfg(test)]
impl Reader {
    /// A reader over an already-decoded resident set, with no worker: what a test needs to
    /// hand the mixer a specific state.
    pub(crate) fn with_residents(
        dry: Arc<dyn SegmentSource>,
        wet: Option<Arc<dyn SegmentSource>>,
        residents: Vec<Resident>,
    ) -> Arc<Self> {
        let reader = Arc::new(Self::assemble(dry, wet));
        for segment in residents {
            reader
                .empty_slot()
                .expect("a free resident slot")
                .store(Some(Arc::new(segment)));
        }
        reader
    }
}

/// A segment source that computes its samples from the frame index, so a test can compare
/// a segment-backed mixer with a whole-buffer one. `offset` separates the two sides.
#[cfg(test)]
pub(crate) struct MemorySource {
    frames: usize,
    frames_per_segment: usize,
    offset: f32,
}

#[cfg(test)]
impl MemorySource {
    pub(crate) fn new(frames: usize, frames_per_segment: usize, offset: f32) -> Self {
        Self {
            frames,
            frames_per_segment,
            offset,
        }
    }
}

/// What [`MemorySource`] returns for one sample, shared with the whole-buffer side of a
/// test so the two differ only in how they are addressed.
#[cfg(test)]
pub(crate) fn memory_sample(frame: usize, channel: usize, offset: f32) -> f32 {
    frame as f32 + channel as f32 * 0.5 + offset
}

/// The interleaved whole-track buffer of a [`MemorySource`], for a whole-buffer mixer.
#[cfg(test)]
pub(crate) fn memory_interleaved(frames: usize, offset: f32, channels: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames * channels);
    for frame in 0..frames {
        for channel in 0..channels {
            out.push(memory_sample(frame, channel, offset));
        }
    }
    out
}

#[cfg(test)]
impl SegmentSource for MemorySource {
    fn frames(&self) -> usize {
        self.frames
    }

    fn frames_per_segment(&self) -> usize {
        self.frames_per_segment
    }

    fn read_segment(&self, index: usize) -> Result<Vec<f32>> {
        let frames = self
            .segment_frames(index)
            .with_context(|| format!("no segment {index}"))?;
        let start = index * self.frames_per_segment;
        let mut planar = Vec::with_capacity(frames * 2);
        for channel in 0..2 {
            for offset in 0..frames {
                planar.push(memory_sample(start + offset, channel, self.offset));
            }
        }
        Ok(planar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEGMENT: usize = 4_800;

    fn reader(frames: usize, has_rendered: bool) -> Arc<Reader> {
        let dry = Arc::new(MemorySource::new(frames, SEGMENT, 0.0));
        let wet: Option<Arc<dyn SegmentSource>> = has_rendered.then(|| {
            Arc::new(MemorySource::new(frames, SEGMENT, 1_000.0)) as Arc<dyn SegmentSource>
        });
        Reader::open(dry, wet).expect("a reader")
    }

    #[test]
    fn given_a_new_reader_when_asked_for_the_first_frame_then_its_segment_is_already_there() {
        let reader = reader(4 * SEGMENT, true);
        assert!(reader.segment_at(0).as_ref().is_some());
        assert_eq!(reader.resident_indexes(), [0, 1]);
        assert!(reader
            .segment_at(0)
            .as_ref()
            .expect("a resident segment")
            .has_wet());
    }

    #[test]
    fn given_playback_across_a_boundary_when_the_next_segment_is_asked_for_then_it_is_prefetched() {
        let reader = reader(6 * SEGMENT, true);
        reader.request(SEGMENT + 10);
        assert!(
            reader.wait_until_ready(SEGMENT + 10, Duration::from_secs(5)),
            "the worker must decode the requested segment"
        );
        assert!(
            reader.wait_until_ready(2 * SEGMENT, Duration::from_secs(5)),
            "the next segment must already be resident when the playhead reaches it"
        );
        assert_eq!(reader.resident_indexes(), [0, 1, 2]);
    }

    #[test]
    fn given_a_seek_to_another_segment_when_the_worker_catches_up_then_the_previous_one_is_dropped()
    {
        let reader = reader(20 * SEGMENT, true);
        reader.request(0);
        assert!(reader.wait_until_ready(0, Duration::from_secs(5)));

        reader.request(10 * SEGMENT + 100);
        assert!(reader.wait_until_ready(10 * SEGMENT + 100, Duration::from_secs(5)));
        assert_eq!(
            reader.resident_indexes(),
            [10, 11],
            "the segments around the old playhead must be gone"
        );
        assert!(
            reader.segment_at(0).as_ref().is_none(),
            "frame 0 must no longer be readable from the resident set"
        );
    }

    #[test]
    fn given_many_seeks_when_the_worker_follows_them_then_at_most_three_segments_are_resident() {
        let reader = reader(40 * SEGMENT, true);
        for index in [0, 1, 2, 3, 20, 19, 39, 0, 7] {
            reader.request(index * SEGMENT);
            assert!(
                reader.wait_until_ready(index * SEGMENT, Duration::from_secs(5)),
                "segment {index} must become resident"
            );
            let resident = reader.resident_indexes();
            assert!(
                resident.len() <= RESIDENT_LIMIT,
                "{} segments resident after a request for {index}: {resident:?}",
                resident.len()
            );
        }
    }

    #[test]
    fn given_a_segment_with_no_rendered_side_when_read_then_the_rendered_sample_is_the_original() {
        let segment = Resident::new(
            3,
            3 * SEGMENT,
            SEGMENT,
            Arc::from(vec![0.25f32; SEGMENT * 2]),
            None,
        );
        assert_eq!(segment.wet(17, 0), 0.25);
        assert_eq!(segment.wet(17, 1), 0.25);
        assert!(!segment.has_wet());
    }

    #[test]
    fn given_the_rendered_side_of_a_segment_when_read_then_both_planes_are_its_own() {
        let mut wet = vec![0.0f32; SEGMENT * 2];
        wet[5] = 0.5;
        wet[SEGMENT + 5] = -0.5;
        let segment = Resident::new(
            1,
            SEGMENT,
            SEGMENT,
            Arc::from(vec![0.0f32; SEGMENT * 2]),
            Some(Arc::from(wet)),
        );
        assert_eq!(segment.wet(5, 0), 0.5);
        assert_eq!(segment.wet(5, 1), -0.5);
    }

    #[test]
    fn given_a_source_with_two_sides_that_disagree_when_opened_then_it_is_an_error() {
        let dry = Arc::new(MemorySource::new(3 * SEGMENT, SEGMENT, 0.0));
        let wet = Arc::new(MemorySource::new(3 * SEGMENT + 1, SEGMENT, 1.0));
        let error = Reader::open(dry, Some(wet))
            .err()
            .expect("the sides must cover one track");
        assert!(
            format!("{error:#}").contains("rendered side holds"),
            "{error:#}"
        );
    }

    #[test]
    fn given_a_source_tier_and_a_cache_entry_for_the_same_audio_when_read_then_both_sides_are_their_own_samples(
    ) {
        // One 24-bit step: a sample already on this grid survives the FLAC round trip.
        const QUANTUM: f32 = 1.0 / 8_388_608.0;
        // The cache's own segment length for a 4 800-frame window with a 2 400 overlap.
        const SEGMENT_FRAMES: usize = 16_800;
        let frames = 3 * SEGMENT_FRAMES;
        let root = std::env::temp_dir().join(format!(
            "flacify-segments-files-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("a clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("creating the test directory");

        let mut planar = Vec::with_capacity(frames * 2);
        for channel in 0..2 {
            for frame in 0..frames {
                let step = (frame as i64 * (7_919 + channel as i64 * 3_217)) % 4_096 - 2_048;
                planar.push(step as f32 * QUANTUM);
            }
        }
        let source_path = root.join("source.wav");
        crate::wav::write(&source_path, 48_000, 2, &planar).expect("writing the source");
        let tier = crate::source::Tier::new(&root, SEGMENT_FRAMES).expect("a tier");
        let source = tier
            .build_or_reuse(&source_path, "hash")
            .expect("building the source tier");

        // The rendered side is the source inverted, so reading the wrong side is visible.
        let cache = crate::cache::Cache::new(root.clone());
        let fields = crate::cache::Fields {
            source_hash: "hash".to_string(),
            window: 4_800,
            overlap: 2_400,
            model_id: "model".to_string(),
            splice: false,
        };
        let render = crate::render::Render {
            planar: planar.iter().map(|sample| -sample).collect(),
            limit: crate::limit::LimitReport {
                peak_before: 0.0,
                peak_after: 0.0,
                limited_samples: 0,
            },
            device: "cpu".to_string(),
            gain_db: 0.0,
            lufs_input: 0.0,
            lufs_output: 0.0,
        };
        cache
            .store("key", &render, 12_000, &fields)
            .expect("storing");
        let rendered = cache
            .segments("key")
            .expect("opening")
            .expect("a complete entry");

        let reader = Reader::open(Arc::new(source), Some(Arc::new(rendered))).expect("a reader");
        let at = SEGMENT_FRAMES + 17;
        assert!(reader.wait_until_ready(at, Duration::from_secs(10)));
        let guard = reader.segment_at(at);
        let segment = guard.as_ref().expect("a resident segment");
        let offset = at - segment.start();
        assert_eq!(segment.dry(offset, 0), planar[at]);
        assert_eq!(segment.dry(offset, 1), planar[frames + at]);
        assert_eq!(segment.wet(offset, 0), -planar[at]);
        assert_eq!(segment.wet(offset, 1), -planar[frames + at]);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_source_segment_that_does_not_decode_when_opened_then_the_error_names_it() {
        struct Failing;

        impl SegmentSource for Failing {
            fn frames(&self) -> usize {
                3 * SEGMENT
            }

            fn frames_per_segment(&self) -> usize {
                SEGMENT
            }

            fn read_segment(&self, index: usize) -> Result<Vec<f32>> {
                anyhow::bail!("segment {index} is not readable")
            }
        }

        let error = Reader::open(Arc::new(Failing), None)
            .err()
            .expect("a source that cannot decode");
        let text = format!("{error:#}");
        assert!(text.contains("segment 0 is not readable"), "{text}");
    }

    #[test]
    fn given_a_track_of_one_segment_when_opened_then_the_next_segment_is_not_requested() {
        let reader = reader(SEGMENT, false);
        assert_eq!(reader.resident_indexes(), [0]);
        reader.request(SEGMENT + 10);
        assert!(reader.wait_until_ready(SEGMENT - 1, Duration::from_secs(5)));
    }

    /// A 60-minute entry at the shipped segment length: the memory the reader holds for it,
    /// against the 1.38 GB one whole-track buffer of each side costs.
    #[test]
    fn given_a_sixty_minute_track_when_playback_moves_then_resident_memory_fits_three_segments() {
        let frames_per_segment =
            crate::cache::frames_per_segment(crate::render::WINDOW, crate::render::OVERLAP)
                .expect("the shipped segment length");
        let frames = 60 * 60 * 48_000;
        let dry = Arc::new(MemorySource::new(frames, frames_per_segment, 0.0));
        let wet = Arc::new(MemorySource::new(frames, frames_per_segment, 0.0));
        let reader = Reader::open(dry, Some(wet)).expect("a reader");
        let segment_bytes = 2 * frames_per_segment * std::mem::size_of::<f32>();

        // A seek to 55:00: the playhead and the next segment, nothing else.
        let at = 55 * 60 * 48_000;
        reader.request(at);
        assert!(
            reader.wait_until_ready(at, Duration::from_secs(10)),
            "the segment at 55:00 must decode"
        );
        assert!(
            reader.wait_until_ready(at + frames_per_segment, Duration::from_secs(10)),
            "the segment after 55:00 must be prefetched"
        );
        let after_seek = reader.resident_bytes();
        assert_eq!(reader.resident_indexes().len(), 2);

        // Then playback crosses a boundary: the previous segment stays, so three are
        // resident, which is the most the reader ever holds.
        reader.request(at + frames_per_segment);
        assert!(reader.wait_until_ready(at + 2 * frames_per_segment, Duration::from_secs(10)));
        assert_eq!(reader.resident_indexes().len(), RESIDENT_LIMIT);
        let steady = reader.resident_bytes();
        println!(
            "resident: {after_seek} bytes after a seek, {steady} bytes in steady playback, \
             for a 60-minute track"
        );

        let whole_track = frames * 2 * std::mem::size_of::<f32>();
        assert_eq!(
            steady,
            RESIDENT_LIMIT * 2 * segment_bytes,
            "three segments, both sides"
        );
        assert!(
            steady < whole_track / 20,
            "{steady} bytes resident is not small against {whole_track}"
        );
    }
}
