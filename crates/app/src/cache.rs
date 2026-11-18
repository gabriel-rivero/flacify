//! Render once, play forever: the segmented FLAC cache.
//!
//! Rendering a three-minute track costs about 95 s at RTF 0.54, and without a cache
//! every play pays it again. The cache is keyed by everything that can change the
//! output: the source file's bytes, the window and overlap, the pipeline version and the
//! model weights. A hit is exactly the audio a fresh render would have produced, and a
//! change to any of those misses instead of serving the wrong file.
//!
//! One entry is one track: `manifest.json` plus 24-bit FLAC segments of
//! `7 * (window - overlap)` frames each. The manifest is the source of truth for what
//! the entry holds; a segment it does not list is not part of the track.
//!
//! An entry is `partial` until its last segment lands, then `complete`. A partial entry
//! is a miss, but it stays on disk: the writer resumes it and adopts the segments a
//! killed render left. `store` still builds a whole render in a temporary directory and
//! renames it, so that path is never published half-written. A corrupt entry is an
//! error, not a miss: silence about one is how you end up debugging audio that plays the
//! wrong thing.
//!
//! Two ways out of an entry: [`Cache::load`] gathers every segment into one buffer, for the
//! paths that need a whole track, and [`Cache::segments`] opens the entry for reads of one
//! segment at a time, which is what playback of a complete entry does. Seeking needs
//! nothing from the cache beyond that: the playhead is an index into a segment.
//!
//! Eviction ([`Cache::evict`]) keeps the cache under a cap, oldest `last_used` first. It
//! never removes an entry something has open or part of one: an opener claims
//! `.claims/entry-<key>` and refreshes the claim on every segment read, eviction skips a
//! claim younger than an hour, and it re-checks the claim just before renaming. An opener
//! in that gap misses cleanly; one reading through it fails naming the missing file.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::render::Render;

/// Bump this when anything about the pipeline changes the samples: window, overlap,
/// preprocessing, the seed schedule, the limiter, or the layout a render is stored in.
///
/// 6 (2026-11-16): v5 stored each render window's planar chunk back to back, so the file
/// was block-major while the reader treated it as channel-major. A cache hit played the
/// right channel from the middle of the track. Every v5 entry has to be re-rendered.
///
/// 7 (2026-11-17): an entry is a manifest plus FLAC segments. Every v6 entry misses; its
/// directory is left for eviction rather than deleted here.
///
/// 8 (2026-11-17): the manifest's identity field is the source file's hash, not the
/// decoded samples' hash. Every v7 entry misses; its directory is left for eviction.
///
/// 9 (2026-11-17): the manifest records whether its segments are the whole track, so a
/// killed render resumes instead of starting over. Every v8 entry misses.
pub const CACHE_VERSION: &str = "flacify-cache-9";

/// Steps per segment. A segment has to start on a multiple of `window - overlap`, so its
/// length is derived from the settings and is not a wall-clock 30 s.
const STEPS_PER_SEGMENT: usize = 7;

/// The one file in an entry that is not audio.
const MANIFEST: &str = "manifest.json";

/// The source tier's directory inside the cache root.
const SOURCE_DIR: &str = "source";

/// The default cap eviction works against: 20 GB, from `architecture.md` and D9.
pub const DEFAULT_CAP_BYTES: u64 = 20_000_000_000;

/// The directory holding one claim file per entry and per source entry. They name the
/// process using the entry, live outside every directory eviction removes, and are not
/// counted.
pub(crate) const CLAIMS: &str = ".claims";

/// The infix a directory is renamed to before it is deleted, so a half-deleted entry is
/// never one a lookup can reach.
const EVICTING: &str = ".evict-";

/// One FLAC file of an entry, as the manifest describes it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Segment {
    /// Position in the track; also the number in the file name.
    pub index: usize,
    pub frames: usize,
    pub bytes: u64,
}

/// Whether an entry's segment table is the whole track yet. A partial entry is a miss,
/// but a resumable one: it exists so a killed render does not have to start over.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum State {
    /// Still being written, or killed before it could finish.
    #[default]
    Partial,
    Complete,
}

/// The manifest of one entry: where the audio came from, what the limiter did to it, and
/// which segments hold it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Meta {
    pub version: String,
    pub frames: usize,
    pub sample_rate: u32,
    #[serde(default)]
    pub frames_per_segment: usize,
    pub cutoff_hz: u32,
    /// What produced this entry: the source file's hash and the render settings. Stored
    /// so `find` can match them when the model is not installed and its id cannot be
    /// computed. Empty on entries written before these fields were recorded; `annotate`
    /// fills them on the next ready play.
    #[serde(default)]
    pub source_hash: String,
    #[serde(default)]
    pub window: usize,
    #[serde(default)]
    pub overlap: usize,
    #[serde(default)]
    pub splice: bool,
    /// Peak before and after the limiter ran, so the UI can still say what happened.
    pub peak_before: f32,
    pub peak_after: f32,
    pub limited_samples: usize,
    /// Loudness gain applied to the generated side, in dB; `None` until a writer records
    /// it. The renderer fixes it from segment 0 and every later segment needs it, so it is
    /// written before the first segment rather than only when the entry completes.
    #[serde(default)]
    pub gain_db: Option<f64>,
    /// The device the render ran on. Informational: a cache hit on another device is
    /// accepted, which is a deliberate prototype compromise (fp32 devices agree to
    /// within the gate, but they are not bit-identical).
    pub device: String,
    /// Whether the segment table is the whole track. A manifest written before this
    /// field existed reads as partial; only a writer marks an entry complete.
    #[serde(default)]
    pub state: State,
    /// Frames the listed segments cover. On a complete entry this is all of `frames`.
    #[serde(default)]
    pub rendered_frames: usize,
    /// The segments on disk, ordered by index. On a complete entry they are the track;
    /// a partial entry may have gaps, because the renderer follows the playhead.
    #[serde(default)]
    pub segments: Vec<Segment>,
    /// Unix seconds, for eviction: `store` sets both, `load` moves `last_used`.
    #[serde(default)]
    pub last_used: u64,
    #[serde(default)]
    pub created_at: u64,
}

/// What identifies an entry, in a form that can also be written into `manifest.json`.
///
/// The directory name is their hash, but a lookup on a machine without the model cannot
/// recompute that hash: the model id comes from files that are gone. The stored fields let
/// [`Cache::find`] match the source file and the settings instead.
#[derive(Clone, Debug)]
pub struct Fields {
    /// blake3 of the source file's bytes, first 32 hex characters.
    pub source_hash: String,
    pub window: usize,
    pub overlap: usize,
    pub model_id: String,
    pub splice: bool,
}

impl Fields {
    pub fn key(&self) -> String {
        key(
            &self.source_hash,
            self.window,
            self.overlap,
            &self.model_id,
            self.splice,
        )
    }
}

#[derive(Debug)]
pub struct Cached {
    pub planar: Vec<f32>,
    pub meta: Meta,
}

/// A complete entry opened for segment reads: the manifest, validated, and nothing decoded.
#[derive(Debug)]
pub struct Segments {
    /// Held for as long as this handle reads the entry, so eviction skips it, refreshed on
    /// every segment read because reading is what a play does.
    in_use: InUse,
    dir: PathBuf,
    meta: Meta,
}

impl Segments {
    /// What the entry records about itself.
    pub fn meta(&self) -> &Meta {
        &self.meta
    }

    /// Planar 48 kHz stereo for segment `index`, whole, in its own frame range.
    pub fn read_segment(&self, index: usize) -> Result<Vec<f32>> {
        self.in_use.refresh();
        let manifest = self.dir.join(MANIFEST);
        let expected =
            expected_segment_frames(self.meta.frames, self.meta.frames_per_segment, index)
                .with_context(|| format!("{} has no segment {index}", manifest.display()))?;
        Ok(decode_segment(&self.dir, &manifest, index, expected)?.planar)
    }

    /// The whole entry as one channel-major buffer, for the paths that need a track.
    pub fn read_all(&self) -> Result<Vec<f32>> {
        let mut left = Vec::with_capacity(self.meta.frames);
        let mut right = Vec::with_capacity(self.meta.frames);
        for segment in &self.meta.segments {
            let planar = self.read_segment(segment.index)?;
            left.extend_from_slice(&planar[..segment.frames]);
            right.extend_from_slice(&planar[segment.frames..]);
        }
        let mut planar = left;
        planar.extend_from_slice(&right);
        Ok(planar)
    }
}

/// What a writer records before its first segment: which render the entry belongs to,
/// how long it is, and the cutoff the render used.
pub struct EntrySpec {
    pub frames: usize,
    pub cutoff_hz: u32,
    pub fields: Fields,
}

/// What a finished render reports about the audio, recorded so a cache hit can still say
/// what the limiter and the loudness gain did.
pub struct Report {
    pub limit: crate::limit::LimitReport,
    pub device: String,
    pub gain_db: f64,
}

/// What a render has established before its entry is complete.
///
/// `finish` records these too, but a resumed render cannot recompute them for the segments
/// another process wrote: the gain comes from the first block of the track, and the
/// limiter's report covers samples the resumed pass never sees.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EntryProgress {
    /// The gain segment 0 fixed, or `None` until its first block has been measured.
    pub gain_db: Option<f64>,
    /// What the limiter did to the segments on disk so far.
    pub limit: crate::limit::LimitReport,
}

/// How long a claim stays believable without being refreshed. A play refreshes it on every
/// segment read, so this covers two cases: a process that died leaves its entry evictable
/// again within the hour, and a play paused for longer than that can be evicted, whose next
/// segment read then fails naming the missing file.
const STALE_CLAIM_SECS: u64 = 3_600;

/// Numbers claims within one process, so the last opener to claim an entry owns the record
/// and an earlier handle's drop does not clear it.
static NEXT_CLAIM: AtomicU64 = AtomicU64::new(1);

/// A claim that an entry is in use, held for as long as this value lives: `claims/<name>`
/// records `<pid> <claim> <unix seconds>`, and eviction skips an entry whose claim is
/// younger than [`STALE_CLAIM_SECS`]. The claim is advisory, not a kernel lock: eviction
/// re-checks it immediately before removing the directory, and an opener that arrives in
/// that gap finds the entry gone and misses cleanly.
#[derive(Debug)]
pub struct InUse {
    claim: Option<Claim>,
}

#[derive(Debug)]
struct Claim {
    path: PathBuf,
    pid: u32,
    id: u64,
}

/// One claim as its marker file holds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Marker {
    pid: u32,
    id: u64,
    secs: u64,
}

impl Marker {
    fn parse(text: &str) -> Option<Self> {
        let mut fields = text.split_whitespace();
        Some(Self {
            pid: fields.next()?.parse().ok()?,
            id: fields.next()?.parse().ok()?,
            secs: fields.next()?.parse().ok()?,
        })
    }

    /// A claim in the future counts as live: a clock that moved backwards should keep an
    /// entry, not delete it.
    fn is_live(&self) -> bool {
        self.secs.saturating_add(STALE_CLAIM_SECS) > now()
    }
}

impl InUse {
    /// The claim name for an entry directory: a staging directory claims the entry it
    /// publishes, so a sweep cannot remove a render mid-write.
    fn entry_name(dir: &Path) -> String {
        entry_claim(key_of(&file_name(dir)))
    }

    /// Record that an entry is in use, or an empty claim when the cache root cannot be
    /// written. An entry that cannot be claimed still plays: eviction then has only
    /// `last_used` to go on.
    pub(crate) fn acquire(claims: &Path, name: &str) -> Self {
        if let Err(error) = std::fs::create_dir_all(claims) {
            tracing::warn!(%error, dir = %claims.display(), "could not create the cache claim directory");
            return Self { claim: None };
        }
        let claim = Claim {
            path: claims.join(name),
            pid: std::process::id(),
            id: NEXT_CLAIM.fetch_add(1, Ordering::Relaxed),
        };
        if let Err(error) = claim.write(now()) {
            tracing::warn!(
                %error,
                marker = %claim.path.display(),
                "could not record that a cache entry is in use; eviction may remove it"
            );
            return Self { claim: None };
        }
        Self { claim: Some(claim) }
    }

    /// Say the entry is still in use. Every segment read does this, so a play that keeps
    /// reading keeps its claim.
    pub(crate) fn refresh(&self) {
        if let Some(claim) = &self.claim {
            let _ = claim.write(now());
        }
    }
}

impl Claim {
    fn write(&self, secs: u64) -> std::io::Result<()> {
        std::fs::write(&self.path, format!("{} {} {}\n", self.pid, self.id, secs))
    }
}

impl Drop for InUse {
    fn drop(&mut self) {
        let Some(claim) = &self.claim else {
            return;
        };
        // Only a claim still carrying this handle's number is ours to clear; a later opener
        // that overwrote it keeps its own.
        let Ok(text) = std::fs::read_to_string(&claim.path) else {
            return;
        };
        let Some(marker) = Marker::parse(&text) else {
            return;
        };
        if marker.pid == claim.pid && marker.id == claim.id {
            let _ = std::fs::remove_file(&claim.path);
        }
    }
}

/// Whether something claims the named entry. No marker means nobody does; a marker that
/// cannot be read counts as a claim, so eviction keeps what it could not inspect.
fn is_held_open(claims: &Path, name: &str) -> bool {
    let Ok(text) = std::fs::read_to_string(claims.join(name)) else {
        return false;
    };
    Marker::parse(&text).map_or(true, |marker| marker.is_live())
}

/// A staging directory a writer is filling, or one eviction renamed aside: no lookup reads
/// either, and the sweep removes both.
fn is_transient(name: &str) -> bool {
    name.ends_with(".tmp") || name.contains(EVICTING)
}

/// The load-bearing name, with an entry directory or path suffix stripped: `<key>.tmp` and
/// `<key>.evict-<pid>` both belong to the claim of `<key>`.
fn key_of(name: &str) -> &str {
    match name.find(EVICTING) {
        Some(at) => &name[..at],
        None => name.strip_suffix(".tmp").unwrap_or(name),
    }
}

/// One term for one concept: an entry's claim is named for its key, a tier's for its
/// source hash, and both modules build the name here.
pub(crate) fn entry_claim(key: &str) -> String {
    format!("entry-{key}")
}

pub(crate) fn source_claim(source_hash: &str) -> String {
    format!("source-{source_hash}")
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// What one part of the cache holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    pub entries: usize,
    pub bytes: u64,
}

impl Tally {
    fn add(&mut self, bytes: u64) {
        self.entries += 1;
        self.bytes += bytes;
    }
}

/// What the cache holds, split by what each part is.
///
/// Which bucket a directory falls in comes from its manifest; the bytes are what the
/// directory holds, so the cap is measured against the disk rather than against a claim.
/// Where the two disagree the entry is counted in `unaccounted_bytes` or
/// `overstated_entries` instead of being believed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    /// Complete entries: rendered segments, which are what a replay reads.
    pub rendered: Tally,
    /// Entries a killed render left resumable. A miss, but the work is not lost yet.
    pub partial: Tally,
    /// The decoded source tier, which any render can rebuild.
    pub source: Tally,
    /// Directories this build cannot read: another `CACHE_VERSION`, or no manifest at all.
    pub unknown: Tally,
    /// Leftover staging directories and half-deleted entries.
    pub stale: Tally,
    /// Bytes inside entries that their manifests do not name, such as an interrupted
    /// segment write or a stray file.
    pub unaccounted_bytes: u64,
    /// Entries whose manifests name more bytes than their files hold.
    pub overstated_entries: usize,
}

impl Usage {
    /// Everything on disk under the cache root, which is what the cap is measured against.
    pub fn total(&self) -> u64 {
        self.rendered.bytes
            + self.partial.bytes
            + self.source.bytes
            + self.unknown.bytes
            + self.stale.bytes
    }

    /// Complete and partial entries: the renders, as opposed to the source tier.
    pub fn entries(&self) -> usize {
        self.rendered.entries + self.partial.entries
    }
}

/// One directory eviction removes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Removal {
    /// Relative to the cache root, as the user reads it: `abc123`, `source/abc123`.
    pub name: String,
    pub bytes: u64,
    /// When the entry was last read, for ordering. A sweep has no clock: zero.
    pub last_used: u64,
}

/// What `evict` would remove for a cap, before anything is touched. It is advisory: a
/// holder can appear between the plan and the removal, so `evict` re-checks every entry and
/// reports what it skipped.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EvictionPlan {
    pub cap: u64,
    pub before: u64,
    /// Leftover staging directories and entries this build cannot read. These go first,
    /// because none of them can be played.
    pub sweeps: Vec<Removal>,
    /// Source tiers, one entry at a time, before the render they belong to.
    pub tiers: Vec<Removal>,
    /// Whole entries, least recently used first.
    pub entries: Vec<Removal>,
    /// Entries and tiers a live holder protects. Listed so the plan explains what it kept.
    pub open: Vec<Removal>,
    /// The total once every removal above has happened.
    pub after: u64,
}

/// What `evict` did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Eviction {
    pub swept: Vec<Removal>,
    pub tiers: Vec<Removal>,
    pub entries: Vec<Removal>,
    /// Planned removals a holder protected between the plan and the removal.
    pub skipped: Vec<Removal>,
    pub freed: u64,
    /// Measured after the removals.
    pub after: u64,
}

impl Eviction {
    /// How many directories the run removed.
    pub fn removals(&self) -> usize {
        self.swept.len() + self.tiers.len() + self.entries.len()
    }
}

/// One source tier entry, as the scan found it.
#[derive(Clone, Copy, Debug)]
struct TierFound {
    bytes: u64,
    created_at: u64,
    is_open: bool,
}

/// One entry the plan can remove.
#[derive(Debug)]
struct Candidate {
    key: String,
    bytes: u64,
    last_used: u64,
    /// The entry's source tier, when it has one nothing holds open.
    tier: Option<Removal>,
}

/// One pass over the cache root: the accounting, and everything eviction chooses from.
#[derive(Debug, Default)]
struct Scan {
    usage: Usage,
    sweeps: Vec<Removal>,
    orphan_tiers: Vec<Removal>,
    /// Complete and partial entries in one list: the least recently used goes first,
    /// whatever state it is in. A partial nobody has resumed for a month is the least
    /// valuable thing the cache holds.
    entries: Vec<Candidate>,
    open: Vec<Removal>,
}

pub struct Cache {
    root: PathBuf,
}

impl Cache {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// The directory holding every entry. The source tier lives in its `source/`.
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn entry_dir(&self, key: &str) -> PathBuf {
        self.root.join(key)
    }

    /// `None` when there is no entry, when it is partial, or when it was written by
    /// another version; an error when there is one and it cannot be read, because silence
    /// about a corrupt cache entry is how you end up debugging audio.
    pub fn load(&self, key: &str) -> Result<Option<Cached>> {
        self.load_dir(&self.entry_dir(key))
    }

    /// The complete entry at `key`, opened for segment reads instead of decoded whole.
    /// `None` on the same terms as [`load`](Self::load).
    pub fn segments(&self, key: &str) -> Result<Option<Segments>> {
        self.segments_dir(&self.entry_dir(key))
    }

    /// The complete entry for the source file and settings it was made from, ignoring
    /// which model rendered it. Used when the model is not installed: the id cannot be
    /// computed, and an existing render is better than a setup screen. Only entries that
    /// recorded their fields can match; `annotate` fills them on a ready play. A partial
    /// entry is a miss, so the search skips it.
    pub fn find_segments(&self, fields: &Fields) -> Result<Option<Segments>> {
        match self.matching_dir(fields)? {
            Some(dir) => self.segments_dir(&dir),
            None => Ok(None),
        }
    }

    /// The directory of the complete entry whose recorded fields match, in name order so a
    /// source rendered by more than one model resolves to the same entry on every run.
    fn matching_dir(&self, fields: &Fields) -> Result<Option<PathBuf>> {
        let Ok(entries) = std::fs::read_dir(&self.root) else {
            return Ok(None);
        };
        let mut dirs: Vec<PathBuf> = entries
            .flatten()
            .filter(|entry| !is_transient(&entry.file_name().to_string_lossy()))
            .filter(|entry| entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false))
            .map(|entry| entry.path())
            .collect();
        dirs.sort();
        for dir in dirs {
            let Ok(Some(meta)) = self.read_meta(&dir) else {
                continue;
            };
            if meta.version != CACHE_VERSION
                || meta.state != State::Complete
                || meta.source_hash != fields.source_hash
                || meta.window != fields.window
                || meta.overlap != fields.overlap
                || meta.splice != fields.splice
            {
                continue;
            }
            return Ok(Some(dir));
        }
        Ok(None)
    }

    /// Fill in the identity fields on an entry written before they were recorded. Best
    /// effort: a cache hit that cannot be annotated still plays.
    pub fn annotate(&self, key: &str, fields: &Fields) -> Result<()> {
        let dir = self.entry_dir(key);
        let _in_use = self.hold_open(&dir);
        let Some(mut meta) = self.read_meta(&dir)? else {
            return Ok(());
        };
        if !meta.source_hash.is_empty() {
            return Ok(());
        }
        meta.source_hash = fields.source_hash.clone();
        meta.window = fields.window;
        meta.overlap = fields.overlap;
        meta.splice = fields.splice;
        write_manifest(&dir, &meta)
    }

    fn load_dir(&self, dir: &Path) -> Result<Option<Cached>> {
        // Held across the read, so eviction cannot take the bytes out from under it. The
        // samples are in memory by the time this returns, so the claim ends there.
        let in_use = self.hold_open(dir);
        let Some(mut meta) = self.read_meta(dir)? else {
            return Ok(None);
        };
        if meta.version != CACHE_VERSION {
            return Ok(None);
        }
        // A half-written entry is a miss. Its segments stay for the writer to resume.
        if meta.state != State::Complete {
            return Ok(None);
        }
        let manifest = dir.join(MANIFEST);
        validate_manifest(&manifest, &meta)?;

        let planar = Segments {
            in_use,
            dir: dir.to_path_buf(),
            meta: meta.clone(),
        }
        .read_all()?;

        touch(dir, &mut meta);
        Ok(Some(Cached { planar, meta }))
    }

    /// The same checks as [`load_dir`](Self::load_dir), without decoding a sample. Every
    /// segment the manifest names has to be there; whether it decodes is checked when it
    /// is read.
    fn segments_dir(&self, dir: &Path) -> Result<Option<Segments>> {
        // The claim comes before the manifest, so an eviction either sees this opener or has
        // already removed the entry by the time the manifest is read, which then misses.
        let in_use = self.hold_open(dir);
        let Some(mut meta) = self.read_meta(dir)? else {
            return Ok(None);
        };
        if meta.version != CACHE_VERSION || meta.state != State::Complete {
            return Ok(None);
        }
        let manifest = dir.join(MANIFEST);
        validate_manifest(&manifest, &meta)?;
        for segment in &meta.segments {
            let path = dir.join(segment_name(segment.index));
            anyhow::ensure!(
                path.is_file(),
                "{} names a segment that is missing: {}",
                manifest.display(),
                path.display()
            );
        }
        touch(dir, &mut meta);
        Ok(Some(Segments {
            in_use,
            dir: dir.to_path_buf(),
            meta,
        }))
    }

    fn read_meta(&self, dir: &Path) -> Result<Option<Meta>> {
        let path = dir.join(MANIFEST);
        if !path.is_file() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let meta =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        Ok(Some(meta))
    }

    /// Publish a whole render as one segment per `7 * (window - overlap)` frames. It is
    /// written through the entry writer into `<key>.tmp` and renamed, so this path is
    /// never visible half-written.
    pub fn store(
        &self,
        key: &str,
        render: &Render,
        cutoff_hz: u32,
        fields: &Fields,
    ) -> Result<PathBuf> {
        anyhow::ensure!(
            render.planar.len() % 2 == 0,
            "a render of {} samples is not whole stereo frames",
            render.planar.len()
        );
        let frames = render.planar.len() / 2;
        anyhow::ensure!(frames > 0, "a cache entry needs at least one frame");
        let segment_frames = frames_per_segment(fields.window, fields.overlap)?;

        let temp_dir = self.root.join(format!("{key}.tmp"));
        clear(&temp_dir)?;
        let mut writer = self.begin_at(
            &temp_dir,
            &EntrySpec {
                frames,
                cutoff_hz,
                fields: fields.clone(),
            },
        )?;
        for (index, start) in (0..frames).step_by(segment_frames).enumerate() {
            let count = segment_frames.min(frames - start);
            writer.write_segment(index, &stereo_slice(&render.planar, frames, start, count))?;
        }
        writer.finish(&Report {
            limit: render.limit,
            device: render.device.clone(),
            gain_db: render.gain_db,
        })?;

        let final_dir = self.entry_dir(key);
        clear(&final_dir)?;
        std::fs::rename(&temp_dir, &final_dir)
            .with_context(|| format!("publishing {}", final_dir.display()))?;
        Ok(final_dir)
    }

    /// Start an entry, or resume the partial one at `key`; a complete entry is never
    /// reopened. Segments a killed writer left are adopted when they fit their index and
    /// removed when they do not, and a recorded segment is never rewritten.
    pub fn begin(&self, key: &str, spec: &EntrySpec) -> Result<EntryWriter> {
        self.begin_at(&self.entry_dir(key), spec)
    }

    /// The writer's one way in, wherever the entry is built: `store` hands it the
    /// staging directory, `begin` the entry directory.
    fn begin_at(&self, dir: &Path, spec: &EntrySpec) -> Result<EntryWriter> {
        anyhow::ensure!(spec.frames > 0, "a cache entry needs at least one frame");
        let frames_per_segment = frames_per_segment(spec.fields.window, spec.fields.overlap)?;
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;

        let mut meta = match self.read_meta(dir)? {
            Some(existing) => {
                anyhow::ensure!(
                    existing.version == CACHE_VERSION,
                    "{} was written by cache version {}",
                    dir.display(),
                    existing.version
                );
                anyhow::ensure!(
                    existing.state == State::Partial,
                    "{} is a complete entry, and a finished entry is not reopened",
                    dir.display()
                );
                validate_manifest(&dir.join(MANIFEST), &existing)?;
                anyhow::ensure!(
                    existing.frames == spec.frames
                        && existing.frames_per_segment == frames_per_segment
                        && existing.cutoff_hz == spec.cutoff_hz
                        && existing.source_hash == spec.fields.source_hash
                        && existing.window == spec.fields.window
                        && existing.overlap == spec.fields.overlap
                        && existing.splice == spec.fields.splice,
                    "{} holds a partial entry for another render of {} frames at {} Hz",
                    dir.display(),
                    existing.frames,
                    existing.cutoff_hz
                );
                // Resuming is using it, and the LRU clock has to say so.
                let mut existing = existing;
                existing.last_used = now();
                existing
            }
            None => {
                let created_at = now();
                Meta {
                    version: CACHE_VERSION.to_string(),
                    frames: spec.frames,
                    sample_rate: flacify_core::SAMPLE_RATE,
                    frames_per_segment,
                    cutoff_hz: spec.cutoff_hz,
                    source_hash: spec.fields.source_hash.clone(),
                    window: spec.fields.window,
                    overlap: spec.fields.overlap,
                    splice: spec.fields.splice,
                    peak_before: 0.0,
                    peak_after: 0.0,
                    limited_samples: 0,
                    gain_db: None,
                    device: String::new(),
                    state: State::Partial,
                    rendered_frames: 0,
                    segments: Vec::new(),
                    last_used: created_at,
                    created_at,
                }
            }
        };

        meta.segments = adopt_segments(dir, &meta)?;
        meta.rendered_frames = meta.segments.iter().map(|segment| segment.frames).sum();
        write_manifest(dir, &meta)?;
        Ok(EntryWriter {
            in_use: self.hold_open(dir),
            dir: dir.to_path_buf(),
            meta,
            is_finished: false,
        })
    }

    /// The claim name of an entry directory, staging or published.
    fn hold_open(&self, dir: &Path) -> InUse {
        InUse::acquire(&self.root.join(CLAIMS), &InUse::entry_name(dir))
    }

    /// What the cache holds, split by what each part is. See [`Usage`].
    pub fn usage(&self) -> Usage {
        self.scan().usage
    }

    /// What `evict` would remove to get under `cap` bytes, without touching anything.
    pub fn eviction_plan(&self, cap: u64) -> EvictionPlan {
        plan_eviction(cap, self.scan())
    }

    /// Carry out `plan`: remove what it names, skipping anything a holder protected since
    /// the plan was made. Every removal is one whole directory. Returns what was measured
    /// after the work, so the caller does not have to trust the arithmetic.
    pub fn evict(&self, plan: &EvictionPlan) -> Result<Eviction> {
        let claims = self.root.join(CLAIMS);
        let mut done = Eviction::default();
        for removal in &plan.sweeps {
            match self.remove_planned(removal, &claims)? {
                Outcome::Removed(removed) => done.swept.push(removed),
                Outcome::Held => done.skipped.push(removal.clone()),
                Outcome::Absent => {}
            }
        }
        for removal in &plan.tiers {
            match self.remove_planned(removal, &claims)? {
                Outcome::Removed(removed) => done.tiers.push(removed),
                Outcome::Held => done.skipped.push(removal.clone()),
                Outcome::Absent => {}
            }
        }
        for removal in &plan.entries {
            match self.remove_planned(removal, &claims)? {
                Outcome::Removed(removed) => done.entries.push(removed),
                Outcome::Held => done.skipped.push(removal.clone()),
                Outcome::Absent => {}
            }
        }
        done.freed = done
            .swept
            .iter()
            .chain(&done.tiers)
            .chain(&done.entries)
            .map(|removal| removal.bytes)
            .sum();
        done.after = self.usage().total();
        Ok(done)
    }

    /// One planned removal, re-checking the claim first: a holder that appeared since the
    /// plan is seen here. An opener in the gap between this check and the rename misses
    /// cleanly, because the rename takes the whole entry out of the namespace at once.
    fn remove_planned(&self, removal: &Removal, claims: &Path) -> Result<Outcome> {
        if is_held_open(claims, &removal_claim(&removal.name)) {
            return Ok(Outcome::Held);
        }
        let path = self.root.join(&removal.name);
        if !path.exists() {
            return Ok(Outcome::Absent);
        }
        let bytes = remove_tree(&path)?;
        Ok(Outcome::Removed(Removal {
            name: removal.name.clone(),
            bytes,
            last_used: removal.last_used,
        }))
    }

    /// One pass over the cache root. The accounting and the eviction candidates come from
    /// the same walk, so the plan can never spend bytes the total does not have.
    fn scan(&self) -> Scan {
        let claims = self.root.join(CLAIMS);
        let mut scan = Scan::default();
        let mut names = match std::fs::read_dir(&self.root) {
            Ok(entries) => entries
                .flatten()
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .collect::<Vec<_>>(),
            Err(_) => return scan,
        };
        names.sort();
        let tiers = self.scan_tiers(&claims, &mut scan);
        let mut claimed: BTreeSet<String> = BTreeSet::new();

        for name in names {
            if name == CLAIMS || name == SOURCE_DIR {
                continue;
            }
            let path = self.root.join(&name);
            let bytes = path_bytes(&path);
            if !path.is_dir() {
                // This build writes directories here. A file is not ours to delete: it is
                // counted so the total is honest and left alone.
                scan.usage.unknown.add(bytes);
                continue;
            }
            if is_transient(&name) {
                sweep(&mut scan, &claims, Unreadable::Stale, name, bytes);
                continue;
            }
            let meta = match self.read_meta(&path) {
                Ok(Some(meta)) if meta.version == CACHE_VERSION => meta,
                _ => {
                    sweep(&mut scan, &claims, Unreadable::Unknown, name, bytes);
                    continue;
                }
            };

            let recorded = segment_bytes(&meta.segments)
                + std::fs::metadata(path.join(MANIFEST))
                    .map(|meta| meta.len())
                    .unwrap_or(0);
            account_size(&mut scan.usage, bytes, recorded);
            match meta.state {
                State::Complete => scan.usage.rendered.add(bytes),
                State::Partial => scan.usage.partial.add(bytes),
            }
            let tier = self.tier_removal(&meta, &tiers, &mut claimed, &mut scan);
            if is_held_open(&claims, &entry_claim(&name)) {
                scan.open.push(Removal {
                    name,
                    bytes,
                    last_used: meta.last_used,
                });
                continue;
            }
            let candidate = Candidate {
                key: name,
                bytes,
                last_used: meta.last_used,
                tier,
            };
            scan.entries.push(candidate);
        }

        // A tier no entry renders from is the cheapest thing to lose: nothing reads it.
        for (hash, tier) in tiers {
            if claimed.contains(&hash) {
                continue;
            }
            let removal = Removal {
                name: format!("{SOURCE_DIR}/{hash}"),
                bytes: tier.bytes,
                last_used: tier.created_at,
            };
            if tier.is_open {
                scan.open.push(removal);
            } else {
                scan.orphan_tiers.push(removal);
            }
        }

        scan.entries
            .sort_by(|a, b| (a.last_used, &a.key).cmp(&(b.last_used, &b.key)));
        scan.orphan_tiers
            .sort_by(|a, b| (a.last_used, &a.name).cmp(&(b.last_used, &b.name)));
        scan
    }

    /// Every entry of the source tier, by source hash, plus the sweeps and tallies the tier
    /// itself contributes.
    fn scan_tiers(&self, claims: &Path, scan: &mut Scan) -> BTreeMap<String, TierFound> {
        let mut found = BTreeMap::new();
        let dir = self.root.join(SOURCE_DIR);
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return found;
        };
        let mut names: Vec<String> = entries
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        for name in names {
            let path = dir.join(&name);
            let bytes = path_bytes(&path);
            if !path.is_dir() {
                scan.usage.unknown.add(bytes);
                continue;
            }
            if is_transient(&name) {
                sweep(
                    scan,
                    claims,
                    Unreadable::Stale,
                    format!("{SOURCE_DIR}/{name}"),
                    bytes,
                );
                continue;
            }
            match crate::source::measure(&path) {
                Some(meta) => {
                    scan.usage.source.add(bytes);
                    let recorded = segment_bytes(&meta.segments)
                        + std::fs::metadata(path.join("source.json"))
                            .map(|manifest| manifest.len())
                            .unwrap_or(0);
                    account_size(&mut scan.usage, bytes, recorded);
                    found.insert(
                        name.clone(),
                        TierFound {
                            bytes,
                            created_at: meta.created_at,
                            is_open: is_held_open(claims, &source_claim(&name)),
                        },
                    );
                }
                None => {
                    sweep(
                        scan,
                        claims,
                        Unreadable::Unknown,
                        format!("{SOURCE_DIR}/{name}"),
                        bytes,
                    );
                }
            }
        }
        found
    }

    /// The source tier an entry renders from, when one exists and nothing holds it open.
    /// Entries with different settings share one tier, so the first candidate claims it.
    fn tier_removal(
        &self,
        meta: &Meta,
        tiers: &BTreeMap<String, TierFound>,
        claimed: &mut BTreeSet<String>,
        scan: &mut Scan,
    ) -> Option<Removal> {
        if meta.source_hash.is_empty() {
            return None;
        }
        let tier = tiers.get(&meta.source_hash)?;
        let removal = Removal {
            name: format!("{SOURCE_DIR}/{}", meta.source_hash),
            bytes: tier.bytes,
            last_used: meta.last_used,
        };
        let is_first = claimed.insert(meta.source_hash.clone());
        if tier.is_open {
            if is_first {
                scan.open.push(removal);
            }
            return None;
        }
        is_first.then_some(removal)
    }
}

/// What became of one planned removal.
enum Outcome {
    Removed(Removal),
    /// A holder appeared between the plan and the removal.
    Held,
    /// Another process removed it first.
    Absent,
}

/// The eviction policy, as a function of the scan and the cap: sweeps first, then the
/// source tier of each entry least recently used, then whole entries least recently used.
fn plan_eviction(cap: u64, scan: Scan) -> EvictionPlan {
    let Scan {
        usage,
        sweeps,
        orphan_tiers,
        entries,
        open,
    } = scan;
    let mut plan = EvictionPlan {
        cap,
        before: usage.total(),
        sweeps,
        open,
        after: usage.total(),
        ..EvictionPlan::default()
    };
    let mut after = plan.before;
    for sweep in &plan.sweeps {
        after = after.saturating_sub(sweep.bytes);
    }
    for tier in orphan_tiers {
        if after <= cap {
            break;
        }
        after = after.saturating_sub(tier.bytes);
        plan.tiers.push(tier);
    }
    // Every tier before any render: a tier is one decode pass, a render is the model.
    for candidate in &entries {
        if after <= cap {
            break;
        }
        if let Some(tier) = &candidate.tier {
            after = after.saturating_sub(tier.bytes);
            plan.tiers.push(tier.clone());
        }
    }
    for candidate in entries {
        if after <= cap {
            break;
        }
        after = after.saturating_sub(candidate.bytes);
        plan.entries.push(Removal {
            name: candidate.key,
            bytes: candidate.bytes,
            last_used: candidate.last_used,
        });
    }
    plan.after = after;
    plan
}

/// Which tally a directory nothing can read counts under.
#[derive(Clone, Copy)]
enum Unreadable {
    Stale,
    Unknown,
}

/// Note a directory nothing can play: counted under `kind`, and queued for the sweep unless
/// something holds it open.
fn sweep(scan: &mut Scan, claims: &Path, kind: Unreadable, name: String, bytes: u64) {
    match kind {
        Unreadable::Stale => scan.usage.stale.add(bytes),
        Unreadable::Unknown => scan.usage.unknown.add(bytes),
    }
    let removal = Removal {
        name,
        bytes,
        last_used: 0,
    };
    if is_held_open(claims, &removal_claim(&removal.name)) {
        scan.open.push(removal);
    } else {
        scan.sweeps.push(removal);
    }
}

/// The bytes a cache directory or file occupies.
fn path_bytes(path: &Path) -> u64 {
    if path.is_dir() {
        crate::setup::dir_size(path)
    } else {
        std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0)
    }
}

/// The claim that protects a directory named in a plan. A tier lives under `source/`, and
/// so does a staging directory in the tier: both take the hash's claim.
fn removal_claim(name: &str) -> String {
    match name.strip_prefix(&format!("{SOURCE_DIR}/")) {
        Some(rest) => source_claim(key_of(rest)),
        None => entry_claim(key_of(name)),
    }
}

/// Remove one directory by renaming it out of the namespace first, so a lookup either sees
/// it whole or does not see it. A kill during the delete leaves `<name>.evict-<pid>`, which
/// no lookup reads and the next sweep removes.
fn remove_tree(path: &Path) -> Result<u64> {
    let bytes = crate::setup::dir_size(path);
    let name = file_name(path);
    let doomed = path.with_file_name(format!("{name}{EVICTING}{}", std::process::id()));
    std::fs::rename(path, &doomed).with_context(|| format!("moving {} aside", path.display()))?;
    std::fs::remove_dir_all(&doomed).with_context(|| format!("removing {}", doomed.display()))?;
    Ok(bytes)
}

/// The bytes a manifest's segment table accounts for, without the manifest itself.
fn segment_bytes(segments: &[Segment]) -> u64 {
    segments.iter().map(|segment| segment.bytes).sum()
}

/// Note where a manifest and its files disagree. A manifest that names more than the
/// directory holds has been truncated or lived through a bad delete; one that names less
/// has a stray file, which is what a killed segment write leaves.
fn account_size(usage: &mut Usage, actual: u64, recorded: u64) {
    if recorded > actual {
        usage.overstated_entries += 1;
    } else {
        usage.unaccounted_bytes += actual - recorded;
    }
}

/// A byte count as `flacify cache` and `diagnose` print it.
pub fn human_bytes(bytes: u64) -> String {
    match bytes {
        b if b >= 1_000_000_000 => format!("{:.1} GB", b as f64 / 1e9),
        b if b >= 1_000_000 => format!("{:.1} MB", b as f64 / 1e6),
        b if b >= 1_000 => format!("{:.0} kB", b as f64 / 1e3),
        b => format!("{b} bytes"),
    }
}

/// Writes one entry: a segment at a time, with the manifest kept in step, so a kill
/// leaves everything it recorded readable and resumable. Only `finish` marks it complete.
#[derive(Debug)]
pub struct EntryWriter {
    /// Held for as long as the writer lives, refreshed as segments land, so eviction leaves
    /// a render in progress alone however long it runs.
    in_use: InUse,
    dir: PathBuf,
    meta: Meta,
    is_finished: bool,
}

impl EntryWriter {
    /// Publish one segment. The segment is flushed and renamed into place before the
    /// manifest that lists it is replaced, so no manifest names bytes that are not on
    /// disk. Indexes may arrive in any order.
    pub fn write_segment(&mut self, index: usize, planar: &[f32]) -> Result<()> {
        anyhow::ensure!(
            !self.is_finished,
            "{} is a complete entry, and a finished entry is not reopened",
            self.dir.display()
        );
        let expected =
            expected_segment_frames(self.meta.frames, self.meta.frames_per_segment, index)
                .with_context(|| format!("{} has no segment {index}", self.dir.display()))?;
        anyhow::ensure!(
            !self
                .meta
                .segments
                .iter()
                .any(|segment| segment.index == index),
            "{} already records segment {index}",
            self.dir.display()
        );
        anyhow::ensure!(
            planar.len() == expected * 2,
            "segment {index} of {} holds {expected} frames, this has {}",
            self.dir.display(),
            planar.len() / 2
        );
        self.in_use.refresh();

        let name = segment_name(index);
        let path = self.dir.join(&name);
        let temp = self.dir.join(format!("{name}.tmp"));
        crate::flac::write(&temp, flacify_core::SAMPLE_RATE, 2, planar)
            .with_context(|| format!("writing segment {index} of {}", self.dir.display()))?;
        // The bytes have to be on disk before the manifest names them.
        flush(&temp)?;
        std::fs::rename(&temp, &path).with_context(|| format!("replacing {}", path.display()))?;

        let bytes = std::fs::metadata(&path)
            .with_context(|| format!("measuring {}", path.display()))?
            .len();
        self.meta.segments.push(Segment {
            index,
            frames: expected,
            bytes,
        });
        self.meta.segments.sort_by_key(|segment| segment.index);
        self.meta.rendered_frames += expected;
        write_manifest(&self.dir, &self.meta)
    }

    /// Whether the entry already holds segment `index`. A resumed render skips those: a
    /// recorded segment is never rewritten.
    pub fn has_segment(&self, index: usize) -> bool {
        self.meta
            .segments
            .iter()
            .any(|segment| segment.index == index)
    }

    /// What the entry records about the render so far. A resumed render starts from this
    /// instead of re-rendering the segments on disk to recover their gain and report.
    pub fn progress(&self) -> EntryProgress {
        EntryProgress {
            gain_db: self.meta.gain_db,
            limit: crate::limit::LimitReport {
                peak_before: self.meta.peak_before,
                peak_after: self.meta.peak_after,
                limited_samples: self.meta.limited_samples,
            },
        }
    }

    /// Record what the render has established, so a kill does not lose it. The gain is
    /// written before the first segment because a resumed render cannot compute it, and
    /// the limiter's report after each segment because it covers only the audio written.
    pub fn record_progress(&mut self, progress: &EntryProgress) -> Result<()> {
        anyhow::ensure!(
            !self.is_finished,
            "{} is a complete entry, and a finished entry is not reopened",
            self.dir.display()
        );
        anyhow::ensure!(
            self.meta.gain_db.is_none() || progress.gain_db.is_some(),
            "{} already records its loudness gain",
            self.dir.display()
        );
        self.meta.gain_db = progress.gain_db;
        self.meta.peak_before = progress.limit.peak_before;
        self.meta.peak_after = progress.limit.peak_after;
        self.meta.limited_samples = progress.limit.limited_samples;
        write_manifest(&self.dir, &self.meta)
    }

    /// Mark the entry complete, which is what makes it readable. Refuses while the
    /// segment table does not partition the frames, leaving the entry partial.
    pub fn finish(&mut self, report: &Report) -> Result<PathBuf> {
        anyhow::ensure!(
            !self.is_finished,
            "{} is a complete entry, and a finished entry is not reopened",
            self.dir.display()
        );
        validate_complete(&self.dir.join(MANIFEST), &self.meta)?;
        let mut completed = self.meta.clone();
        completed.state = State::Complete;
        completed.peak_before = report.limit.peak_before;
        completed.peak_after = report.limit.peak_after;
        completed.limited_samples = report.limit.limited_samples;
        completed.device = report.device.clone();
        completed.gain_db = Some(report.gain_db);
        // In memory only once it is on disk, so a failed write leaves the writer partial.
        write_manifest(&self.dir, &completed)?;
        self.meta = completed;
        self.is_finished = true;
        Ok(self.dir.clone())
    }

    /// Stop writing and leave the entry partial for the next `begin` to resume. Segments
    /// are never deleted: a kill leaves the same state, and a caller that dies never
    /// reaches this call.
    pub fn abandon(self) {}
}

/// Decode one segment of an entry and check it is what the manifest promises.
fn decode_segment(
    dir: &Path,
    manifest: &Path,
    index: usize,
    frames: usize,
) -> Result<crate::decode::Decoded> {
    let path = dir.join(segment_name(index));
    anyhow::ensure!(
        path.is_file(),
        "{} names a segment that is missing: {}",
        manifest.display(),
        path.display()
    );
    let decoded = crate::decode::decode(&path)
        .with_context(|| format!("reading segment {index} of {}", dir.display()))?;
    anyhow::ensure!(
        decoded.sample_rate == flacify_core::SAMPLE_RATE && decoded.channels == 2,
        "{} is {} Hz with {} channels; cache segments are 48 kHz stereo",
        path.display(),
        decoded.sample_rate,
        decoded.channels
    );
    anyhow::ensure!(
        decoded.frames == frames,
        "{} holds {} frames, {} says {frames}",
        path.display(),
        decoded.frames,
        manifest.display()
    );
    Ok(decoded)
}

/// Frames per segment, derived from the settings so that every segment starts on a
/// multiple of the render step. The source tier uses the same length.
pub(crate) fn frames_per_segment(window: usize, overlap: usize) -> Result<usize> {
    let step = window
        .checked_sub(overlap)
        .filter(|step| *step > 0)
        .with_context(|| format!("window {window} is not longer than overlap {overlap}"))?;
    step.checked_mul(STEPS_PER_SEGMENT)
        .with_context(|| format!("{step} frames per step overflows a segment length"))
}

/// A manifest has to describe its own frame count with its own segments: the exact
/// partition when the entry is complete, and a consistent subset when it is partial.
/// Anything else is a corrupt entry, and a corrupt entry is an error.
fn validate_manifest(manifest: &Path, meta: &Meta) -> Result<()> {
    let expected_frames = frames_per_segment(meta.window, meta.overlap)
        .with_context(|| format!("{} has no usable segment length", manifest.display()))?;
    anyhow::ensure!(
        meta.frames_per_segment == expected_frames,
        "{} says {}-frame segments, its window {} and overlap {} give {expected_frames}",
        manifest.display(),
        meta.frames_per_segment,
        meta.window,
        meta.overlap
    );
    match meta.state {
        State::Complete => validate_complete(manifest, meta),
        State::Partial => validate_partial(manifest, meta),
    }
}

/// A complete entry's table is the whole track, in order, with no gap.
fn validate_complete(manifest: &Path, meta: &Meta) -> Result<()> {
    let mut covered = 0usize;
    for (position, segment) in meta.segments.iter().enumerate() {
        anyhow::ensure!(
            segment.index == position,
            "{} lists segment {} at position {position}",
            manifest.display(),
            segment.index
        );
        let expected = expected_segment_frames(meta.frames, meta.frames_per_segment, segment.index)
            .with_context(|| {
                format!(
                    "{} lists segment {} of a {}-frame track",
                    manifest.display(),
                    segment.index,
                    meta.frames
                )
            })?;
        anyhow::ensure!(
            segment.frames == expected,
            "{} says segment {} holds {} frames, the track needs {expected}",
            manifest.display(),
            segment.index,
            segment.frames
        );
        covered += expected;
    }
    anyhow::ensure!(
        covered == meta.frames,
        "{} promises {} frames but its segments hold {}",
        manifest.display(),
        meta.frames,
        covered
    );
    Ok(())
}

/// A partial entry may have gaps, because the renderer follows the playhead, but every
/// index it claims must be unique, hold the length that index implies, and add up to
/// `rendered_frames`.
fn validate_partial(manifest: &Path, meta: &Meta) -> Result<()> {
    let mut indexes = std::collections::BTreeSet::new();
    let mut rendered = 0usize;
    for segment in &meta.segments {
        anyhow::ensure!(
            indexes.insert(segment.index),
            "{} lists segment {} twice",
            manifest.display(),
            segment.index
        );
        let expected = expected_segment_frames(meta.frames, meta.frames_per_segment, segment.index)
            .with_context(|| {
                format!(
                    "{} lists segment {} of a {}-frame track",
                    manifest.display(),
                    segment.index,
                    meta.frames
                )
            })?;
        anyhow::ensure!(
            segment.frames == expected,
            "{} says segment {} holds {} frames, its index requires {expected}",
            manifest.display(),
            segment.index,
            segment.frames
        );
        rendered += segment.frames;
    }
    anyhow::ensure!(
        meta.rendered_frames == rendered,
        "{} says {} rendered frames, its segments hold {rendered}",
        manifest.display(),
        meta.rendered_frames
    );
    Ok(())
}

/// How many frames segment `index` of a `frames`-frame track holds, or `None` when the
/// index is past the last segment.
pub(crate) fn expected_segment_frames(
    frames: usize,
    frames_per_segment: usize,
    index: usize,
) -> Option<usize> {
    let start = index.checked_mul(frames_per_segment)?;
    if start >= frames {
        return None;
    }
    Some(frames_per_segment.min(frames - start))
}

/// Take over the segment files a killed writer left. One the manifest already lists is
/// kept untouched; one it does not is kept only when it decodes to the length its index
/// requires. Everything else under a segment name is removed.
fn adopt_segments(dir: &Path, meta: &Meta) -> Result<Vec<Segment>> {
    let mut segments = meta.segments.clone();
    let entries = std::fs::read_dir(dir).with_context(|| format!("listing {}", dir.display()))?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with("seg-") {
            continue;
        }
        if let Some(index) = segment_index(&name) {
            if segments.iter().any(|segment| segment.index == index) {
                continue;
            }
        }
        let path = entry.path();
        match adopt_segment(dir, &name, meta) {
            Some(segment) => segments.push(segment),
            None => std::fs::remove_file(&path)
                .with_context(|| format!("removing {}", path.display()))?,
        }
    }
    segments.sort_by_key(|segment| segment.index);
    Ok(segments)
}

/// One stray segment, as the manifest would describe it, or `None` when it is not a file
/// this build would have written for that index.
fn adopt_segment(dir: &Path, name: &str, meta: &Meta) -> Option<Segment> {
    let index = segment_index(name)?;
    if segment_name(index) != name {
        return None;
    }
    let expected = expected_segment_frames(meta.frames, meta.frames_per_segment, index)?;
    let path = dir.join(name);
    let decoded = crate::decode::decode(&path).ok()?;
    if decoded.sample_rate != flacify_core::SAMPLE_RATE
        || decoded.channels != 2
        || decoded.frames != expected
    {
        return None;
    }
    let bytes = std::fs::metadata(&path).ok()?.len();
    Some(Segment {
        index,
        frames: expected,
        bytes,
    })
}

/// The index in `seg-000005.flac`, or `None` when the name is not one of those.
fn segment_index(name: &str) -> Option<usize> {
    name.strip_prefix("seg-")?
        .strip_suffix(".flac")?
        .parse()
        .ok()
}

/// Cut one channel-major slice out of a whole-track buffer.
fn stereo_slice(planar: &[f32], total_frames: usize, start: usize, frames: usize) -> Vec<f32> {
    let mut slice = Vec::with_capacity(frames * 2);
    slice.extend_from_slice(&planar[start..start + frames]);
    let right = total_frames + start;
    slice.extend_from_slice(&planar[right..right + frames]);
    slice
}

fn segment_name(index: usize) -> String {
    format!("seg-{index:06}.flac")
}

/// Write the manifest through a temporary file in the same directory, so a reader never
/// sees half of one, and flush it before the rename so the bytes a manifest names are
/// already on disk.
fn write_manifest(dir: &Path, meta: &Meta) -> Result<()> {
    use std::io::Write as _;
    let path = dir.join(MANIFEST);
    let temp = dir.join(format!("{MANIFEST}.tmp"));
    {
        let mut file =
            std::fs::File::create(&temp).with_context(|| format!("creating {}", temp.display()))?;
        file.write_all((serde_json::to_string_pretty(meta)? + "\n").as_bytes())
            .with_context(|| format!("writing {}", temp.display()))?;
        file.sync_all()
            .with_context(|| format!("flushing {}", temp.display()))?;
    }
    std::fs::rename(&temp, &path).with_context(|| format!("replacing {}", path.display()))
}

/// Record that an entry was read, so eviction sees it as used. Best effort: a playable
/// entry matters more than its timestamp, and a stale time reads as an older entry rather
/// than a corrupt one.
fn touch(dir: &Path, meta: &mut Meta) {
    meta.last_used = now();
    if let Err(error) = write_manifest(dir, meta) {
        tracing::warn!(
            %error,
            entry = %dir.display(),
            "could not record when the cache entry was last used"
        );
    }
}

/// Flush one file's bytes to disk.
fn flush(path: &Path) -> Result<()> {
    std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .and_then(|file| file.sync_all())
        .with_context(|| format!("flushing {}", path.display()))
}

/// Remove an entry or a leftover staging directory, whichever is in the way.
fn clear(path: &Path) -> Result<()> {
    let removed = if path.is_dir() {
        std::fs::remove_dir_all(path)
    } else if path.exists() {
        std::fs::remove_file(path)
    } else {
        return Ok(());
    };
    removed.with_context(|| format!("clearing {}", path.display()))
}

/// Unix seconds, for `created_at` and `last_used` in the manifests.
pub(crate) fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

/// One read while hashing a source file, so the file is never held in memory whole.
const HASH_CHUNK: usize = 1 << 20;

/// blake3 of a file's bytes, first 32 hex characters: the identity part of a cache key.
///
/// Keyed by the container, so the same audio re-encoded is a miss and a retag is a miss;
/// keying on the decoded samples instead costs a full decode before every lookup.
pub fn source_hash(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("opening {} to hash it", path.display()))?;
    let mut hasher = blake3::Hasher::new();
    let mut chunk = vec![0u8; HASH_CHUNK];
    loop {
        let read = file
            .read(&mut chunk)
            .with_context(|| format!("hashing {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&chunk[..read]);
    }
    Ok(hasher.finalize().to_hex()[..32].to_string())
}

/// Everything that can change the rendered samples, hashed into a directory name.
pub fn key(
    source_hash: &str,
    window: usize,
    overlap: usize,
    model_id: &str,
    splice_low_band: bool,
) -> String {
    let mut hasher = blake3::Hasher::new();
    for part in [
        CACHE_VERSION,
        source_hash,
        model_id,
        &window.to_string(),
        &overlap.to_string(),
        if splice_low_band { "splice" } else { "whole" },
    ] {
        hasher.update(part.as_bytes());
        hasher.update(b"\0");
    }
    hasher.finalize().to_hex()[..32].to_string()
}

/// Identifies the weights without reading 3.3 GB: their sizes and modification times.
/// A same-size, same-mtime revision would slip through, which is why `CACHE_VERSION`
/// exists and why that is written down rather than assumed.
pub fn model_id(models_dir: &Path) -> String {
    let mut hasher = blake3::Hasher::new();
    for name in ["student_ldm.pth", "sr_vocoder.pth", "vae.pth"] {
        let path = models_dir.join(name);
        hasher.update(name.as_bytes());
        match std::fs::metadata(&path) {
            Ok(meta) => {
                hasher.update(&meta.len().to_le_bytes());
                if let Ok(modified) = meta.modified() {
                    if let Ok(since) = modified.duration_since(std::time::UNIX_EPOCH) {
                        hasher.update(&since.as_secs().to_le_bytes());
                    }
                }
            }
            Err(_) => {
                hasher.update(b"missing");
            }
        }
    }
    hasher.finalize().to_hex()[..16].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limit::LimitReport;

    /// One 24-bit step: `flac` rounds to this, and a sample already on the grid survives
    /// the round trip unchanged.
    const QUANTUM: f32 = 1.0 / 8_388_608.0;

    fn temp_cache(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "flacify-cache-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        dir
    }

    /// A render whose samples are all exact 24-bit values, so a round trip either returns
    /// them unchanged or shows the bug.
    fn render_of(frames: usize) -> Render {
        let mut planar = Vec::with_capacity(frames * 2);
        for channel in 0..2 {
            for frame in 0..frames {
                let step = (frame as i32 * (7_919 + channel * 3_217)) % 4_096 - 2_048;
                planar.push(step as f32 * QUANTUM);
            }
        }
        Render {
            planar,
            limit: LimitReport {
                peak_before: 0.5,
                peak_after: 0.5,
                limited_samples: 0,
            },
            device: "cpu".to_string(),
            gain_db: 0.0,
            lufs_input: -14.0,
            lufs_output: -15.0,
        }
    }

    /// The shipped settings: a 245 760-frame window with a 48 000-frame overlap.
    fn fields() -> Fields {
        Fields {
            source_hash: "source".into(),
            window: 245_760,
            overlap: 48_000,
            model_id: "model".into(),
            splice: false,
        }
    }

    /// The same entry with 16 800-frame segments, so a test can span several of them
    /// without encoding a minute of audio.
    fn fields_with_short_segments() -> Fields {
        Fields {
            window: 4_800,
            overlap: 2_400,
            ..fields()
        }
    }

    fn manifest_path(cache: &Cache, key: &str) -> PathBuf {
        cache.entry_dir(key).join(MANIFEST)
    }

    fn stored_meta(cache: &Cache, key: &str) -> Meta {
        let path = manifest_path(cache, key);
        serde_json::from_str(&std::fs::read_to_string(&path).expect("reading the manifest"))
            .expect("parsing the manifest")
    }

    fn store_meta(cache: &Cache, key: &str, meta: &Meta) {
        std::fs::write(
            manifest_path(cache, key),
            serde_json::to_string_pretty(meta).expect("serialising the manifest"),
        )
        .expect("writing the manifest");
    }

    /// The segment length the writer tests use: the short settings below.
    const SEGMENT_FRAMES: usize = 16_800;

    /// The planar samples of one segment of a whole-track render.
    fn segment_planar(render: &Render, frames: usize, index: usize) -> Vec<f32> {
        let start = index * SEGMENT_FRAMES;
        let count = SEGMENT_FRAMES.min(frames - start);
        stereo_slice(&render.planar, frames, start, count)
    }

    fn spec(frames: usize, fields: &Fields) -> EntrySpec {
        EntrySpec {
            frames,
            cutoff_hz: 12_000,
            fields: fields.clone(),
        }
    }

    fn report() -> Report {
        Report {
            limit: LimitReport {
                peak_before: 0.75,
                peak_after: 0.5,
                limited_samples: 3,
            },
            device: "cpu".into(),
            gain_db: -1.5,
        }
    }

    /// Four segments of the short-segment settings.
    fn four_segments() -> (Fields, usize, Render) {
        let frames = SEGMENT_FRAMES * 4;
        (fields_with_short_segments(), frames, render_of(frames))
    }

    fn modified(path: &Path) -> std::time::SystemTime {
        std::fs::metadata(path)
            .expect("measuring a file")
            .modified()
            .expect("a modification time")
    }

    #[test]
    fn given_a_render_that_does_not_fill_its_last_segment_when_stored_and_loaded_then_the_samples_and_frame_count_match(
    ) {
        let root = temp_cache("roundtrip");
        let cache = Cache::new(root.clone());
        let fields = fields_with_short_segments();
        let frames = 16_800 * 2 + 5_000;
        let render = render_of(frames);
        cache
            .store("key", &render, 15_328, &fields)
            .expect("storing");

        let cached = cache.load("key").expect("loading").expect("an entry");
        assert_eq!(cached.meta.frames, frames);
        assert_eq!(cached.meta.cutoff_hz, 15_328);
        assert_eq!(cached.meta.segments.len(), 3);
        assert_eq!(cached.planar.len(), render.planar.len());
        for (index, (&expected, &actual)) in render.planar.iter().zip(&cached.planar).enumerate() {
            assert_eq!(actual, expected, "sample {index}: {actual} != {expected}");
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_stored_entry_when_read_a_segment_at_a_time_then_each_segment_is_that_slice_of_the_whole(
    ) {
        let root = temp_cache("segment-reads");
        let cache = Cache::new(root.clone());
        let fields = fields_with_short_segments();
        let frames = SEGMENT_FRAMES * 3 + 500;
        let render = render_of(frames);
        cache
            .store("key", &render, 12_000, &fields)
            .expect("storing");

        let segments = cache
            .segments("key")
            .expect("opening")
            .expect("a complete entry");
        assert_eq!(segments.meta().frames, frames);
        assert_eq!(segments.meta().gain_db, Some(0.0));
        for index in 0..4 {
            assert_eq!(
                segments.read_segment(index).expect("reading a segment"),
                segment_planar(&render, frames, index),
                "segment {index} must be the slice the whole-track read gives"
            );
        }
        let error = segments
            .read_segment(4)
            .expect_err("there is no fifth segment");
        assert!(format!("{error:#}").contains("no segment 4"), "{error:#}");
        assert_eq!(
            segments.read_all().expect("the whole entry"),
            render.planar,
            "the segments assemble into the whole track"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_complete_entry_with_a_missing_segment_when_opened_for_segments_then_it_is_an_error()
    {
        let root = temp_cache("segment-missing");
        let cache = Cache::new(root.clone());
        let (fields, _frames, render) = four_segments();
        cache
            .store("key", &render, 12_000, &fields)
            .expect("storing");
        std::fs::remove_file(cache.entry_dir("key").join("seg-000002.flac"))
            .expect("removing a segment");

        let error = cache
            .segments("key")
            .expect_err("a missing segment is an error, not a miss");
        let text = format!("{error:#}");
        assert!(text.contains("seg-000002.flac"), "{text}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_stored_entry_when_searched_by_its_fields_then_the_same_entry_opens() {
        let root = temp_cache("find-segments");
        let cache = Cache::new(root.clone());
        let (fields, frames, render) = four_segments();
        cache
            .store("key", &render, 12_000, &fields)
            .expect("storing");

        let found = cache
            .find_segments(&fields)
            .expect("searching")
            .expect("the complete entry");
        assert_eq!(found.meta().frames, frames);
        assert!(cache
            .find_segments(&Fields {
                source_hash: "another source".into(),
                ..fields
            })
            .expect("searching")
            .is_none());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_stored_entry_when_read_back_then_each_manifest_segment_probes_and_decodes_and_the_frames_add_up(
    ) {
        let root = temp_cache("segments");
        let cache = Cache::new(root.clone());
        let fields = fields_with_short_segments();
        cache
            .store("key", &render_of(16_800 * 2 + 500), 12_000, &fields)
            .expect("storing");

        let dir = cache.entry_dir("key");
        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .expect("listing the entry")
            .map(|entry| {
                entry
                    .expect("an entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                "manifest.json",
                "seg-000000.flac",
                "seg-000001.flac",
                "seg-000002.flac"
            ]
        );

        let meta = stored_meta(&cache, "key");
        assert_eq!(meta.frames_per_segment, 16_800);
        assert_eq!(meta.segments.len(), 3);
        let mut decoded_frames = 0;
        for segment in &meta.segments {
            let path = dir.join(segment_name(segment.index));
            let decoded = crate::decode::decode(&path).expect("decoding a segment");
            assert_eq!(decoded.sample_rate, 48_000);
            assert_eq!(decoded.channels, 2);
            assert_eq!(decoded.frames, segment.frames);
            assert_eq!(
                std::fs::metadata(&path).expect("measuring a segment").len(),
                segment.bytes
            );
            decoded_frames += decoded.frames;
        }
        assert_eq!(decoded_frames, meta.frames);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_track_shorter_than_one_segment_when_stored_then_the_entry_holds_exactly_one_segment()
    {
        let root = temp_cache("one-segment");
        let cache = Cache::new(root.clone());
        cache
            .store("key", &render_of(2_400), 12_000, &fields())
            .expect("storing");

        let meta = stored_meta(&cache, "key");
        assert_eq!(meta.segments.len(), 1);
        assert_eq!(meta.segments[0].frames, 2_400);
        let dir = cache.entry_dir("key");
        assert!(dir.join("seg-000000.flac").is_file());
        assert!(!dir.join("seg-000001.flac").exists());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_an_entry_whose_last_used_is_old_when_loaded_then_the_manifest_records_the_new_time() {
        let root = temp_cache("last-used");
        let cache = Cache::new(root.clone());
        cache
            .store("key", &render_of(240), 12_000, &fields())
            .expect("storing");
        let mut meta = stored_meta(&cache, "key");
        assert!(meta.created_at > 0, "created_at is set on store");
        meta.last_used = 1;
        store_meta(&cache, "key", &meta);

        let cached = cache.load("key").expect("loading").expect("an entry");
        assert!(cached.meta.last_used > 1);
        assert_eq!(stored_meta(&cache, "key").last_used, cached.meta.last_used);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_no_entry_when_loaded_then_it_is_none_rather_than_an_error() {
        let root = temp_cache("missing");
        let cache = Cache::new(root.clone());
        assert!(cache.load("nothing-here").expect("no error").is_none());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_version_mismatch_when_loaded_then_it_is_a_miss() {
        let root = temp_cache("version");
        let cache = Cache::new(root.clone());
        cache
            .store("key", &render_of(240), 12_000, &fields())
            .expect("storing");
        let mut meta = stored_meta(&cache, "key");
        meta.version = "something-older".into();
        store_meta(&cache, "key", &meta);

        assert!(cache.load("key").expect("no error").is_none());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_manifest_that_names_a_missing_segment_when_loaded_then_it_is_an_error_naming_the_entry(
    ) {
        let root = temp_cache("missing-segment");
        let cache = Cache::new(root.clone());
        let fields = fields_with_short_segments();
        cache
            .store("key", &render_of(17_800), 12_000, &fields)
            .expect("storing");
        std::fs::remove_file(cache.entry_dir("key").join("seg-000001.flac"))
            .expect("removing a segment");

        let error = cache
            .load("key")
            .expect_err("a missing segment is an error");
        let text = format!("{error:#}");
        assert!(
            text.contains(&manifest_path(&cache, "key").display().to_string()),
            "{text}"
        );
        assert!(text.contains("seg-000001.flac"), "{text}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_manifest_with_a_gap_in_its_segment_indexes_when_loaded_then_it_is_an_error() {
        let root = temp_cache("index-gap");
        let cache = Cache::new(root.clone());
        let fields = fields_with_short_segments();
        cache
            .store("key", &render_of(17_800), 12_000, &fields)
            .expect("storing");
        let mut meta = stored_meta(&cache, "key");
        meta.segments[1].index = 2;
        store_meta(&cache, "key", &meta);

        let error = cache.load("key").expect_err("a gap is an error");
        assert!(format!("{error:#}").contains("position 1"), "{error:#}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_manifest_whose_segments_are_out_of_order_when_loaded_then_it_is_an_error() {
        let root = temp_cache("out-of-order");
        let cache = Cache::new(root.clone());
        let fields = fields_with_short_segments();
        cache
            .store("key", &render_of(17_800), 12_000, &fields)
            .expect("storing");
        let mut meta = stored_meta(&cache, "key");
        meta.segments.swap(0, 1);
        store_meta(&cache, "key", &meta);

        let error = cache.load("key").expect_err("out of order is an error");
        assert!(format!("{error:#}").contains("position 0"), "{error:#}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_manifest_whose_segment_length_disagrees_with_its_window_when_loaded_then_it_is_an_error(
    ) {
        let root = temp_cache("segment-length");
        let cache = Cache::new(root.clone());
        cache
            .store("key", &render_of(240), 12_000, &fields())
            .expect("storing");
        let mut meta = stored_meta(&cache, "key");
        meta.frames_per_segment += 1;
        store_meta(&cache, "key", &meta);

        let error = cache
            .load("key")
            .expect_err("a wrong segment length is an error");
        assert!(format!("{error:#}").contains("window 245760"), "{error:#}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_manifest_written_before_the_state_field_when_loaded_then_it_parses_and_misses() {
        let root = temp_cache("no-state");
        let cache = Cache::new(root.clone());
        cache
            .store("key", &render_of(240), 12_000, &fields())
            .expect("storing");
        let path = manifest_path(&cache, "key");
        let mut json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("reading"))
                .expect("parsing");
        let object = json.as_object_mut().expect("an object");
        object.insert("version".into(), "flacify-cache-8".into());
        object.remove("state");
        object.remove("rendered_frames");
        std::fs::write(&path, serde_json::to_string(&json).expect("serialising")).expect("writing");

        assert!(
            cache
                .load("key")
                .expect("an older manifest parses")
                .is_none(),
            "a manifest from another version is a miss, not a parse error"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_manifest_with_no_usable_segment_length_when_loaded_then_it_is_an_error() {
        let root = temp_cache("no-window");
        let cache = Cache::new(root.clone());
        cache
            .store("key", &render_of(240), 12_000, &fields())
            .expect("storing");
        let mut meta = stored_meta(&cache, "key");
        meta.window = 0;
        meta.overlap = 0;
        meta.frames_per_segment = 0;
        store_meta(&cache, "key", &meta);

        let error = cache.load("key").expect_err("no step is an error");
        assert!(
            format!("{error:#}").contains("no usable segment length"),
            "{error:#}"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_manifest_that_stops_before_the_last_frames_when_loaded_then_it_is_an_error() {
        let root = temp_cache("short-manifest");
        let cache = Cache::new(root.clone());
        let fields = fields_with_short_segments();
        cache
            .store("key", &render_of(17_800), 12_000, &fields)
            .expect("storing");
        let mut meta = stored_meta(&cache, "key");
        meta.segments.pop();
        store_meta(&cache, "key", &meta);

        let error = cache
            .load("key")
            .expect_err("a manifest without its last segment is an error");
        assert!(
            format!("{error:#}").contains("promises 17800 frames"),
            "{error:#}"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_manifest_that_understates_a_segment_when_loaded_then_it_is_an_error() {
        let root = temp_cache("segment-frames");
        let cache = Cache::new(root.clone());
        cache
            .store("key", &render_of(240), 12_000, &fields())
            .expect("storing");
        let mut meta = stored_meta(&cache, "key");
        meta.segments[0].frames = 200;
        store_meta(&cache, "key", &meta);

        let error = cache.load("key").expect_err("a short segment is an error");
        assert!(format!("{error:#}").contains("needs 240"), "{error:#}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_segment_that_decodes_to_another_length_when_loaded_then_it_is_an_error() {
        let root = temp_cache("segment-truncated");
        let cache = Cache::new(root.clone());
        cache
            .store("key", &render_of(240), 12_000, &fields())
            .expect("storing");
        let path = cache.entry_dir("key").join("seg-000000.flac");
        crate::flac::write(&path, 48_000, 2, &render_of(200).planar).expect("rewriting a segment");

        let error = cache.load("key").expect_err("a short decode is an error");
        assert!(
            format!("{error:#}").contains("holds 200 frames"),
            "{error:#}"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_segment_at_the_wrong_rate_when_loaded_then_it_is_an_error() {
        let root = temp_cache("segment-rate");
        let cache = Cache::new(root.clone());
        cache
            .store("key", &render_of(240), 12_000, &fields())
            .expect("storing");
        let path = cache.entry_dir("key").join("seg-000000.flac");
        crate::flac::write(&path, 44_100, 2, &render_of(240).planar).expect("rewriting a segment");

        let error = cache
            .load("key")
            .expect_err("a 44.1 kHz segment is an error");
        assert!(format!("{error:#}").contains("44100 Hz"), "{error:#}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_render_with_no_frames_when_stored_then_it_is_an_error() {
        let root = temp_cache("empty");
        let cache = Cache::new(root.clone());
        let error = cache
            .store("key", &render_of(0), 12_000, &fields())
            .expect_err("an entry needs a frame");
        assert!(
            format!("{error:#}").contains("at least one frame"),
            "{error:#}"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_buffer_that_is_not_whole_stereo_frames_when_stored_then_it_is_an_error() {
        let root = temp_cache("odd");
        let cache = Cache::new(root.clone());
        let mut render = render_of(2);
        render.planar.pop();
        let error = cache
            .store("key", &render, 12_000, &fields())
            .expect_err("3 samples are not whole frames");
        assert!(
            format!("{error:#}").contains("not whole stereo frames"),
            "{error:#}"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_fresh_entry_when_two_of_four_segments_are_written_then_the_manifest_is_partial_and_load_misses(
    ) {
        let root = temp_cache("partial-two");
        let cache = Cache::new(root.clone());
        let (fields, frames, render) = four_segments();
        let mut writer = cache
            .begin("key", &spec(frames, &fields))
            .expect("beginning");
        for index in [0, 1] {
            writer
                .write_segment(index, &segment_planar(&render, frames, index))
                .expect("writing a segment");
        }

        let meta = stored_meta(&cache, "key");
        assert_eq!(meta.state, State::Partial);
        assert_eq!(meta.rendered_frames, SEGMENT_FRAMES * 2);
        assert_eq!(
            meta.segments
                .iter()
                .map(|segment| segment.index)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        assert!(
            cache.load("key").expect("no error").is_none(),
            "a partial entry is a miss"
        );
        assert!(
            cache.segments("key").expect("no error").is_none(),
            "a partial entry cannot back segmented playback"
        );
        assert!(
            cache.find_segments(&fields).expect("searching").is_none(),
            "a search never returns a partial entry"
        );
        let dir = cache.entry_dir("key");
        assert!(dir.join("seg-000000.flac").is_file());
        assert!(dir.join("seg-000001.flac").is_file());
        assert!(!dir.join("seg-000002.flac").exists());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_segments_zero_and_two_when_finished_then_it_refuses_until_the_table_partitions_the_frames(
    ) {
        let root = temp_cache("out-of-order-write");
        let cache = Cache::new(root.clone());
        let (fields, frames, render) = four_segments();
        let mut writer = cache
            .begin("key", &spec(frames, &fields))
            .expect("beginning");
        for index in [0, 2] {
            writer
                .write_segment(index, &segment_planar(&render, frames, index))
                .expect("writing a segment");
        }
        assert_eq!(
            stored_meta(&cache, "key")
                .segments
                .iter()
                .map(|segment| segment.index)
                .collect::<Vec<_>>(),
            [0, 2],
            "the gaps are allowed while the entry is partial"
        );

        let error = writer
            .finish(&report())
            .expect_err("a table with a gap is not a whole track");
        assert!(format!("{error:#}").contains("at position 1"), "{error:#}");
        assert_eq!(stored_meta(&cache, "key").state, State::Partial);
        writer.abandon();

        let mut writer = cache
            .begin("key", &spec(frames, &fields))
            .expect("resuming");
        for index in [1, 3] {
            writer
                .write_segment(index, &segment_planar(&render, frames, index))
                .expect("writing a segment");
        }
        writer.finish(&report()).expect("a full table completes");

        let cached = cache.load("key").expect("loading").expect("an entry");
        assert_eq!(cached.meta.state, State::Complete);
        assert_eq!(cached.meta.rendered_frames, frames);
        assert_eq!(cached.planar.len(), render.planar.len());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_partial_entry_when_begun_again_then_the_recorded_segments_keep_their_times_and_still_read(
    ) {
        let root = temp_cache("resume");
        let cache = Cache::new(root.clone());
        let (fields, frames, render) = four_segments();
        let mut writer = cache
            .begin("key", &spec(frames, &fields))
            .expect("beginning");
        for index in [0, 1] {
            writer
                .write_segment(index, &segment_planar(&render, frames, index))
                .expect("writing a segment");
        }
        writer.abandon();

        let dir = cache.entry_dir("key");
        let recorded: Vec<(PathBuf, std::time::SystemTime)> = [0, 1]
            .iter()
            .map(|&index| {
                let path = dir.join(segment_name(index));
                (path.clone(), modified(&path))
            })
            .collect();

        let mut meta = stored_meta(&cache, "key");
        meta.last_used = 1;
        store_meta(&cache, "key", &meta);

        let mut resumed = cache
            .begin("key", &spec(frames, &fields))
            .expect("resuming");
        assert!(
            stored_meta(&cache, "key").last_used > 1,
            "resuming a partial entry is using it"
        );
        let error = resumed
            .write_segment(0, &segment_planar(&render, frames, 0))
            .expect_err("a recorded segment is not rewritten");
        assert!(
            format!("{error:#}").contains("already records segment 0"),
            "{error:#}"
        );
        for (path, was_modified) in &recorded {
            assert_eq!(
                modified(path),
                *was_modified,
                "{} was rewritten",
                path.display()
            );
            let decoded = crate::decode::decode(path).expect("decoding a kept segment");
            assert_eq!(decoded.frames, SEGMENT_FRAMES);
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_stray_segment_that_fits_its_index_when_begun_then_it_is_adopted_and_not_rewritten() {
        let root = temp_cache("adopt");
        let cache = Cache::new(root.clone());
        let (fields, frames, render) = four_segments();
        let mut writer = cache
            .begin("key", &spec(frames, &fields))
            .expect("beginning");
        writer
            .write_segment(0, &segment_planar(&render, frames, 0))
            .expect("writing a segment");
        writer.abandon();

        // What a kill between the segment's rename and the manifest update leaves.
        let dir = cache.entry_dir("key");
        let stray = dir.join(segment_name(1));
        crate::flac::write(
            &stray,
            flacify_core::SAMPLE_RATE,
            2,
            &segment_planar(&render, frames, 1),
        )
        .expect("writing a stray segment");
        let was_modified = modified(&stray);

        let mut resumed = cache
            .begin("key", &spec(frames, &fields))
            .expect("resuming");
        let meta = stored_meta(&cache, "key");
        assert_eq!(
            meta.segments
                .iter()
                .map(|segment| segment.index)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(meta.rendered_frames, SEGMENT_FRAMES * 2);
        assert_eq!(
            modified(&stray),
            was_modified,
            "an adopted segment is not rewritten"
        );

        for index in [2, 3] {
            resumed
                .write_segment(index, &segment_planar(&render, frames, index))
                .expect("writing a segment");
        }
        resumed.finish(&report()).expect("finishing");
        let cached = cache.load("key").expect("loading").expect("an entry");
        for (index, (&expected, &actual)) in render.planar.iter().zip(&cached.planar).enumerate() {
            assert_eq!(actual, expected, "sample {index}: {actual} != {expected}");
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_stray_segment_files_when_begun_then_the_ones_that_do_not_fit_their_index_are_removed()
    {
        let root = temp_cache("adopt-wrong");
        let cache = Cache::new(root.clone());
        let (fields, frames, render) = four_segments();
        let mut writer = cache
            .begin("key", &spec(frames, &fields))
            .expect("beginning");
        writer
            .write_segment(0, &segment_planar(&render, frames, 0))
            .expect("writing a segment");
        writer.abandon();

        let dir = cache.entry_dir("key");
        let short = dir.join(segment_name(1));
        crate::flac::write(&short, flacify_core::SAMPLE_RATE, 2, &render_of(100).planar)
            .expect("writing a short segment");
        let unnumbered = dir.join("seg-one.flac");
        std::fs::write(&unnumbered, b"not a FLAC stream").expect("writing a stray file");
        let interrupted = dir.join("seg-000002.flac.tmp");
        std::fs::write(&interrupted, b"half a write").expect("writing an interrupted segment");

        let _resumed = cache
            .begin("key", &spec(frames, &fields))
            .expect("resuming");
        assert!(!short.exists(), "a segment that does not fit is removed");
        assert!(
            !unnumbered.exists(),
            "a segment name with no number is removed"
        );
        assert!(!interrupted.exists(), "an interrupted write is removed");
        let meta = stored_meta(&cache, "key");
        assert_eq!(
            meta.segments
                .iter()
                .map(|segment| segment.index)
                .collect::<Vec<_>>(),
            [0]
        );
        assert_eq!(meta.rendered_frames, SEGMENT_FRAMES);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_partial_entry_when_its_progress_is_recorded_then_beginning_it_again_reads_that_back()
    {
        let root = temp_cache("progress");
        let cache = Cache::new(root.clone());
        let (fields, frames, render) = four_segments();
        let progress = EntryProgress {
            gain_db: Some(1.25),
            limit: LimitReport {
                peak_before: 0.97,
                peak_after: 0.95,
                limited_samples: 11,
            },
        };
        let mut writer = cache
            .begin("key", &spec(frames, &fields))
            .expect("beginning");
        writer
            .record_progress(&progress)
            .expect("recording the progress");
        writer
            .write_segment(0, &segment_planar(&render, frames, 0))
            .expect("writing a segment");
        writer.abandon();

        let resumed = cache
            .begin("key", &spec(frames, &fields))
            .expect("resuming");
        assert_eq!(
            resumed.progress(),
            progress,
            "a resume needs the gain and the report it did not render"
        );
        assert!(resumed.has_segment(0));
        assert!(!resumed.has_segment(1));
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_recorded_gain_when_progress_without_one_is_recorded_then_the_writer_refuses() {
        let root = temp_cache("progress-gain");
        let cache = Cache::new(root.clone());
        let (fields, frames, _) = four_segments();
        let mut writer = cache
            .begin("key", &spec(frames, &fields))
            .expect("beginning");
        writer
            .record_progress(&EntryProgress {
                gain_db: Some(0.5),
                ..Default::default()
            })
            .expect("recording a gain");

        let error = writer
            .record_progress(&EntryProgress::default())
            .expect_err("a recorded gain is not cleared");
        assert!(
            format!("{error:#}").contains("already records its loudness gain"),
            "{error:#}"
        );
        assert_eq!(writer.progress().gain_db, Some(0.5));
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_written_entry_when_finished_then_load_returns_every_sample_bit_exactly() {
        let root = temp_cache("writer-roundtrip");
        let cache = Cache::new(root.clone());
        let (fields, frames, render) = four_segments();
        let mut writer = cache
            .begin("key", &spec(frames, &fields))
            .expect("beginning");
        for index in 0..4 {
            writer
                .write_segment(index, &segment_planar(&render, frames, index))
                .expect("writing a segment");
        }
        let dir = writer.finish(&report()).expect("finishing");
        assert_eq!(dir, cache.entry_dir("key"));

        let cached = cache.load("key").expect("loading").expect("an entry");
        assert_eq!(cached.meta.state, State::Complete);
        assert_eq!(cached.meta.rendered_frames, frames);
        assert_eq!(cached.meta.gain_db, Some(-1.5));
        assert_eq!(cached.meta.device, "cpu");
        assert_eq!(cached.meta.peak_before, 0.75);
        assert_eq!(cached.meta.peak_after, 0.5);
        assert_eq!(cached.meta.limited_samples, 3);
        for (index, (&expected, &actual)) in render.planar.iter().zip(&cached.planar).enumerate() {
            assert_eq!(actual, expected, "sample {index}: {actual} != {expected}");
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_one_segment_entry_when_a_segment_is_written_after_finishing_then_the_writer_refuses()
    {
        let root = temp_cache("finished");
        let cache = Cache::new(root.clone());
        let fields = fields_with_short_segments();
        let frames = SEGMENT_FRAMES;
        let render = render_of(frames);
        let mut writer = cache
            .begin("key", &spec(frames, &fields))
            .expect("beginning");
        writer
            .write_segment(0, &render.planar)
            .expect("writing the only segment");
        writer.finish(&report()).expect("finishing");

        let path = cache.entry_dir("key").join(segment_name(0));
        let was_modified = modified(&path);
        let error = writer
            .write_segment(0, &render.planar)
            .expect_err("a finished entry is not reopened");
        assert!(format!("{error:#}").contains("not reopened"), "{error:#}");
        assert_eq!(
            modified(&path),
            was_modified,
            "the finished segment is untouched"
        );
        let error = writer
            .finish(&report())
            .expect_err("finishing twice is a mistake");
        assert!(format!("{error:#}").contains("not reopened"), "{error:#}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_render_when_stored_then_the_published_entry_is_complete_and_the_staging_directory_is_gone(
    ) {
        let root = temp_cache("store-complete");
        let cache = Cache::new(root.clone());
        let fields = fields_with_short_segments();
        let frames = SEGMENT_FRAMES * 2 + 500;
        let render = render_of(frames);
        cache
            .store("key", &render, 15_328, &fields)
            .expect("storing");

        assert!(
            !cache.root().join("key.tmp").exists(),
            "the staging directory is renamed, never left beside the entry"
        );
        let meta = stored_meta(&cache, "key");
        assert_eq!(meta.state, State::Complete);
        assert_eq!(meta.rendered_frames, frames);
        let cached = cache.load("key").expect("loading").expect("an entry");
        assert_eq!(cached.meta.cutoff_hz, 15_328);
        assert_eq!(cached.planar.len(), render.planar.len());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_stored_entry_when_stored_again_then_it_is_replaced_and_still_loads() {
        let root = temp_cache("store-again");
        let cache = Cache::new(root.clone());
        let fields = fields_with_short_segments();
        cache
            .store("key", &render_of(SEGMENT_FRAMES), 12_000, &fields)
            .expect("storing");
        cache
            .store("key", &render_of(SEGMENT_FRAMES * 2), 12_000, &fields)
            .expect("replacing");

        let cached = cache.load("key").expect("loading").expect("an entry");
        assert_eq!(cached.meta.frames, SEGMENT_FRAMES * 2);
        assert_eq!(cached.meta.segments.len(), 2);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_complete_entry_when_begun_then_it_is_an_error() {
        let root = temp_cache("begin-complete");
        let cache = Cache::new(root.clone());
        let fields = fields_with_short_segments();
        cache
            .store("key", &render_of(SEGMENT_FRAMES), 12_000, &fields)
            .expect("storing");

        let error = cache
            .begin("key", &spec(SEGMENT_FRAMES, &fields))
            .expect_err("a finished entry is not reopened");
        assert!(format!("{error:#}").contains("not reopened"), "{error:#}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_partial_manifest_that_lists_a_segment_twice_when_begun_then_it_is_an_error() {
        let root = temp_cache("partial-duplicate");
        let cache = Cache::new(root.clone());
        let (fields, frames, render) = four_segments();
        let mut writer = cache
            .begin("key", &spec(frames, &fields))
            .expect("beginning");
        writer
            .write_segment(0, &segment_planar(&render, frames, 0))
            .expect("writing a segment");
        writer.abandon();

        let mut meta = stored_meta(&cache, "key");
        let duplicate = meta.segments[0].clone();
        meta.segments.push(duplicate);
        store_meta(&cache, "key", &meta);

        let error = cache
            .begin("key", &spec(frames, &fields))
            .expect_err("a duplicate index is corrupt");
        assert!(
            format!("{error:#}").contains("lists segment 0 twice"),
            "{error:#}"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_partial_manifest_whose_segment_length_is_wrong_when_begun_then_it_is_an_error() {
        let root = temp_cache("partial-length");
        let cache = Cache::new(root.clone());
        let (fields, frames, render) = four_segments();
        let mut writer = cache
            .begin("key", &spec(frames, &fields))
            .expect("beginning");
        writer
            .write_segment(0, &segment_planar(&render, frames, 0))
            .expect("writing a segment");
        writer.abandon();

        let mut meta = stored_meta(&cache, "key");
        meta.segments[0].frames = 100;
        store_meta(&cache, "key", &meta);

        let error = cache
            .begin("key", &spec(frames, &fields))
            .expect_err("a segment of the wrong length is corrupt");
        assert!(
            format!("{error:#}").contains("its index requires 16800"),
            "{error:#}"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_partial_manifest_whose_rendered_frames_disagree_when_begun_then_it_is_an_error() {
        let root = temp_cache("partial-rendered");
        let cache = Cache::new(root.clone());
        let (fields, frames, render) = four_segments();
        let mut writer = cache
            .begin("key", &spec(frames, &fields))
            .expect("beginning");
        writer
            .write_segment(0, &segment_planar(&render, frames, 0))
            .expect("writing a segment");
        writer.abandon();

        let mut meta = stored_meta(&cache, "key");
        meta.rendered_frames += 1;
        store_meta(&cache, "key", &meta);

        let error = cache
            .begin("key", &spec(frames, &fields))
            .expect_err("a rendered count that disagrees is corrupt");
        assert!(
            format!("{error:#}").contains("rendered frames, its segments hold"),
            "{error:#}"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_partial_entry_and_a_complete_one_for_the_same_source_when_found_then_the_partial_is_skipped(
    ) {
        let root = temp_cache("find-partial");
        let cache = Cache::new(root.clone());
        let fields = fields_with_short_segments();
        let render = render_of(SEGMENT_FRAMES);
        // "aaa" sorts first, so a search that stopped at the partial entry would miss
        // the complete one.
        let mut writer = cache
            .begin("aaa-partial", &spec(SEGMENT_FRAMES, &fields))
            .expect("beginning");
        writer
            .write_segment(0, &render.planar)
            .expect("writing a segment");
        writer.abandon();
        cache
            .store("zzz-complete", &render, 12_000, &fields)
            .expect("storing");

        let found = cache
            .find_segments(&fields)
            .expect("searching")
            .expect("the complete entry");
        assert_eq!(found.meta().state, State::Complete);
        assert_eq!(found.meta().frames, SEGMENT_FRAMES);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_an_entry_whose_manifest_cannot_be_rewritten_when_loaded_then_it_still_plays() {
        let root = temp_cache("read-only");
        let cache = Cache::new(root.clone());
        cache
            .store("key", &render_of(240), 12_000, &fields())
            .expect("storing");
        let dir = cache.entry_dir("key");
        let original = std::fs::metadata(&dir).expect("stat").permissions();
        let mut read_only_permissions = original.clone();
        read_only_permissions.set_readonly(true);
        std::fs::set_permissions(&dir, read_only_permissions).expect("making the entry read-only");

        let loaded = cache.load("key").expect("a read-only entry still loads");

        std::fs::set_permissions(&dir, original).expect("restoring the permissions");
        assert!(loaded.is_some());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_two_models_when_found_by_source_then_the_settings_match_and_other_source_does_not() {
        let root = temp_cache("find");
        let cache = Cache::new(root.clone());
        let render = render_of(4_800);
        let a = Fields {
            model_id: "model-a".into(),
            ..fields()
        };
        let b = Fields {
            model_id: "model-b".into(),
            ..fields()
        };
        cache
            .store(&a.key(), &render, 15_328, &a)
            .expect("storing a");
        cache
            .store(&b.key(), &render, 15_328, &b)
            .expect("storing b");

        // The model id is ignored: on a machine without the model it cannot be computed,
        // and either entry is the same source file and settings.
        let found = cache
            .find_segments(&fields())
            .expect("searching")
            .expect("an entry");
        assert_eq!(found.meta().frames, 4_800);
        assert_eq!(found.meta().source_hash, "source");

        let other = Fields {
            source_hash: "different".into(),
            ..fields()
        };
        assert!(cache.find_segments(&other).expect("searching").is_none());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_an_entry_without_audio_fields_when_annotated_then_it_can_be_found() {
        let root = temp_cache("annotate");
        let cache = Cache::new(root.clone());
        let render = render_of(2_400);
        let key = fields().key();
        let blank = Fields {
            source_hash: String::new(),
            ..fields()
        };
        cache.store(&key, &render, 12_000, &blank).expect("storing");

        assert!(
            cache.find_segments(&fields()).expect("searching").is_none(),
            "an entry without its fields cannot match"
        );
        cache.annotate(&key, &fields()).expect("annotating");
        assert!(
            cache.find_segments(&fields()).expect("searching").is_some(),
            "the annotation makes it findable"
        );
        let meta = stored_meta(&cache, &key);
        assert_eq!(meta.source_hash, "source");
        assert_eq!(meta.segments.len(), 1, "annotating keeps the segment table");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_the_shipped_settings_when_a_segment_is_derived_then_it_holds_seven_steps() {
        assert_eq!(
            frames_per_segment(245_760, 48_000).expect("a segment"),
            1_384_320
        );
        assert!(frames_per_segment(48_000, 48_000).is_err(), "a zero step");
        assert!(
            frames_per_segment(24_000, 48_000).is_err(),
            "overlap is longer"
        );
        assert!(
            frames_per_segment(usize::MAX, 0).is_err(),
            "a segment length that overflows"
        );
    }

    #[test]
    fn given_the_same_inputs_when_keyed_twice_then_the_key_matches_and_a_setting_change_moves_it() {
        let a = key("source", 245_760, 48_000, "model", false);
        assert_eq!(a, key("source", 245_760, 48_000, "model", false));
        assert_eq!(a, fields().key());
        // Pinned so a change to the hash inputs shows up as a changed key, not as a silent
        // reuse of every v9 entry. The value is run-independent: blake3 over the six parts.
        assert_eq!(a, "fa3eb48f3cfa0fc219872eb40939f0c3");
        assert_ne!(a, key("source", 122_880, 48_000, "model", false));
        assert_ne!(a, key("source", 245_760, 24_000, "model", false));
        assert_ne!(a, key("source", 245_760, 48_000, "other-model", false));
        assert_ne!(a, key("source", 245_760, 48_000, "model", true));
        assert_eq!(a.len(), 32);
    }

    #[test]
    fn given_a_missing_models_directory_when_hashed_then_it_still_produces_an_id() {
        let id = model_id(Path::new("/nonexistent/flacify/models"));
        assert_eq!(id.len(), 16);
        assert_eq!(id, model_id(Path::new("/nonexistent/flacify/models")));
    }

    /// PCM that both containers carry unchanged: these samples already sit on the 24-bit
    /// grid, so a WAV and a FLAC of them hold the same audio.
    fn pcm_for_containers() -> Vec<f32> {
        let mut planar = Vec::with_capacity(4_800 * 2);
        for channel in 0..2 {
            for frame in 0..4_800 {
                planar.push(((frame * 31 + channel * 7) % 512 - 256) as f32 * QUANTUM);
            }
        }
        planar
    }

    #[test]
    fn given_the_same_pcm_as_wav_and_as_flac_when_hashed_then_the_hashes_differ_and_a_byte_copy_matches(
    ) {
        let root = temp_cache("source-hash-containers");
        std::fs::create_dir_all(&root).expect("creating the test directory");
        let wav = root.join("same-pcm.wav");
        let flac = root.join("same-pcm.flac");
        crate::wav::write(&wav, 48_000, 2, &pcm_for_containers()).expect("writing a wav");
        crate::flac::write(&flac, 48_000, 2, &pcm_for_containers()).expect("writing a flac");

        let wav_hash = source_hash(&wav).expect("hashing the wav");
        let flac_hash = source_hash(&flac).expect("hashing the flac");
        assert_ne!(
            wav_hash, flac_hash,
            "the key is the container, not the audio"
        );

        let copy = root.join("copy.wav");
        std::fs::copy(&wav, &copy).expect("copying the wav");
        assert_eq!(
            source_hash(&copy).expect("hashing the copy"),
            wav_hash,
            "the same bytes hash the same"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_file_of_several_chunks_when_hashed_then_it_is_blake3_of_all_its_bytes_in_32_characters(
    ) {
        let root = temp_cache("source-hash-bytes");
        std::fs::create_dir_all(&root).expect("creating the test directory");
        let path = root.join("bytes.bin");
        let bytes: Vec<u8> = (0..HASH_CHUNK * 2 + 3).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &bytes).expect("writing the file");

        let expected = blake3::hash(&bytes).to_hex().to_string();
        let actual = source_hash(&path).expect("hashing the file");
        assert_eq!(actual, expected[..32]);
        assert_eq!(actual.len(), 32);
        std::fs::remove_dir_all(root).ok();
    }

    /// Store a one-segment entry and set its LRU clock, so the order under test is the
    /// test's decision and not the wall clock's. Returns the bytes the entry occupies.
    fn store_one(cache: &Cache, key: &str, last_used: u64) -> u64 {
        cache
            .store(
                key,
                &render_of(SEGMENT_FRAMES),
                12_000,
                &fields_with_short_segments(),
            )
            .expect("storing");
        let mut meta = stored_meta(cache, key);
        meta.last_used = last_used;
        store_meta(cache, key, &meta);
        crate::setup::dir_size(&cache.entry_dir(key))
    }

    fn names(removals: &[Removal]) -> Vec<&str> {
        removals
            .iter()
            .map(|removal| removal.name.as_str())
            .collect()
    }

    /// A cache root with a wav in it and a built tier under the hash the entry records.
    fn tier_fixture(name: &str, source_hash: &str) -> (PathBuf, Cache, crate::source::Tier) {
        let root = temp_cache(name);
        std::fs::create_dir_all(&root).expect("creating the cache root");
        let wav = root.join("source.wav");
        crate::wav::write(&wav, 48_000, 2, &pcm_for_containers()).expect("writing the source");
        let tier = crate::source::Tier::new(&root, SEGMENT_FRAMES).expect("a tier");
        let entry = tier
            .build_or_reuse(&wav, source_hash)
            .expect("building the tier");
        drop(entry);
        (root.clone(), Cache::new(root), tier)
    }

    #[test]
    fn given_a_cache_under_its_cap_when_evicted_then_nothing_is_removed() {
        let root = temp_cache("evict-under-cap");
        let cache = Cache::new(root.clone());
        store_one(&cache, "key", 100);
        let usage = cache.usage();

        let plan = cache.eviction_plan(usage.total());
        assert!(plan.sweeps.is_empty(), "{:?}", plan.sweeps);
        assert!(plan.tiers.is_empty(), "{:?}", plan.tiers);
        assert!(plan.entries.is_empty(), "{:?}", plan.entries);
        assert_eq!(plan.after, usage.total());

        let outcome = cache.evict(&plan).expect("evicting nothing");
        assert_eq!(outcome.removals(), 0);
        assert_eq!(outcome.after, usage.total());
        assert!(cache.load("key").expect("loading").is_some());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_three_entries_over_the_cap_when_evicted_then_only_the_oldest_goes_and_the_total_is_under_the_cap(
    ) {
        let root = temp_cache("evict-cap");
        let cache = Cache::new(root.clone());
        let oldest = store_one(&cache, "a", 100);
        store_one(&cache, "b", 200);
        store_one(&cache, "c", 300);
        let usage = cache.usage();
        assert_eq!(usage.rendered.entries, 3);
        let cap = usage.total() - oldest;

        let plan = cache.eviction_plan(cap);
        assert_eq!(names(&plan.entries), ["a"]);
        assert_eq!(plan.after, usage.total() - oldest);

        let outcome = cache.evict(&plan).expect("evicting");
        assert_eq!(names(&outcome.entries), ["a"]);
        assert!(outcome.after <= cap, "{} > {cap}", outcome.after);
        assert_eq!(cache.usage().rendered.entries, 2);
        assert!(cache.load("a").expect("a miss").is_none());
        assert!(cache.load("b").expect("loading").is_some());
        assert!(cache.load("c").expect("loading").is_some());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_partial_entry_older_than_a_complete_one_when_evicted_then_the_partial_goes_first() {
        let root = temp_cache("evict-partial");
        let cache = Cache::new(root.clone());
        let (fields, frames, _) = four_segments();
        let partial_bytes;
        {
            let mut writer = cache
                .begin("half", &spec(frames, &fields))
                .expect("beginning a partial entry");
            writer
                .write_segment(0, &render_of(SEGMENT_FRAMES).planar)
                .expect("writing a segment");
            writer.abandon();
            let mut meta = stored_meta(&cache, "half");
            meta.last_used = 100;
            store_meta(&cache, "half", &meta);
            partial_bytes = crate::setup::dir_size(&cache.entry_dir("half"));
        }
        store_one(&cache, "done", 200);
        let usage = cache.usage();
        assert_eq!(usage.partial.entries, 1);
        assert_eq!(usage.rendered.entries, 1);

        let plan = cache.eviction_plan(usage.total() - partial_bytes);
        assert_eq!(names(&plan.entries), ["half"], "the LRU clock decides");
        assert_eq!(plan.entries[0].bytes, partial_bytes);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_two_entries_when_the_older_is_read_then_the_newer_is_evicted_instead() {
        let root = temp_cache("evict-lru");
        let cache = Cache::new(root.clone());
        let bytes = store_one(&cache, "old", 100);
        store_one(&cache, "new", 200);
        let total = cache.usage().total();
        let cap = total - bytes;

        let plan = cache.eviction_plan(cap);
        assert_eq!(
            names(&plan.entries),
            ["old"],
            "stored oldest, so chosen first"
        );

        // Reading an entry is using it, and the manifest says so.
        cache.segments("old").expect("opening").expect("an entry");
        let total = cache.usage().total();
        let cap = total - crate::setup::dir_size(&cache.entry_dir("new"));
        let plan = cache.eviction_plan(cap);
        assert_eq!(
            names(&plan.entries),
            ["new"],
            "the read moved old out of the way"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_an_entry_open_for_reading_when_evicted_then_it_is_kept_and_named_in_the_plan() {
        let root = temp_cache("evict-open");
        let cache = Cache::new(root.clone());
        store_one(&cache, "key", 100);

        let reader = cache.segments("key").expect("opening").expect("an entry");
        let plan = cache.eviction_plan(0);
        assert!(plan.entries.is_empty(), "{:?}", plan.entries);
        assert_eq!(names(&plan.open), ["key"]);

        let outcome = cache.evict(&plan).expect("evicting");
        assert_eq!(outcome.removals(), 0);
        assert!(outcome.after > 0, "the cap is not met and the plan says so");
        assert!(cache.entry_dir("key").is_dir());
        assert!(
            root.join(CLAIMS).join(entry_claim("key")).exists(),
            "the claim is on disk while the handle lives"
        );
        drop(reader);
        assert!(
            !root.join(CLAIMS).join(entry_claim("key")).exists(),
            "closing the handle clears the claim"
        );

        let plan = cache.eviction_plan(0);
        assert_eq!(
            names(&plan.entries),
            ["key"],
            "with the reader gone it goes"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_two_orphan_tiers_when_one_covers_the_excess_then_only_that_one_is_dropped() {
        let root = temp_cache("evict-two-tiers");
        std::fs::create_dir_all(&root).expect("creating the cache root");
        let wav = root.join("source.wav");
        crate::wav::write(&wav, 48_000, 2, &pcm_for_containers()).expect("writing the source");
        let tier = crate::source::Tier::new(&root, SEGMENT_FRAMES).expect("a tier");
        for hash in ["a", "b"] {
            drop(tier.build_or_reuse(&wav, hash).expect("building a tier"));
        }
        let cache = Cache::new(root.clone());
        let usage = cache.usage();
        assert_eq!(usage.source.entries, 2);
        let cap = usage.total() - usage.source.bytes / 2;

        let plan = cache.eviction_plan(cap);
        assert_eq!(plan.tiers.len(), 1, "one tier is enough: {:?}", plan.tiers);
        assert!(plan.entries.is_empty());
        assert!(plan.after <= cap);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_two_renders_each_with_a_tier_when_one_tier_covers_the_excess_then_only_it_goes() {
        let (root, cache, tier) = tier_fixture("evict-two-renders", "a");
        let wav = root.join("source.wav");
        drop(
            tier.build_or_reuse(&wav, "b")
                .expect("building the second tier"),
        );
        for (hash, model) in [("a", "model-a"), ("b", "model-b")] {
            let fields = Fields {
                source_hash: hash.into(),
                model_id: model.into(),
                ..fields_with_short_segments()
            };
            cache
                .store(&fields.key(), &render_of(SEGMENT_FRAMES), 12_000, &fields)
                .expect("storing");
        }
        let usage = cache.usage();
        assert_eq!(usage.source.entries, 2);
        assert_eq!(usage.rendered.entries, 2);
        let cap = usage.total() - usage.source.bytes / 2;

        let plan = cache.eviction_plan(cap);
        assert_eq!(plan.tiers.len(), 1, "one tier is enough: {:?}", plan.tiers);
        assert!(
            plan.entries.is_empty(),
            "both renders stay: {:?}",
            plan.entries
        );
        assert!(plan.after <= cap);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_two_entries_that_share_a_tier_when_evicted_then_the_tier_is_named_and_counted_once() {
        let (root, cache, tier) = tier_fixture("evict-shared-tier", "shared");
        let first = Fields {
            source_hash: "shared".into(),
            model_id: "model-a".into(),
            ..fields_with_short_segments()
        };
        let second = Fields {
            model_id: "model-b".into(),
            ..first.clone()
        };
        for fields in [&first, &second] {
            cache
                .store(&fields.key(), &render_of(SEGMENT_FRAMES), 12_000, fields)
                .expect("storing");
        }
        assert_eq!(cache.usage().rendered.entries, 2);

        let plan = cache.eviction_plan(0);
        assert_eq!(
            names(&plan.tiers),
            ["source/shared"],
            "the tier is one directory"
        );
        assert_eq!(plan.entries.len(), 2);
        // The wav the fixture keeps in the cache root is not something eviction removes, so
        // it is exactly what the plan says is left: a tier counted twice would say zero.
        let wav = std::fs::metadata(root.join("source.wav"))
            .expect("the fixture's wav")
            .len();
        assert_eq!(plan.after, wav);
        cache.evict(&plan).expect("evicting");
        assert!(!tier.entry_dir("shared").exists());
        let usage = cache.usage();
        assert_eq!(usage.rendered.entries, 0);
        assert_eq!(usage.source.entries, 0);
        std::fs::remove_dir_all(root).ok();
    }

    #[cfg(unix)]
    #[test]
    fn given_an_eviction_that_cannot_finish_then_the_entry_is_gone_rather_than_half_removed() {
        let root = temp_cache("evict-read-only");
        let cache = Cache::new(root.clone());
        store_one(&cache, "key", 100);
        let dir = cache.entry_dir("key");
        let original = std::fs::metadata(&dir).expect("stat").permissions();
        let mut read_only = original.clone();
        read_only.set_readonly(true);
        std::fs::set_permissions(&dir, read_only).expect("making the entry read-only");

        let plan = cache.eviction_plan(0);
        let error = cache
            .evict(&plan)
            .expect_err("a read-only directory cannot be emptied");
        assert!(format!("{error:#}").contains("removing"), "{error:#}");

        // The rename came first, so the entry is out of the namespace even though its bytes
        // are still there: a miss, not an error and not a half entry.
        assert!(!dir.exists(), "the entry directory is gone from its name");
        assert!(cache.load("key").expect("a miss, not an error").is_none());
        let leftover = std::fs::read_dir(&root)
            .expect("listing")
            .flatten()
            .map(|entry| entry.path())
            .find(|path| file_name(path).starts_with(&format!("key{EVICTING}")))
            .expect("the renamed directory is still there");
        std::fs::set_permissions(&leftover, original).expect("restoring the permissions");

        let plan = cache.eviction_plan(0);
        assert_eq!(names(&plan.sweeps), [file_name(&leftover)]);
        cache.evict(&plan).expect("sweeping what the failure left");
        assert!(!leftover.exists());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_claim_older_than_the_stale_window_when_evicted_then_the_entry_is_not_pinned_forever()
    {
        let root = temp_cache("evict-stale-claim");
        let cache = Cache::new(root.clone());
        store_one(&cache, "key", 100);
        let claims = root.join(CLAIMS);
        std::fs::create_dir_all(&claims).expect("creating the claim directory");
        let marker = claims.join(entry_claim("key"));

        // A claim written by a process that died an hour ago.
        std::fs::write(
            &marker,
            format!("999999 1 {}\n", now() - STALE_CLAIM_SECS - 1),
        )
        .expect("writing a stale claim");
        let plan = cache.eviction_plan(0);
        assert_eq!(
            names(&plan.entries),
            ["key"],
            "a stale claim protects nothing"
        );

        // A claim written a moment ago is live, whatever the process that wrote it.
        std::fs::write(&marker, format!("999999 2 {}\n", now())).expect("writing a live claim");
        let plan = cache.eviction_plan(0);
        assert!(plan.entries.is_empty(), "{:?}", plan.entries);
        assert_eq!(names(&plan.open), ["key"]);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_reader_who_arrives_after_the_plan_when_evicted_then_the_entry_is_skipped() {
        let root = temp_cache("evict-raced");
        let cache = Cache::new(root.clone());
        store_one(&cache, "key", 100);
        let plan = cache.eviction_plan(0);
        assert_eq!(names(&plan.entries), ["key"]);

        let reader = cache.segments("key").expect("opening").expect("an entry");
        let outcome = cache.evict(&plan).expect("evicting");
        assert_eq!(names(&outcome.skipped), ["key"]);
        assert_eq!(outcome.removals(), 0);
        assert!(cache.entry_dir("key").is_dir());
        drop(reader);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_an_entry_with_a_source_tier_when_the_tier_covers_the_excess_then_the_render_stays() {
        let (root, cache, tier) = tier_fixture("evict-tier", "shared");
        let fields = Fields {
            source_hash: "shared".into(),
            ..fields_with_short_segments()
        };
        cache
            .store("key", &render_of(SEGMENT_FRAMES), 12_000, &fields)
            .expect("storing");
        let usage = cache.usage();
        assert_eq!(usage.source.entries, 1);
        assert_eq!(usage.rendered.entries, 1);
        assert!(usage.source.bytes > 0);

        let cap = usage.total() - usage.source.bytes;
        let plan = cache.eviction_plan(cap);
        assert_eq!(names(&plan.tiers), ["source/shared"]);
        assert!(
            plan.entries.is_empty(),
            "the tier was enough, so the render is kept: {:?}",
            plan.entries
        );

        let outcome = cache.evict(&plan).expect("evicting");
        assert_eq!(outcome.after, cap);
        assert!(!tier.entry_dir("shared").exists());
        assert_eq!(
            cache
                .load("key")
                .expect("loading")
                .expect("the render")
                .meta
                .frames,
            SEGMENT_FRAMES
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_an_orphan_tier_with_no_render_when_evicted_then_it_goes_before_any_entry() {
        let (root, cache, tier) = tier_fixture("evict-orphan-tier", "orphan");
        store_one(&cache, "key", 100);
        let usage = cache.usage();
        assert_eq!(usage.source.entries, 1);

        let plan = cache.eviction_plan(0);
        assert_eq!(names(&plan.tiers), ["source/orphan"]);
        assert_eq!(names(&plan.entries), ["key"], "the entry follows its tier");
        cache.evict(&plan).expect("evicting");
        assert!(!tier.entry_dir("orphan").exists());
        assert!(!cache.entry_dir("key").exists());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_tier_a_reader_holds_when_evicted_then_the_tier_is_kept() {
        let (root, cache, tier) = tier_fixture("evict-tier-open", "held");
        let reader = crate::source::Tier::new(&root, SEGMENT_FRAMES)
            .expect("a tier")
            .build_or_reuse(&root.join("source.wav"), "held")
            .expect("reusing the entry");

        let plan = cache.eviction_plan(0);
        assert!(plan.tiers.is_empty(), "{:?}", plan.tiers);
        assert!(
            plan.open
                .iter()
                .any(|removal| removal.name == "source/held"),
            "{:?}",
            plan.open
        );
        let outcome = cache.evict(&plan).expect("evicting");
        assert_eq!(outcome.removals(), 0);
        assert!(tier.entry_dir("held").is_dir());
        drop(reader);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_staging_directory_a_render_is_writing_when_evicted_then_it_is_left_alone() {
        let root = temp_cache("evict-staging-claim");
        let cache = Cache::new(root.clone());
        let staging = root.join("key.tmp");
        std::fs::create_dir_all(&staging).expect("creating a staging directory");
        std::fs::write(staging.join("seg-000000.flac.tmp"), vec![0u8; 128]).expect("writing");

        // What a render holds while it fills the staging directory: the claim of the entry
        // that directory publishes.
        let writer = cache.hold_open(&staging);
        let plan = cache.eviction_plan(u64::MAX);
        assert!(plan.sweeps.is_empty(), "{:?}", plan.sweeps);
        assert_eq!(names(&plan.open), ["key.tmp"]);
        drop(writer);

        let plan = cache.eviction_plan(u64::MAX);
        assert_eq!(
            names(&plan.sweeps),
            ["key.tmp"],
            "with the writer gone it goes"
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_an_unknown_version_entry_and_a_staging_directory_when_evicted_then_both_are_swept_and_counted(
    ) {
        let root = temp_cache("evict-sweep");
        let cache = Cache::new(root.clone());
        store_one(&cache, "key", 100);
        // An entry from the version before this one.
        store_one(&cache, "old-entry", 100);
        let mut old_meta = stored_meta(&cache, "old-entry");
        old_meta.version = "flacify-cache-8".into();
        store_meta(&cache, "old-entry", &old_meta);
        // A staging directory a killed render left.
        let staging = root.join("killed.tmp");
        std::fs::create_dir_all(&staging).expect("creating a staging directory");
        std::fs::write(staging.join("seg-000000.flac"), vec![0u8; 4_096]).expect("writing");

        let usage = cache.usage();
        assert_eq!(usage.unknown.entries, 1, "the other version is unreadable");
        assert_eq!(usage.stale.entries, 1);

        let plan = cache.eviction_plan(usage.total());
        assert_eq!(names(&plan.sweeps), ["killed.tmp", "old-entry"]);
        assert!(
            plan.entries.is_empty(),
            "under the cap, so nothing else goes"
        );

        let outcome = cache.evict(&plan).expect("evicting");
        assert_eq!(outcome.swept.len(), 2);
        assert!(outcome.freed > 0);
        assert!(!staging.exists());
        assert!(!cache.entry_dir("old-entry").exists());
        let usage = cache.usage();
        assert_eq!(usage.unknown.entries, 0);
        assert_eq!(usage.stale.entries, 0);
        assert_eq!(usage.rendered.entries, 1, "the readable entry stays");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_an_eviction_interrupted_after_the_rename_then_nothing_is_readable_and_the_next_sweep_removes_it(
    ) {
        let root = temp_cache("evict-interrupted");
        let cache = Cache::new(root.clone());
        store_one(&cache, "key", 100);
        // What `remove_tree` leaves when the process dies between its rename and its delete.
        let doomed = root.join(format!("key{EVICTING}{}", std::process::id()));
        std::fs::rename(cache.entry_dir("key"), &doomed).expect("moving the entry aside");

        assert!(
            cache.load("key").expect("a miss, not an error").is_none(),
            "a half-deleted entry is never read"
        );
        assert!(cache
            .segments("key")
            .expect("a miss, not an error")
            .is_none());
        assert!(cache
            .find_segments(&fields_with_short_segments())
            .expect("a miss, not an error")
            .is_none());
        let usage = cache.usage();
        assert_eq!(usage.stale.entries, 1);
        assert_eq!(usage.rendered.entries, 0);

        let plan = cache.eviction_plan(0);
        assert_eq!(
            names(&plan.sweeps),
            [format!("key{EVICTING}{}", std::process::id())]
        );
        cache.evict(&plan).expect("sweeping");
        assert!(!doomed.exists());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_an_entry_with_a_file_its_manifest_does_not_name_when_measured_then_the_bytes_are_reported(
    ) {
        let root = temp_cache("usage-unaccounted");
        let cache = Cache::new(root.clone());
        store_one(&cache, "key", 100);
        assert_eq!(cache.usage().unaccounted_bytes, 0);

        let stray = cache.entry_dir("key").join("seg-000001.flac.tmp");
        std::fs::write(&stray, vec![7u8; 1_000]).expect("writing a stray file");
        let usage = cache.usage();
        assert_eq!(usage.unaccounted_bytes, 1_000);
        assert_eq!(usage.overstated_entries, 0);
        assert_eq!(usage.rendered.entries, 1, "it is still one entry");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn given_a_manifest_that_names_more_than_its_files_hold_when_measured_then_it_is_reported() {
        let root = temp_cache("usage-overstated");
        let cache = Cache::new(root.clone());
        store_one(&cache, "key", 100);
        let mut meta = stored_meta(&cache, "key");
        meta.segments[0].bytes += 10_000;
        store_meta(&cache, "key", &meta);

        let usage = cache.usage();
        assert_eq!(usage.overstated_entries, 1);
        assert_eq!(usage.unaccounted_bytes, 0);
        std::fs::remove_dir_all(root).ok();
    }
}
