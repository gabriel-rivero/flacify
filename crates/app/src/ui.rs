//! The window: one track, one switch, and the truth about what has been generated.
//!
//! Playback starts immediately; the renderer fills a frontier behind it. The switch is
//! only enabled where generated audio actually exists, so it can never claim something
//! the file does not have — and once the frontier is comfortably ahead it flips itself,
//! because the point of the product is hearing the difference, not finding the button.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use eframe::egui;

use crate::device::{self, AudioEngine};
use crate::first_run;
use crate::frontier::Frontier;
use crate::playlist;
use crate::process::Cancel;
use crate::render_status::RenderStatus;
use crate::segments::Resident;
use crate::session::{needs_setup, LivePlayback, Playback, SegmentView, Session, SinkRoute};
use crate::setup;
use crate::sink::{CaptureState, SILENT_HINT};
use crate::source::SecondPeaks;
use crate::spectrum;
use crate::youtube;

/// What the window needs to know about the track, none of it expensive.
pub struct TrackInfo {
    /// File name, always present, used when there are no tags.
    pub name: String,
    pub title: Option<String>,
    pub artist: Option<String>,
    pub cutoff_hz: u32,
    pub audio_secs: f64,
    pub device: String,
    pub from_cache: bool,
    /// Loudness gain applied to the generated side, for the status line.
    pub gain_db: f64,
}

/// A remote entry being fetched on its own thread, so the window never blocks on yt-dlp.
struct PendingFetch {
    index: usize,
    label: String,
    cancel: Cancel,
    handle: JoinHandle<anyhow::Result<playlist::Resolved>>,
}

/// A pasted URL being listed on its own thread. A playlist or a radio mix is a
/// `yt-dlp --flat-playlist` call, and the window has a track to keep playing while it runs.
struct PendingQueue {
    label: String,
    handle: JoinHandle<anyhow::Result<playlist::Queue>>,
}

/// A pasted URL from the empty window being listed and downloaded on its own thread, so
/// the window that has nothing to play does not freeze while yt-dlp works.
struct PendingOpen {
    label: String,
    handle: JoinHandle<anyhow::Result<(playlist::Queue, playlist::Resolved)>>,
}

/// Whether the radio toggle applies to a pasted link: it seeds a mix from one video, and a
/// playlist URL is a queue of its own.
fn radio_requested(text: &str, radio_seed: bool) -> bool {
    radio_seed && youtube::parse(text).is_some() && !youtube::is_playlist_url(text)
}

/// What the current queue entry is doing.
enum LoadState {
    Ready,
    Fetching { label: String },
    Failed(String),
}

pub struct PlayerApp {
    /// One worker for the whole window, so a dropped file does not reload the model.
    session: Session,
    engine: AudioEngine,
    frontier: Option<Arc<Frontier>>,
    /// Present only when this window is watching a background render.
    status: Option<Arc<RenderStatus>>,
    /// Present only when this window is playing a live stream (`play --live`).
    live: Option<LivePlayback>,
    /// Present only when this window is the sink: the route and its drift correction.
    sink: Option<SinkRoute>,
    info: TrackInfo,
    /// The file on screen, so an error names it and a reload knows what to ask for.
    path: PathBuf,
    /// Cancels this file's renderer when another file replaces it.
    cancel: Option<Arc<AtomicBool>>,
    error: Option<String>,
    promote_margin: usize,
    /// Playback starts when this many frames of generated audio exist, so the first
    /// note is already the product and the frontier stays ahead from there.
    prebuffer_frames: usize,
    started: bool,
    opened_at: std::time::Instant,
    /// The window title names the file; set once per load, not every frame.
    title_dirty: bool,
    /// Per-bar RMS of the original, normalized, computed once per file.
    waveform: Vec<f32>,
    /// Device frames each waveform bar covers: at least one second, never more bars than
    /// pixels.
    waveform_bar_frames: usize,
    /// Present when the track plays from segments: the peaks and the resident segment
    /// its waveform and spectrum read.
    segments: Option<SegmentView>,
    /// The queue this playback belongs to; `None` keeps a bare single track.
    queue: Option<playlist::Queue>,
    /// A remote entry being fetched, for the cursor or as a prefetch of the next one.
    pending: Option<PendingFetch>,
    /// A downloaded entry waiting for its turn.
    prefetched: Option<(usize, playlist::Resolved)>,
    /// What the current remote entry is doing.
    load_state: LoadState,
    /// Set once an end-of-track has advanced the queue, so a pending fetch is not
    /// advanced again on every frame.
    auto_advanced: bool,
    /// The URL box: a pasted link becomes the queue, or seeds a radio from its video.
    url: String,
    /// Whether the box seeds a radio instead of playing the one video.
    radio_seed: bool,
    /// A pasted URL being listed, so the window keeps drawing while yt-dlp works.
    pending_queue: Option<PendingQueue>,
}

/// No more waveform bars than this; a pixel is narrower than a bar worth drawing.
const WAVEFORM_MAX_BARS: usize = 512;

/// How long a loaded renderer may take to reach the prebuffer before the window starts on
/// the original anyway. It applies only once the model has finished loading: a cold load is
/// not slowness, and waiting on it is what the product promises.
pub(crate) const START_BUDGET_SECS: f64 = 30.0;

/// The parts of the window, in the order they draw. The match in `draw_transport` is
/// exhaustive, so a new variant does not compile until its drawing exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TransportSection {
    Heading,
    Toggle,
    Queue,
    Url,
    Spectrum,
    Waveform,
    KeyHint,
    Playback,
    Progress,
    RenderStatus,
    Live,
    Sink,
    Metadata,
    Disclaimer,
    StreamErrors,
    CpuWarning,
    Error,
}

/// The draw order. Add a variant here as well as to `TransportSection`; a list entry with
/// no arm is a compile error, an arm with no list entry is a section that never draws.
const TRANSPORT_SECTIONS: [TransportSection; 17] = [
    TransportSection::Heading,
    TransportSection::Toggle,
    TransportSection::Queue,
    TransportSection::Url,
    TransportSection::Spectrum,
    TransportSection::Waveform,
    TransportSection::KeyHint,
    TransportSection::Playback,
    TransportSection::Progress,
    TransportSection::RenderStatus,
    TransportSection::Live,
    TransportSection::Sink,
    TransportSection::Metadata,
    TransportSection::Disclaimer,
    TransportSection::StreamErrors,
    TransportSection::CpuWarning,
    TransportSection::Error,
];

impl PlayerApp {
    pub fn new(session: Session, playback: Playback) -> Self {
        let engine = playback.engine;
        let segments = playback.segments;
        let (waveform, waveform_bar_frames) = waveform_for(
            engine.controls.frames,
            engine.device_channels,
            engine.device_rate,
            &engine.display_original,
            peaks_of(segments.as_ref()),
        );
        // One second of slack before the switch is offered, so a toggle cannot land on
        // the edge of what exists and fall back mid-ramp. A live stream has no frontier to
        // run ahead of: its own coverage is the only gate.
        let promote_margin = match playback.live.is_some() {
            true => 0,
            false => engine.device_rate as usize,
        };
        let prebuffer_frames = (engine.device_rate as usize).saturating_mul(2);
        Self {
            session,
            engine,
            frontier: playback.frontier,
            status: playback.status,
            live: playback.live,
            sink: playback.sink,
            info: playback.info,
            path: playback.path,
            cancel: playback.cancel,
            error: None,
            promote_margin,
            prebuffer_frames,
            started: false,
            opened_at: std::time::Instant::now(),
            title_dirty: true,
            waveform,
            waveform_bar_frames,
            segments,
            queue: None,
            pending: None,
            prefetched: None,
            load_state: LoadState::Ready,
            auto_advanced: false,
            url: String::new(),
            radio_seed: false,
            pending_queue: None,
        }
    }

    /// A playback that belongs to a queue: the window can advance it and move through it.
    pub fn with_queue(session: Session, playback: Playback, queue: playlist::Queue) -> Self {
        let mut app = Self::new(session, playback);
        app.queue = Some(queue);
        // Fetch the next remote entry while the first one plays; nothing else would start
        // that download until the current track ended.
        app.prefetch_next();
        app
    }

    /// Replace what is playing, keeping the worker. A failed load leaves the current
    /// file alone and says why, instead of stopping playback with nothing to show.
    /// Returns whether the new track is playing.
    fn load(&mut self, path: &Path) -> bool {
        let mut playback = match self.session.open(path) {
            Ok(playback) => playback,
            Err(error) => {
                self.error = Some(format!("could not play {}: {error:#}", path.display()));
                return false;
            }
        };
        // The new file owns the worker's next window; the old renderer must let go.
        if let Some(cancel) = self.cancel.take() {
            cancel.store(true, Ordering::Relaxed);
        }
        self.cancel = playback.cancel.take();
        // Dropping the old engine stops its stream.
        self.engine = playback.engine;
        self.frontier = playback.frontier;
        self.status = playback.status;
        self.live = playback.live;
        self.sink = playback.sink;
        self.info = playback.info;
        self.path = playback.path;
        self.segments = playback.segments;
        self.promote_margin = match self.live {
            Some(_) => 0,
            None => self.engine.device_rate as usize,
        };
        self.prebuffer_frames = (self.engine.device_rate as usize).saturating_mul(2);
        let (waveform, bar_frames) = waveform_for(
            self.engine.controls.frames,
            self.engine.device_channels,
            self.engine.device_rate,
            &self.engine.display_original,
            peaks_of(self.segments.as_ref()),
        );
        self.waveform = waveform;
        self.waveform_bar_frames = bar_frames;
        self.started = false;
        self.opened_at = std::time::Instant::now();
        self.title_dirty = true;
        self.error = None;
        self.load_state = LoadState::Ready;
        self.auto_advanced = false;
        true
    }

    /// The directory remote entries are downloaded into, from the queue when there is one.
    fn downloads_dir(&self) -> PathBuf {
        if let Some(queue) = &self.queue {
            return queue.downloads().to_path_buf();
        }
        setup::Paths::discover()
            .map(|paths| paths.downloads_dir())
            .unwrap_or_default()
    }

    /// Play the queue's current entry: local files load now, remote ones are fetched on a
    /// thread so the window keeps drawing while yt-dlp works.
    fn play_current(&mut self) {
        let Some(queue) = &self.queue else {
            return;
        };
        let index = queue.cursor();
        let Some(entry) = queue.current().cloned() else {
            return;
        };
        match entry {
            playlist::Entry::Local { path, .. } => {
                self.load(&path);
            }
            playlist::Entry::Remote { request, label } => {
                if self.prefetched.as_ref().is_some_and(|(at, _)| *at == index) {
                    let (_, resolved) = self.prefetched.take().expect("just checked");
                    self.load_resolved(resolved, index);
                    return;
                }
                if self
                    .pending
                    .as_ref()
                    .is_some_and(|pending| pending.index == index)
                {
                    self.engine.controls.set_playing(false);
                    self.load_state = LoadState::Fetching { label };
                    return;
                }
                if let Some(pending) = self.pending.take() {
                    pending.cancel.cancel();
                }
                self.engine.controls.set_playing(false);
                self.spawn_fetch(index, request, label, true);
            }
        }
    }

    /// Start downloading one remote entry. `current` says whether the queue is waiting on
    /// it, which is what the strip reports; a prefetch is silent until it is needed.
    fn spawn_fetch(
        &mut self,
        index: usize,
        request: youtube::Request,
        label: String,
        current: bool,
    ) {
        let Some(downloads) = self
            .queue
            .as_ref()
            .map(|queue| queue.downloads().to_path_buf())
        else {
            return;
        };
        if current {
            self.load_state = LoadState::Fetching {
                label: label.clone(),
            };
        }
        let cancel = Cancel::default();
        let token = cancel.clone();
        let entry = playlist::Entry::Remote {
            request,
            label: label.clone(),
        };
        let handle =
            std::thread::spawn(move || playlist::resolve(&entry, &downloads, Some(&token)));
        self.pending = Some(PendingFetch {
            index,
            label,
            cancel,
            handle,
        });
    }

    /// Adopt a downloaded entry: open its file and let yt-dlp's names win over the tags.
    fn load_resolved(&mut self, resolved: playlist::Resolved, index: usize) {
        if self.load(&resolved.path) {
            if resolved.title.is_some() {
                self.info.title = resolved.title;
            }
            if resolved.artist.is_some() {
                self.info.artist = resolved.artist;
            }
        }
        self.load_state = LoadState::Ready;
        if index
            == self
                .queue
                .as_ref()
                .map_or(usize::MAX, playlist::Queue::cursor)
        {
            self.prefetch_next();
        }
    }

    /// Collect a finished background fetch, once.
    fn poll_fetch(&mut self) {
        let Some(pending) = self.pending.as_ref() else {
            return;
        };
        if !pending.handle.is_finished() {
            return;
        }
        let pending = self.pending.take().expect("just checked");
        let cursor = self.queue.as_ref().map(playlist::Queue::cursor);
        match pending.handle.join() {
            Ok(Ok(resolved)) => {
                if Some(pending.index) == cursor {
                    self.load_resolved(resolved, pending.index);
                } else {
                    self.prefetched = Some((pending.index, resolved));
                }
            }
            Ok(Err(error)) => {
                if Some(pending.index) == cursor {
                    self.load_state = LoadState::Failed(format!("{error:#}"));
                }
            }
            Err(_) => {
                if Some(pending.index) == cursor {
                    self.load_state =
                        LoadState::Failed(format!("fetching {} failed", pending.label));
                }
            }
        }
    }

    /// Fetch the next remote entry while the current one plays, so the transition costs
    /// nothing but the download that already ran.
    fn prefetch_next(&mut self) {
        if self.pending.is_some() || self.prefetched.is_some() {
            return;
        }
        let Some(queue) = &self.queue else {
            return;
        };
        let Some(index) = queue.next_index() else {
            return;
        };
        let Some(entry) = queue.at(index).cloned() else {
            return;
        };
        if let playlist::Entry::Remote { request, label } = entry {
            self.spawn_fetch(index, request, label, false);
        }
    }

    fn next_track(&mut self) {
        let advanced = self.queue.as_mut().is_some_and(playlist::Queue::advance);
        if advanced {
            self.begin_entry();
        }
    }

    fn previous_track(&mut self) {
        let rewound = self.queue.as_mut().is_some_and(playlist::Queue::rewind);
        if rewound {
            self.begin_entry();
        }
    }

    /// Stop the current render and start the queue's current entry.
    fn begin_entry(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            cancel.store(true, Ordering::Relaxed);
        }
        self.load_state = LoadState::Ready;
        self.play_current();
    }

    /// Replace the queue with one built from dropped paths.
    fn open_paths(&mut self, paths: &[PathBuf]) {
        let downloads = self.downloads_dir();
        let queue = match playlist::Queue::from_local(paths, &downloads) {
            Ok(queue) => queue,
            Err(error) => {
                self.error = Some(format!("{error:#}"));
                return;
            }
        };
        self.adopt_queue(queue);
    }

    /// A pasted URL becomes the queue, or a radio mix seeded from its video. The listing
    /// runs on a thread because a playlist or a radio expansion is a `yt-dlp` call, and
    /// the current track keeps playing while it runs.
    fn submit_url(&mut self) {
        let text = self.url.trim().to_string();
        if text.is_empty() || self.pending_queue.is_some() {
            return;
        }
        if youtube::parse(&text).is_none() {
            self.error = Some(format!("{text} is not a URL"));
            return;
        }
        let downloads = self.downloads_dir();
        // A playlist URL is already a queue, so radio applies only to a single video.
        let radio = radio_requested(&text, self.radio_seed);
        let label = text.clone();
        let handle =
            std::thread::spawn(move || playlist::Queue::from_url(&text, &downloads, radio, false));
        self.pending_queue = Some(PendingQueue { label, handle });
        self.error = None;
    }

    /// Collect a finished listing, once. A failure leaves the queue that is playing alone.
    fn poll_queue(&mut self) {
        let Some(pending) = self.pending_queue.as_ref() else {
            return;
        };
        if !pending.handle.is_finished() {
            return;
        }
        let pending = self.pending_queue.take().expect("just checked");
        match pending.handle.join() {
            Ok(Ok(queue)) => self.adopt_queue(queue),
            Ok(Err(error)) => self.error = Some(format!("{error:#}")),
            Err(_) => self.error = Some(format!("listing {} failed", pending.label)),
        }
    }

    /// Play a freshly built queue from its first entry.
    fn adopt_queue(&mut self, queue: playlist::Queue) {
        if let Some(pending) = self.pending.take() {
            pending.cancel.cancel();
        }
        self.prefetched = None;
        self.queue = Some(queue);
        self.url.clear();
        self.begin_entry();
    }

    /// Advance a queue when the track has ended, once per track.
    fn auto_advance(&mut self) {
        if self.auto_advanced || self.queue.is_none() || self.live.is_some() {
            return;
        }
        if !self.engine.controls.at_end() {
            return;
        }
        self.auto_advanced = true;
        let advanced = self.queue.as_mut().is_some_and(playlist::Queue::advance);
        if advanced {
            self.begin_entry();
        }
    }

    fn ready_frames(&self) -> usize {
        match &self.frontier {
            Some(frontier) => frontier.ready_frames(),
            None => self.engine.controls.frames,
        }
    }

    fn draw_transport(&mut self, ui: &mut egui::Ui) {
        // The first progressive render cannot know the device until the worker has loaded;
        // the panel keeps asking so the status line says `cuda` once it does instead of
        // reporting "rendering" for the life of the track.
        if self.info.device == "rendering" {
            if let Some(device) = self.session.worker().device() {
                self.info.device = device;
            }
        }
        // One read of the frontier, so two sections cannot disagree about this frame.
        let prepared_frames = self.ready_frames();
        for section in TRANSPORT_SECTIONS {
            match section {
                TransportSection::Heading => self.draw_heading(ui),
                TransportSection::Toggle => self.draw_toggle(ui, prepared_frames),
                TransportSection::Queue => self.draw_queue(ui),
                TransportSection::Url => self.draw_url(ui),
                TransportSection::Spectrum => self.draw_spectrum(ui),
                TransportSection::Waveform => self.draw_waveform(ui),
                TransportSection::KeyHint => self.draw_key_hint(ui),
                TransportSection::Playback => self.draw_playback(ui, prepared_frames),
                TransportSection::Progress => self.draw_progress(ui),
                TransportSection::RenderStatus => self.draw_render_status(ui, prepared_frames),
                TransportSection::Live => self.draw_live(ui),
                TransportSection::Sink => self.draw_sink(ui),
                TransportSection::Metadata => self.draw_metadata(ui),
                TransportSection::Disclaimer => self.draw_disclaimer(ui),
                TransportSection::StreamErrors => self.draw_stream_errors(ui),
                TransportSection::CpuWarning => self.draw_cpu_warning(ui),
                TransportSection::Error => self.draw_error(ui),
            }
        }
    }

    /// The track, and who made it.
    fn draw_heading(&mut self, ui: &mut egui::Ui) {
        ui.add_space(6.0);
        let heading = self
            .info
            .title
            .clone()
            .unwrap_or_else(|| self.info.name.clone());
        ui.label(egui::RichText::new(heading).size(17.0).strong());
        if let Some(artist) = &self.info.artist {
            ui.label(egui::RichText::new(artist).size(12.0).weak());
        }
        ui.add_space(8.0);
    }

    /// The switch. Enabled only where generated audio exists.
    fn draw_toggle(&mut self, ui: &mut egui::Ui, prepared_frames: usize) {
        let rate = self.engine.device_rate;
        let controls = Arc::clone(&self.engine.controls);
        let generated_here =
            prepared_frames >= controls.position_frames() + self.promote_margin.min(48_000);
        let live = controls.is_enhanced();
        // "0.0 s ready" while the weights load is indistinguishable from a wedged renderer,
        // so the loading state says what is actually happening instead.
        let loading = self.frontier.is_some() && !self.session.worker().is_loaded();
        let label = if !controls.has_enhanced() {
            "Full-band source".to_string()
        } else if loading {
            format!(
                "Loading the model… {:.0} s",
                self.opened_at.elapsed().as_secs_f64()
            )
        } else if !generated_here {
            format!(
                "Generating… {:.0} s ready",
                prepared_frames as f64 / f64::from(rate)
            )
        } else if live {
            "Hearing: generated top band".to_string()
        } else {
            "Hearing: original".to_string()
        };

        let button = egui::Button::new(egui::RichText::new(label).size(21.0))
            .min_size(egui::vec2(400.0, 68.0));
        if ui
            .add_enabled(controls.has_enhanced() && generated_here, button)
            .clicked()
        {
            controls.toggle();
        }
    }

    /// Where this track sits in the queue, and the two ways to move through it.
    fn draw_queue(&mut self, ui: &mut egui::Ui) {
        let Some(queue) = &self.queue else {
            return;
        };
        if queue.len() < 2 && self.pending.is_none() {
            return;
        }
        let indicator = queue.indicator();
        let current = queue
            .current()
            .map(|entry| entry.label().to_string())
            .unwrap_or_default();
        let next = queue
            .next_index()
            .and_then(|index| queue.at(index))
            .map(|entry| entry.label().to_string());
        let has_previous = queue.previous_index().is_some();
        let has_next = queue.next_index().is_some();
        let pending_next = self
            .pending
            .as_ref()
            .filter(|pending| Some(pending.index) != Some(queue.cursor()))
            .map(|pending| pending.label.clone());
        let state = match &self.load_state {
            LoadState::Fetching { label } => Some(format!("loading {label}…")),
            LoadState::Failed(error) => Some(format!("failed: {error}")),
            LoadState::Ready => None,
        };

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui
                .add_enabled(has_previous, egui::Button::new("◀"))
                .clicked()
            {
                self.previous_track();
            }
            ui.label(
                egui::RichText::new(format!("{indicator} · {current}"))
                    .weak()
                    .size(11.0),
            );
            if ui.add_enabled(has_next, egui::Button::new("▶")).clicked() {
                self.next_track();
            }
            if let Some(state) = state {
                ui.label(egui::RichText::new(state).weak().size(11.0));
            } else if let Some(label) = pending_next {
                ui.label(
                    egui::RichText::new(format!("fetching {label}"))
                        .weak()
                        .size(11.0),
                );
            } else if !has_next {
                ui.label(egui::RichText::new("end of queue").weak().size(11.0));
            } else if let Some(label) = next {
                ui.label(
                    egui::RichText::new(format!("next: {label}"))
                        .weak()
                        .size(11.0),
                );
            }
        });
    }

    /// The keys that do something, for the user without a mouse.
    fn draw_key_hint(&mut self, ui: &mut egui::Ui) {
        ui.add_space(6.0);
        ui.label(
            egui::RichText::new(
                "space toggles   ·   p plays   ·   ←/→ seek 5 s   ·   n/b track   ·   drop a file   ·   esc quits",
            )
            .weak()
            .size(11.0),
        );
    }

    /// The URL box: paste a link and it becomes the queue, or seeds a radio from its
    /// video. This is the only way into a remote source without a command line, which
    /// matters for the packaged app that launches with no arguments.
    fn draw_url(&mut self, ui: &mut egui::Ui) {
        ui.add_space(8.0);
        let listing = self
            .pending_queue
            .as_ref()
            .map(|pending| format!("listing {}…", pending.label));
        ui.horizontal(|ui| {
            let response = ui.add(
                egui::TextEdit::singleline(&mut self.url)
                    .hint_text("paste a YouTube or yt-dlp URL")
                    .desired_width((ui.available_width() - 150.0).max(80.0)),
            );
            let entered =
                response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
            let ready = !self.url.trim().is_empty() && self.pending_queue.is_none();
            if ui
                .add_enabled(ready, egui::Button::new("Play URL"))
                .clicked()
                || (entered && ready)
            {
                self.submit_url();
            }
            ui.checkbox(&mut self.radio_seed, "radio");
        });
        if let Some(listing) = listing {
            ui.label(egui::RichText::new(listing).weak().size(11.0));
        }
    }

    /// Play/pause, the playhead, and how much of the file the renderer has reached.
    fn draw_playback(&mut self, ui: &mut egui::Ui, prepared_frames: usize) {
        let rate = self.engine.device_rate;
        let controls = Arc::clone(&self.engine.controls);
        let position = controls.position_secs(rate);
        let duration = self.info.audio_secs.max(0.001);
        let is_live = self.live.is_some();
        ui.add_space(10.0);
        ui.horizontal(|ui| {
            let playing = controls.is_playing();
            if ui.button(if playing { "Pause" } else { "Play" }).clicked() {
                controls.set_playing(!playing);
            }
            // A live stream has no length to count against and nothing already rendered, so
            // it shows the playhead alone rather than a fraction of a track it never holds.
            if is_live {
                ui.label(
                    egui::RichText::new(format!(
                        "{:02}:{:02} · live",
                        position as u64 / 60,
                        position as u64 % 60
                    ))
                    .weak()
                    .size(11.0),
                );
                return;
            }
            ui.label(
                egui::RichText::new(format!(
                    "{:02}:{:02} / {:02}:{:02}",
                    position as u64 / 60,
                    position as u64 % 60,
                    duration as u64 / 60,
                    duration as u64 % 60
                ))
                .weak()
                .size(11.0),
            );
            let prepared_secs = prepared_frames as f64 / f64::from(rate);
            ui.label(
                egui::RichText::new(if prepared_frames >= controls.frames {
                    format!("generated: all {:.0} s", self.info.audio_secs)
                } else {
                    format!(
                        "generated: {prepared_secs:.0} s of {:.0} s",
                        self.info.audio_secs
                    )
                })
                .weak()
                .size(11.0),
            );
            let dropouts = controls.underrun_frames();
            if dropouts > 0 {
                ui.label(
                    egui::RichText::new(format!(
                        "dropouts: {:.1} s",
                        dropouts as f64 / f64::from(rate)
                    ))
                    .weak()
                    .size(11.0),
                );
            }
        });
    }

    /// How much of the file the renderer has written, as a bar.
    fn draw_progress(&mut self, ui: &mut egui::Ui) {
        ui.add_space(10.0);
        if let Some(frontier) = &self.frontier {
            let fraction =
                (frontier.ready_frames() as f32 / frontier.frames().max(1) as f32).clamp(0.0, 1.0);
            ui.add(
                egui::ProgressBar::new(fraction)
                    .desired_width(f32::INFINITY)
                    .text(if frontier.is_complete() {
                        "generated".to_string()
                    } else {
                        format!("generating · {:.0}%", fraction * 100.0)
                    }),
            );
        }
    }

    /// What it is playing with: device, cutoff, source, output format.
    fn draw_metadata(&mut self, ui: &mut egui::Ui) {
        let rate = self.engine.device_rate;
        ui.add_space(10.0);
        let source = if self.info.from_cache {
            "cached"
        } else {
            "rendering now"
        };
        let matched = if self.info.gain_db.abs() > 0.05 {
            format!(" · loudness {:+.1} dB", self.info.gain_db)
        } else {
            String::new()
        };
        ui.label(
            egui::RichText::new(format!(
                "{} · cutoff {} Hz · {} · out \"{}\" {} Hz × {} ch{}",
                self.info.device,
                self.info.cutoff_hz,
                source,
                self.engine.device_name,
                rate,
                self.engine.device_channels,
                matched
            ))
            .weak()
            .size(11.0),
        );
    }

    /// The one claim the window must not overstate.
    fn draw_disclaimer(&mut self, ui: &mut egui::Ui) {
        ui.label(
            egui::RichText::new("the top band is generated, not recovered")
                .weak()
                .size(11.0),
        );
    }

    /// The output stream's own failures, when the device stops keeping up.
    fn draw_stream_errors(&mut self, ui: &mut egui::Ui) {
        if let Some(warning) = device::stream_warning(&self.engine.device_name, device::counters())
        {
            ui.label(egui::RichText::new(warning).weak().size(11.0));
        }
    }

    /// The hardware gate: say it rather than let the switch silently never unlock.
    fn draw_cpu_warning(&mut self, ui: &mut egui::Ui) {
        if is_cpu_device(&self.info.device) {
            ui.add_space(6.0);
            ui.colored_label(
                egui::Color32::from_rgb(200, 160, 60),
                "CPU inference: this machine renders slower than realtime, so the A/B switch \
                 may unlock late or not at all. The original plays throughout.",
            );
        }
    }

    /// Why the last load failed, if it did; the previous file is still playing.
    fn draw_error(&mut self, ui: &mut egui::Ui) {
        if let Some(error) = &self.error {
            ui.add_space(8.0);
            ui.colored_label(egui::Color32::from_rgb(220, 90, 90), error);
        }
    }

    /// The live transport's own numbers: the preset that was chosen, what it costs in
    /// latency, and whether the playhead has had to fall back to the dry side.
    fn draw_live(&mut self, ui: &mut egui::Ui) {
        let Some(live) = &self.live else {
            return;
        };
        let stats = live.source.stats();
        ui.add_space(6.0);
        ui.label(
            egui::RichText::new(live_line(
                &live.policy,
                stats.dropouts(),
                stats.measured_rtf(),
            ))
            .weak()
            .size(11.0),
        );
        if let Some(reason) = self.engine.controls.enhanced_blocked() {
            ui.label(
                egui::RichText::new(format!("A/B unavailable: {reason}"))
                    .weak()
                    .size(11.0),
            );
        }
    }

    /// The sink's route and its clock correction: what is captured from what, what it is
    /// played to, what it costs in latency, and what the drift loop has had to do.
    fn draw_sink(&mut self, ui: &mut egui::Ui) {
        let Some(sink) = &self.sink else {
            return;
        };
        let dropouts = self
            .live
            .as_ref()
            .map_or(0, |live| live.source.stats().dropouts());
        ui.add_space(6.0);
        ui.label(
            egui::RichText::new(sink_line(sink, dropouts))
                .weak()
                .size(11.0),
        );
        if sink.stats.capture_state() == CaptureState::Silent {
            // Louder than the rest of the line, because this one means nothing is being
            // captured at all: a working-looking sink with silence in it is the failure mode.
            ui.colored_label(egui::Color32::from_rgb(200, 160, 60), SILENT_HINT);
        }
        ui.label(
            egui::RichText::new(sink_correction_line(sink))
                .weak()
                .size(11.0),
        );
        if let Some(reason) = &sink.fallback {
            ui.label(
                egui::RichText::new(format!("output: {reason}"))
                    .weak()
                    .size(11.0),
            );
        }
    }

    /// The two things the frontier cannot say on its own: that the render stopped, and
    /// that it is not going to catch up. Both otherwise look the same from here — an
    /// empty generated curve and a switch that never lights.
    fn draw_render_status(&self, ui: &mut egui::Ui, prepared_frames: usize) {
        ui.add_space(8.0);
        let Some(status) = &self.status else {
            return;
        };
        if let Some(failure) = status.failure() {
            ui.colored_label(
                egui::Color32::from_rgb(220, 90, 90),
                format!("rendering failed: {failure}"),
            );
            ui.label(
                egui::RichText::new(
                    "the original plays on; the A/B switch needs generated audio, so it \
                     stays locked",
                )
                .weak()
                .size(11.0),
            );
            return;
        }
        if status.is_done() {
            return;
        }
        let rate = self.engine.device_rate as usize;
        let position = self.engine.controls.position_frames();
        // A second of slack: the frontier runs a window ahead by design, and a hair
        // behind is not the news this line exists to deliver.
        let behind = position
            .saturating_sub(prepared_frames)
            .saturating_sub(rate);
        if behind > 0 {
            ui.label(
                egui::RichText::new(format!(
                    "the renderer is {:.0} s behind the playhead · the A/B switch unlocks \
                     once it catches up · this machine renders slower than realtime",
                    behind as f64 / rate as f64
                ))
                .weak()
                .size(11.0),
            );
        }
    }
}

impl PlayerApp {
    /// Start once there is something worth hearing.
    ///
    /// The normal case is a frontier that reaches the prebuffer and plays already enhanced.
    /// The patience budget exists for a machine that will never get there, and it is only a
    /// fallback for being **slow**, not for being **not ready**: a cold model load measured
    /// 22.8 s on a CUDA box (16.9 s of weights + 2.5 s of warmup), so a 30 s wall-clock
    /// timer sat right on the boundary and sometimes started on the original only because
    /// the weights were still loading. Starting without the generated band was never the
    /// intent, so while the model is still loading the window waits however long it takes;
    /// the budget applies only once it has loaded and the renderer is genuinely behind.
    fn maybe_start(&mut self) {
        if self.started {
            return;
        }
        let ready = self.ready_frames();
        // A frontier is only present when this track has to render, and the model loads
        // lazily on that path, so "no device yet" and "still loading" are the same state.
        let model_loading = self.frontier.is_some() && !self.session.worker().is_loaded();
        if should_start(
            ready,
            self.prebuffer_frames,
            self.frontier.is_some(),
            model_loading,
            self.opened_at.elapsed().as_secs_f64(),
        ) {
            self.started = true;
            self.engine.controls.set_playing(true);
        }
    }
}

/// Whether playback may begin, given what exists and whether the model is still loading.
///
/// Split out from [`PlayerApp::maybe_start`] so the rule `AGENTS.md` asks about — the one
/// that started on the original because the weights were still loading — is testable
/// without a device, a window or a model.
fn should_start(
    ready_frames: usize,
    prebuffer_frames: usize,
    has_frontier: bool,
    model_loading: bool,
    elapsed_secs: f64,
) -> bool {
    // Nothing to generate (a full-band file, a cache hit) starts at once.
    !has_frontier
        // The product: the generated band exists for the first note.
        || ready_frames >= prebuffer_frames
        // Only a renderer that is loaded and still behind is "slow". Loading is not slow;
        // it is not ready, and waiting out a 24 s weight load is what was promised.
        || (elapsed_secs > START_BUDGET_SECS && !model_loading)
}

impl PlayerApp {
    /// The whole track as per-bar RMS, with the render frontier marked: bars the renderer
    /// has reached are bright, the rest are dim. Click or drag anywhere to seek; this is
    /// the transport, not a picture.
    fn draw_waveform(&self, ui: &mut egui::Ui) {
        ui.add_space(6.0);
        let height = 64.0;
        let (response, painter) = ui.allocate_painter(
            egui::vec2(ui.available_width(), height),
            egui::Sense::click_and_drag(),
        );
        let rect = response.rect;
        painter.rect_filled(rect, 3.0, egui::Color32::from_rgb(17, 19, 23));

        let controls = &self.engine.controls;
        let frames = controls.frames.max(1);
        let prepared = self.ready_frames().min(frames);
        let position = controls.position_frames().min(frames);
        let bars = self.waveform.len();
        if bars > 0 {
            let width = rect.width().max(1.0);
            let bar_width = width / bars as f32;
            let mid = rect.center().y;
            let half = (rect.height() * 0.5 - 4.0).max(1.0);
            let accent = egui::Color32::from_rgb(64, 200, 180);
            let dim = egui::Color32::from_rgb(64, 70, 78);
            for (index, rms) in self.waveform.iter().enumerate() {
                // A full-band source has nothing to generate, so there is no frontier to
                // mark and one colour is the honest picture.
                let colour =
                    if !controls.has_enhanced() || index * self.waveform_bar_frames < prepared {
                        accent
                    } else {
                        dim
                    };
                // sqrt so quiet passages stay legible without flattening the peaks.
                let reach = (rms.sqrt() * half).max(0.5);
                let x = rect.left() + bar_width * index as f32;
                let w = (bar_width - 1.0).max(1.0);
                painter.rect_filled(
                    egui::Rect::from_min_max(
                        egui::pos2(x, mid - reach),
                        egui::pos2(x + w, mid + reach),
                    ),
                    0.0,
                    colour,
                );
            }
        }

        let x_of = |frame: usize| {
            rect.left() + rect.width() * (frame as f32 / frames as f32).clamp(0.0, 1.0)
        };
        if self.frontier.is_some() && prepared < frames {
            let x = x_of(prepared);
            painter.line_segment(
                [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(64, 200, 180)),
            );
        }
        let x = x_of(position);
        painter.line_segment(
            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
            egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(230, 232, 236)),
        );

        if response.clicked() || response.dragged() {
            if let Some(pointer) = response.interact_pointer_pos() {
                let rate = self.engine.device_rate.max(1);
                let fraction = ((pointer.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                let seconds = f64::from(fraction) * frames as f64 / f64::from(rate);
                controls.seek_secs(seconds, rate);
            }
        }
    }

    /// The two spectra at the playhead. This is the product's explanation: below the
    /// cutoff the curves sit on top of each other, above it they part company.
    fn draw_spectrum(&self, ui: &mut egui::Ui) {
        ui.add_space(10.0);
        let height = 108.0;
        let (response, painter) = ui.allocate_painter(
            egui::vec2(ui.available_width(), height),
            egui::Sense::hover(),
        );
        let rect = response.rect;
        painter.rect_filled(rect, 3.0, egui::Color32::from_rgb(17, 19, 23));

        let position = self.engine.controls.position_frames();
        let Some((original, generated)) = self.spectra(position) else {
            return;
        };

        let top = 0.0f32;
        let bottom = -90.0f32;
        let sample_rate = self.engine.device_rate as f32;
        let nyquist = sample_rate / 2.0;
        let width = rect.width().max(1.0);
        let to_x = |hz: f32| {
            let low = 40.0f32.min(nyquist * 0.01);
            let span = (nyquist / low).ln();
            let t = (hz.max(low) / low).ln() / span;
            rect.left() + t * width
        };
        let to_y = |db: f32| {
            let t = ((db.clamp(bottom, top) - bottom) / (top - bottom)).clamp(0.0, 1.0);
            rect.bottom() - t * rect.height()
        };

        // Grid: every 12 dB, and the frequencies a person can name.
        let grid = egui::Color32::from_rgb(38, 42, 48);
        for db in [-60.0f32, -30.0] {
            painter.line_segment(
                [
                    egui::pos2(rect.left(), to_y(db)),
                    egui::pos2(rect.right(), to_y(db)),
                ],
                egui::Stroke::new(1.0_f32, grid),
            );
        }
        for hz in [1_000.0f32, 4_000.0, 8_000.0, 16_000.0] {
            if hz < nyquist {
                painter.line_segment(
                    [
                        egui::pos2(to_x(hz), rect.top()),
                        egui::pos2(to_x(hz), rect.bottom()),
                    ],
                    egui::Stroke::new(1.0_f32, grid),
                );
            }
        }

        // One point per pixel column: the loudest bin it covers.
        let curve = |spectrum: &[f32], painter: &egui::Painter| -> Vec<egui::Pos2> {
            let bins = spectrum.len();
            let mut points = Vec::with_capacity(width as usize + 2);
            for pixel in 0..width as usize {
                let f0 = frequency_at(pixel as f32 / width, sample_rate);
                let f1 = frequency_at((pixel + 1) as f32 / width, sample_rate);
                let b0 = (f0 / nyquist * (bins - 1) as f32).floor().max(0.0) as usize;
                let b1 = (f1 / nyquist * (bins - 1) as f32).ceil() as usize;
                let peak = spectrum[b0.min(bins - 1)..b1.clamp(b0 + 1, bins)]
                    .iter()
                    .cloned()
                    .fold(bottom, f32::max);
                points.push(egui::pos2(rect.left() + pixel as f32, to_y(peak)));
            }
            painter.add(egui::Shape::line(
                points.clone(),
                egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(120, 126, 135)),
            ));
            points
        };

        let original_points = curve(&original, &painter);
        if let Some(generated) = &generated {
            let generated_points = curve(generated, &painter);
            // Draw the generated curve over the original, and shade between them.
            let generated_colour = egui::Color32::from_rgb(64, 200, 180);
            painter.add(egui::Shape::line(
                generated_points.clone(),
                egui::Stroke::new(1.5_f32, generated_colour),
            ));
            let _ = original_points;
            painter.add(egui::Shape::Path(egui::epaint::PathShape {
                points: generated_points,
                closed: false,
                fill: egui::Color32::TRANSPARENT,
                stroke: egui::Stroke::NONE.into(),
            }));
        }

        // The cutoff, which is where the two curves stop agreeing.
        let cutoff_x = to_x(self.info.cutoff_hz as f32);
        painter.line_segment(
            [
                egui::pos2(cutoff_x, rect.top()),
                egui::pos2(cutoff_x, rect.bottom()),
            ],
            egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(200, 160, 60)),
        );
        painter.text(
            egui::pos2(cutoff_x + 4.0, rect.top() + 3.0),
            egui::Align2::LEFT_TOP,
            format!("{:.1} kHz", self.info.cutoff_hz as f32 / 1000.0),
            egui::FontId::proportional(9.0),
            egui::Color32::from_rgb(200, 170, 90),
        );
        let legend = if generated.is_some() {
            "original · generated"
        } else {
            "original (generating…)"
        };
        painter.text(
            egui::pos2(rect.right() - 4.0, rect.top() + 3.0),
            egui::Align2::RIGHT_TOP,
            legend,
            egui::FontId::proportional(9.0),
            egui::Color32::from_rgb(120, 126, 135),
        );
    }

    /// Both curves at `position`, from whichever source this playback holds: the
    /// whole-track buffers, the render frontier, or the one resident segment.
    fn spectra(&self, position: usize) -> Option<(Vec<f32>, Option<Vec<f32>>)> {
        let channels = self.engine.device_channels;
        let has_whole_track = !self.engine.display_original.is_empty();
        if has_whole_track {
            let original = spectrum::analyse(&self.engine.display_original, channels, position)?;
            let generated = match &self.engine.display_generated {
                Some(buffer) => spectrum::analyse(buffer, channels, position),
                None => self.frontier.as_ref().and_then(|frontier| {
                    window_from_frontier(frontier, position)
                        .and_then(|window| spectrum::analyse(&window, 2, spectrum::WINDOW / 2))
                }),
            };
            return Some((original, generated));
        }
        // No whole-track buffer: a segmented playback reads the segment at the playhead,
        // and one resident segment answers both sides of the window.
        let segment = self.segments.as_ref()?.reader.segment_at(position);
        let segment = segment.as_ref()?;
        segment_spectra(segment, position)
    }
}

/// A worker on the CPU is a renderer, not a player; the window says so rather than letting
/// the switch silently never unlock.
fn is_cpu_device(device: &str) -> bool {
    device == "cpu"
}

/// The live line: the preset, what the latency is made of, and what has gone wrong so far.
///
/// The policy's total is the number D3 says to print: the preset alone is the model window,
/// not the latency a listener hears.
fn live_line(
    policy: &flacify_core::LivePolicy,
    dropouts: usize,
    measured_rtf: Option<f64>,
) -> String {
    let measured = match measured_rtf {
        Some(rtf) => format!(" · RTF {rtf:.2}"),
        None => String::new(),
    };
    let unit = match policy.preset.granules() {
        1 => "granule",
        _ => "granules",
    };
    format!(
        "live · preset {} {unit} · latency {:.2} s = window {:.2} + buffer {:.2} + device \
         {:.2} · dropouts {dropouts}{measured}",
        policy.preset.granules(),
        policy.total_latency_secs,
        policy.model_latency_secs,
        policy.buffer_latency_secs,
        policy.device_latency_secs,
    )
}

/// The sink's line: the route, the latency it costs, and what has gone wrong so far.
///
/// The latency includes the capture ring's target fill, which the panel would otherwise
/// leave out: on the sink that ring is delay the listener hears. The capture state comes
/// last because it decides whether any other number on the line means anything.
fn sink_line(route: &SinkRoute, dropouts: usize) -> String {
    let fallback = match route.fallback {
        Some(_) => " (default)",
        None => "",
    };
    let state = match route.stats.capture_state() {
        CaptureState::Signal => "capture signal",
        CaptureState::Silent => "capture silent",
    };
    format!(
        "sink · {} → {}{fallback} · latency {:.2} s · dropouts {dropouts} · resyncs {} · {state}",
        route.from,
        route.to,
        route.latency_secs,
        route.stats.resyncs()
    )
}

/// The drift loop's own line: the correction it is applying, and how full the ring is.
fn sink_correction_line(route: &SinkRoute) -> String {
    format!(
        "drift {:+.0} ppm (peak {:.0}) · ring fill {} of {} frames · {} dropped",
        route.stats.correction_ppm(),
        route.stats.peak_correction_ppm(),
        route.stats.fill_frames(),
        route.ring_capacity_frames,
        route.stats.dropped_frames()
    )
}

/// Frequency at a fraction of the way across a log axis from 40 Hz to Nyquist.
fn frequency_at(fraction: f32, sample_rate: f32) -> f32 {
    let nyquist = sample_rate / 2.0;
    let low = 40.0f32.min(nyquist * 0.01);
    low * (nyquist / low).powf(fraction.clamp(0.0, 1.0))
}

/// Per-bar RMS of an interleaved buffer, normalized to its loudest bar.
///
/// Bars are at least one second long, and there are never more bars than `max_bars`, so a
/// very long track loses per-second detail rather than drawing sub-pixel bars. Returns the
/// bars and the number of frames each one covers.
fn waveform_bars(
    interleaved: &[f32],
    channels: usize,
    sample_rate: u32,
    max_bars: usize,
) -> (Vec<f32>, usize) {
    let channels = channels.max(1);
    let rate = sample_rate.max(1) as usize;
    let frames = interleaved.len() / channels;
    if frames == 0 || max_bars == 0 {
        return (Vec::new(), rate);
    }
    let bar_frames = rate.max(frames.div_ceil(max_bars)).max(1);
    let bars = frames.div_ceil(bar_frames);
    let mut out = Vec::with_capacity(bars);
    for bar in 0..bars {
        let start = bar * bar_frames;
        let end = ((bar + 1) * bar_frames).min(frames);
        let mut sum = 0.0f64;
        for frame in start..end {
            let base = frame * channels;
            for channel in 0..channels {
                let sample = f64::from(interleaved[base + channel]);
                sum += sample * sample;
            }
        }
        let count = (end - start) * channels;
        out.push((sum / count.max(1) as f64).sqrt() as f32);
    }
    normalize(&mut out);
    (out, bar_frames)
}

/// Scale the bars so the loudest is 1.0; silence stays silence rather than being amplified.
fn normalize(bars: &mut [f32]) {
    let peak = bars.iter().copied().fold(0.0f32, f32::max);
    if peak > 1e-6 {
        for value in bars.iter_mut() {
            *value = (*value / peak).clamp(0.0, 1.0);
        }
    }
}

/// The waveform for one playback: the source tier's peaks when the track plays as
/// segments, the whole-track buffer otherwise.
fn waveform_for(
    frames: usize,
    channels: usize,
    sample_rate: u32,
    original: &[f32],
    peaks: &[SecondPeaks],
) -> (Vec<f32>, usize) {
    if peaks.is_empty() {
        return waveform_bars(original, channels, sample_rate, WAVEFORM_MAX_BARS);
    }
    waveform_bars_from_peaks(peaks, frames, sample_rate, WAVEFORM_MAX_BARS)
}

/// The peaks a segmented playback carries; empty on the whole-track paths.
fn peaks_of(segments: Option<&SegmentView>) -> &[SecondPeaks] {
    match segments {
        Some(segments) => &segments.peaks,
        None => &[],
    }
}

/// Per-bar RMS from the source tier's one-second peaks, normalized to its loudest bar.
///
/// The segmented path's replacement for [`waveform_bars`]: two numbers per second
/// instead of the whole track, on the same bar grid, with each bar weighting the seconds
/// it covers. The tier measures one peak per [`flacify_core::SAMPLE_RATE`] frames.
fn waveform_bars_from_peaks(
    peaks: &[SecondPeaks],
    frames: usize,
    sample_rate: u32,
    max_bars: usize,
) -> (Vec<f32>, usize) {
    let rate = sample_rate.max(1) as usize;
    if frames == 0 || max_bars == 0 || peaks.is_empty() {
        return (Vec::new(), rate);
    }
    let peak_frames = flacify_core::SAMPLE_RATE as usize;
    let bar_frames = rate.max(frames.div_ceil(max_bars)).max(1);
    let bars = frames.div_ceil(bar_frames);
    let mut out = Vec::with_capacity(bars);
    for bar in 0..bars {
        let start = bar * bar_frames;
        let end = ((bar + 1) * bar_frames).min(frames);
        let first = start / peak_frames;
        let last = ((end - 1) / peak_frames).min(peaks.len() - 1);
        let mut energy = 0.0f64;
        let mut covered = 0.0f64;
        for (index, peak) in peaks.iter().enumerate().take(last + 1).skip(first) {
            let second_start = index * peak_frames;
            let second_end = second_start.saturating_add(peak_frames).min(frames);
            let overlap = end.min(second_end).saturating_sub(start.max(second_start)) as f64;
            if overlap <= 0.0 {
                continue;
            }
            let left = f64::from(peak.left_rms);
            let right = f64::from(peak.right_rms);
            energy += overlap * (left * left + right * right) / 2.0;
            covered += overlap;
        }
        out.push(match covered > 0.0 {
            true => (energy / covered).sqrt() as f32,
            false => 0.0,
        });
    }
    normalize(&mut out);
    (out, bar_frames)
}

/// `spectrum::WINDOW` frames of interleaved stereo around `centre`, if the frontier has
/// rendered all of them.
fn window_from_frontier(frontier: &Frontier, centre: usize) -> Option<Vec<f32>> {
    let start = centre.saturating_sub(spectrum::WINDOW / 2);
    let end = start + spectrum::WINDOW;
    if end > frontier.ready_frames() {
        return None;
    }
    let mut window = Vec::with_capacity(spectrum::WINDOW * 2);
    for frame in start..end {
        let (left, right) = frontier.frame(frame)?;
        window.push(left);
        window.push(right);
    }
    Some(window)
}

/// Both curves from the one resident segment at `position`: the source's own audio and,
/// when the segment has it, the rendered side. `None` for a segment shorter than a window,
/// which is only the last one of a track.
fn segment_spectra(segment: &Resident, position: usize) -> Option<(Vec<f32>, Option<Vec<f32>>)> {
    let start = segment.start();
    let frames = segment.frames();
    let window = window_from_segment(start, frames, position, |offset, channel| {
        segment.dry(offset, channel)
    })?;
    let original = spectrum::analyse(&window, 2, spectrum::WINDOW / 2)?;
    let generated = match segment.has_wet() {
        false => None,
        true => window_from_segment(start, frames, position, |offset, channel| {
            segment.wet(offset, channel)
        })
        .and_then(|window| spectrum::analyse(&window, 2, spectrum::WINDOW / 2)),
    };
    Some((original, generated))
}

/// `spectrum::WINDOW` frames of interleaved stereo around `centre`, clamped inside the
/// segment at `start`, so one resident segment answers the whole window.
fn window_from_segment(
    start: usize,
    frames: usize,
    centre: usize,
    sample: impl Fn(usize, usize) -> f32,
) -> Option<Vec<f32>> {
    if frames < spectrum::WINDOW {
        return None;
    }
    let offset = centre
        .saturating_sub(start)
        .saturating_sub(spectrum::WINDOW / 2)
        .min(frames - spectrum::WINDOW);
    let mut window = Vec::with_capacity(spectrum::WINDOW * 2);
    for frame in offset..offset + spectrum::WINDOW {
        for channel in 0..2 {
            window.push(sample(frame, channel));
        }
    }
    Some(window)
}

impl eframe::App for PlayerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.title_dirty {
            self.title_dirty = false;
            let name = self
                .path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| "flacify".to_string());
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(format!("flacify — {name}")));
        }

        // The window takes another file, or several, without restarting the app, which is
        // what lets the resident worker pay for itself: the next track skips the weight load.
        let dropped: Vec<PathBuf> = ctx.input(|input| {
            input
                .raw
                .dropped_files
                .iter()
                .filter_map(|file| file.path.clone())
                .collect()
        });
        if !dropped.is_empty() {
            self.open_paths(&dropped);
        }

        self.poll_fetch();
        self.poll_queue();
        self.auto_advance();
        self.maybe_start();

        let mut next = false;
        let mut previous = false;
        // A focused text field owns the keyboard: `b` in `youtu.be` is a letter, not a
        // track change, and space in a paste must not toggle the A/B switch.
        if !ctx.wants_keyboard_input() {
            let engine = &self.engine;
            let controls = &engine.controls;
            ctx.input(|input| {
                if input.key_pressed(egui::Key::Space) {
                    controls.toggle();
                }
                if input.key_pressed(egui::Key::P) {
                    controls.set_playing(!controls.is_playing());
                }
                let rate = engine.device_rate as usize;
                if input.key_pressed(egui::Key::ArrowLeft) {
                    let back = controls.position_frames().saturating_sub(5 * rate);
                    controls.seek_frames(back);
                }
                if input.key_pressed(egui::Key::ArrowRight) {
                    controls.seek_frames(controls.position_frames() + 5 * rate);
                }
                if input.key_pressed(egui::Key::Escape) {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
                next = input.key_pressed(egui::Key::N);
                previous = input.key_pressed(egui::Key::B);
            });
        }
        let playing = self.engine.controls.is_playing();
        if playing || self.pending_queue.is_some() || self.pending.is_some() {
            ctx.request_repaint_after(Duration::from_millis(33));
        } else {
            ctx.request_repaint_after(Duration::from_millis(250));
        }
        if next {
            self.next_track();
        }
        if previous {
            self.previous_track();
        }

        egui::CentralPanel::default().show(ctx, |ui| self.draw_transport(ui));
    }
}

impl Drop for PlayerApp {
    /// Closing the window stops this file's renderer; the session then drops the
    /// worker, and `Connection::drop` kills the child.
    fn drop(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            cancel.store(true, Ordering::Relaxed);
        }
        if let Some(pending) = self.pending.take() {
            pending.cancel.cancel();
        }
    }
}

pub fn run(app: impl eframe::App + 'static) -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([560.0, 560.0])
            .with_resizable(false)
            .with_title("flacify"),
        ..Default::default()
    };
    eframe::run_native("flacify", options, Box::new(|_cc| Ok(Box::new(app))))
}

/// The window before anything is playing: the app launched with no argument.
enum EmptyState {
    Waiting,
    Player(PlayerApp),
    Setup(first_run::FirstRun),
}

struct EmptyApp {
    paths: setup::Paths,
    splice: bool,
    state: EmptyState,
    error: Option<String>,
    /// The URL box, the same one the player carries, for the window that launched empty.
    url: String,
    radio_seed: bool,
    /// A pasted URL being listed and fetched, so the window keeps drawing.
    pending: Option<PendingOpen>,
}

impl EmptyApp {
    fn new(paths: setup::Paths, splice: bool) -> Self {
        Self {
            paths,
            splice,
            state: EmptyState::Waiting,
            error: None,
            url: String::new(),
            radio_seed: false,
            pending: None,
        }
    }

    /// Build a queue from dropped paths and start it. These are filesystem paths, so no
    /// network is touched here and the UI thread does not block on a download.
    fn open(&mut self, dropped: &[PathBuf]) {
        let downloads = self.paths.downloads_dir();
        let queue = match playlist::Queue::from_local(dropped, &downloads) {
            Ok(queue) => queue,
            Err(error) => {
                self.error = Some(format!("{error:#}"));
                return;
            }
        };
        let Some(entry) = queue.current() else {
            self.error = Some("the queue is empty".to_string());
            return;
        };
        let resolved = match playlist::resolve(entry, &downloads, None) {
            Ok(resolved) => resolved,
            Err(error) => {
                self.error = Some(format!("{error:#}"));
                return;
            }
        };
        self.open_queue(queue, resolved);
    }

    /// A pasted URL is listed and its first track fetched on a thread; the window that has
    /// nothing playing is the one place a blocking yt-dlp call is least forgivable.
    fn submit_url(&mut self) {
        let text = self.url.trim().to_string();
        if text.is_empty() || self.pending.is_some() {
            return;
        }
        if youtube::parse(&text).is_none() {
            self.error = Some(format!("{text} is not a URL"));
            return;
        }
        let downloads = self.paths.downloads_dir();
        let radio = radio_requested(&text, self.radio_seed);
        let label = text.clone();
        let handle = std::thread::spawn(move || {
            let queue = playlist::Queue::from_url(&text, &downloads, radio, false)?;
            let resolved = match queue.current() {
                Some(entry) => playlist::resolve(entry, &downloads, None)?,
                None => anyhow::bail!("the queue is empty"),
            };
            Ok((queue, resolved))
        });
        self.pending = Some(PendingOpen { label, handle });
        self.error = None;
    }

    /// Collect a finished listing, once.
    fn poll_open(&mut self) {
        let Some(pending) = self.pending.as_ref() else {
            return;
        };
        if !pending.handle.is_finished() {
            return;
        }
        let pending = self.pending.take().expect("just checked");
        match pending.handle.join() {
            Ok(Ok((queue, resolved))) => {
                self.url.clear();
                self.open_queue(queue, resolved);
            }
            Ok(Err(error)) => self.error = Some(format!("{error:#}")),
            Err(_) => self.error = Some(format!("listing {} failed", pending.label)),
        }
    }

    /// Open the queue's first track and hand the window to the player, or to setup when the
    /// model is what the track needs.
    fn open_queue(&mut self, queue: playlist::Queue, resolved: playlist::Resolved) {
        let session = match Session::new(&self.paths, self.splice) {
            Ok(session) => session,
            Err(error) => {
                self.error = Some(format!("{error:#}"));
                return;
            }
        };
        match session.open(&resolved.path) {
            Ok(playback) => {
                self.state = EmptyState::Player(PlayerApp::with_queue(session, playback, queue));
                self.error = None;
            }
            Err(error) if needs_setup(&error) => {
                self.state = EmptyState::Setup(first_run::FirstRun::with_queue(
                    self.paths.clone(),
                    queue,
                    self.splice,
                ));
                self.error = None;
            }
            Err(error) => {
                self.error = Some(format!(
                    "could not play {}: {error:#}",
                    resolved.path.display()
                ));
            }
        }
    }
}

impl eframe::App for EmptyApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        match &mut self.state {
            EmptyState::Player(player) => {
                player.update(ctx, frame);
                return;
            }
            EmptyState::Setup(setup) => {
                setup.update(ctx, frame);
                return;
            }
            EmptyState::Waiting => {}
        }

        self.poll_open();

        let dropped: Vec<PathBuf> = ctx.input(|input| {
            input
                .raw
                .dropped_files
                .iter()
                .filter_map(|file| file.path.clone())
                .collect()
        });
        if !dropped.is_empty() {
            self.open(&dropped);
            ctx.request_repaint();
            return;
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(12.0);
            ui.label(egui::RichText::new("flacify").size(20.0).strong());
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new("drop an audio file, a folder, or an .m3u/.pls playlist")
                    .weak(),
            );
            ui.add_space(10.0);
            let listing = self
                .pending
                .as_ref()
                .map(|pending| format!("listing and fetching {}…", pending.label));
            let busy = self.pending.is_some();
            ui.horizontal(|ui| {
                let response = ui.add_enabled(
                    !busy,
                    egui::TextEdit::singleline(&mut self.url)
                        .hint_text("or paste a YouTube or yt-dlp URL")
                        .desired_width((ui.available_width() - 150.0).max(80.0)),
                );
                let entered =
                    response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
                let ready = !busy && !self.url.trim().is_empty();
                if ui
                    .add_enabled(ready, egui::Button::new("Play URL"))
                    .clicked()
                    || (entered && ready)
                {
                    self.submit_url();
                }
                ui.add_enabled(!busy, egui::Checkbox::new(&mut self.radio_seed, "radio"));
            });
            if let Some(listing) = listing {
                ui.add_space(4.0);
                ui.label(egui::RichText::new(listing).weak().size(11.0));
            }
            if let Some(error) = &self.error {
                ui.add_space(6.0);
                ui.colored_label(egui::Color32::from_rgb(220, 90, 90), error);
            }
        });
        if self.pending.is_some() {
            ctx.request_repaint_after(Duration::from_millis(120));
        }
    }
}

/// Run the window with nothing loaded; `flacify` with no argument uses this.
pub fn run_empty(paths: setup::Paths, splice: bool) -> eframe::Result<()> {
    run(EmptyApp::new(paths, splice))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn given_quiet_then_loud_audio_when_analysed_then_the_bars_follow_it() {
        // Four seconds of stereo: -20 dBFS for two seconds, then -6 dBFS.
        let mut buffer = Vec::with_capacity(48_000 * 4 * 2);
        for frame in 0..48_000 * 4 {
            let amplitude = if frame < 48_000 * 2 { 0.1 } else { 0.5 };
            buffer.push(amplitude);
            buffer.push(amplitude);
        }
        let (bars, bar_frames) = waveform_bars(&buffer, 2, 48_000, WAVEFORM_MAX_BARS);
        assert_eq!(bar_frames, 48_000, "one bar per second");
        assert_eq!(bars.len(), 4);
        assert!((bars[0] - 0.2).abs() < 1e-3, "{}", bars[0]);
        assert!((bars[3] - 1.0).abs() < 1e-3, "{}", bars[3]);
        assert_eq!(bars[0], bars[1]);
        assert_eq!(bars[2], bars[3]);
    }

    #[test]
    fn given_more_seconds_than_bars_when_analysed_then_bars_do_not_exceed_the_cap() {
        // Ten minutes of silence: 600 one-second bars is more than the width can show,
        // so each bar has to cover several seconds instead of overlapping its neighbour.
        let frames = 600 * 48_000;
        let buffer = vec![0.0f32; frames * 2];
        let (bars, bar_frames) = waveform_bars(&buffer, 2, 48_000, 100);
        assert_eq!(bars.len(), 100);
        assert!(bar_frames >= 6 * 48_000, "{bar_frames}");
    }

    #[test]
    fn given_silence_when_analysed_then_the_peak_is_not_amplified_into_noise() {
        let buffer = vec![0.0f32; 48_000 * 2];
        let (bars, _) = waveform_bars(&buffer, 2, 48_000, WAVEFORM_MAX_BARS);
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0], 0.0);
    }

    #[test]
    fn given_no_frames_when_analysed_then_there_are_no_bars() {
        let (bars, bar_frames) = waveform_bars(&[], 2, 44_100, WAVEFORM_MAX_BARS);
        assert!(bars.is_empty());
        assert_eq!(bar_frames, 44_100);
    }

    /// Planar stereo whose amplitude changes once per second, the layout the source tier
    /// measures its peaks from.
    fn stepped_planar(amplitudes: &[f32], frames: usize) -> Vec<f32> {
        let mut planar = Vec::with_capacity(frames * 2);
        for _ in 0..2 {
            for frame in 0..frames {
                planar.push(amplitudes[(frame / 48_000).min(amplitudes.len() - 1)]);
            }
        }
        planar
    }

    /// The interleaved form of [`stepped_planar`], which is what the mixer and the
    /// whole-buffer waveform read.
    fn stepped_interleaved(planar: &[f32], frames: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(frames * 2);
        for frame in 0..frames {
            out.push(planar[frame]);
            out.push(planar[frames + frame]);
        }
        out
    }

    #[test]
    fn given_peaks_measured_from_a_stepped_track_when_the_waveform_is_built_then_it_matches_the_whole_buffer(
    ) {
        let frames = 48_000 * 4;
        let planar = stepped_planar(&[0.1, 0.1, 0.5, 0.5], frames);
        let buffer = stepped_interleaved(&planar, frames);
        let peaks = crate::source::second_peaks(&planar, frames);
        let (from_peaks, peak_bar_frames) =
            waveform_bars_from_peaks(&peaks, frames, 48_000, WAVEFORM_MAX_BARS);
        let (from_samples, sample_bar_frames) =
            waveform_bars(&buffer, 2, 48_000, WAVEFORM_MAX_BARS);
        assert_eq!(peak_bar_frames, sample_bar_frames);
        assert_eq!(from_peaks.len(), 4, "one bar per second of the track");
        assert_eq!(from_peaks.len(), from_samples.len());
        for (peak, sample) in from_peaks.iter().zip(&from_samples) {
            assert!((peak - sample).abs() < 1e-6, "{peak} against {sample}");
        }
    }

    #[test]
    fn given_bars_that_do_not_end_on_a_second_when_built_from_peaks_then_they_match_the_sample_bars(
    ) {
        // 10.5 seconds and seven bars: each bar is 72 000 frames, so most of them straddle
        // a second boundary and the weighting is what makes them agree.
        let frames = 48_000 * 10 + 24_000;
        let planar = stepped_planar(
            &[0.2, 0.4, 0.1, 0.8, 0.05, 0.6, 0.3, 0.9, 0.15, 0.7, 0.5],
            frames,
        );
        let buffer = stepped_interleaved(&planar, frames);
        let peaks = crate::source::second_peaks(&planar, frames);
        let (from_peaks, peak_bar_frames) = waveform_bars_from_peaks(&peaks, frames, 48_000, 7);
        let (from_samples, sample_bar_frames) = waveform_bars(&buffer, 2, 48_000, 7);
        assert_eq!(peak_bar_frames, 72_000);
        assert_eq!(sample_bar_frames, peak_bar_frames);
        assert_eq!(from_peaks.len(), 7);
        assert_eq!(from_peaks.len(), from_samples.len());
        for (peak, sample) in from_peaks.iter().zip(&from_samples) {
            assert!((peak - sample).abs() < 1e-5, "{peak} against {sample}");
        }
    }

    #[test]
    fn given_a_segmented_playback_when_the_waveform_is_built_then_it_reads_the_peaks_not_the_buffer(
    ) {
        let frames = 48_000 * 2;
        let planar = stepped_planar(&[0.25, 0.5], frames);
        let peaks = crate::source::second_peaks(&planar, frames);
        // No whole-track buffer: this is the state a cache hit is in.
        let (bars, bar_frames) = waveform_for(frames, 2, 48_000, &[], &peaks);
        assert_eq!(bar_frames, 48_000);
        assert_eq!(bars.len(), 2);
        assert!((bars[0] - 0.5).abs() < 1e-6, "{}", bars[0]);
        assert!((bars[1] - 1.0).abs() < 1e-6, "{}", bars[1]);
    }

    #[test]
    fn given_no_peaks_when_the_waveform_is_built_then_it_reads_the_whole_buffer() {
        let frames = 48_000;
        let planar = stepped_planar(&[0.5], frames);
        let buffer = stepped_interleaved(&planar, frames);
        let (bars, bar_frames) = waveform_for(frames, 2, 48_000, &buffer, &[]);
        assert_eq!(bar_frames, 48_000);
        assert_eq!(bars.len(), 1);
        assert!((bars[0] - 1.0).abs() < 1e-6, "{}", bars[0]);
    }

    #[test]
    fn given_no_peaks_when_the_peaks_waveform_is_built_then_there_are_no_bars() {
        let (bars, bar_frames) = waveform_bars_from_peaks(&[], 48_000, 44_100, WAVEFORM_MAX_BARS);
        assert!(bars.is_empty());
        assert_eq!(bar_frames, 44_100);
    }

    #[test]
    fn given_an_hour_of_peaks_when_the_waveform_is_built_then_it_allocates_the_bars_not_the_track()
    {
        let peaks = vec![
            SecondPeaks {
                left_rms: 0.25,
                right_rms: 0.25,
            };
            3_600
        ];
        let frames = 3_600 * 48_000;
        let ((bars, _bar_frames), allocated) = crate::alloc_counter::measure(|| {
            waveform_bars_from_peaks(&peaks, frames, 48_000, WAVEFORM_MAX_BARS)
        });
        let whole_track = frames * 2 * std::mem::size_of::<f32>();
        println!("waveform from an hour of peaks: {allocated} bytes, track is {whole_track} bytes");
        assert_eq!(bars.len(), WAVEFORM_MAX_BARS);
        assert!(
            allocated <= WAVEFORM_MAX_BARS * std::mem::size_of::<f32>(),
            "{allocated} bytes for {} bars",
            bars.len()
        );
        assert!(allocated * 10_000 < whole_track);
    }

    /// Planar stereo holding `hz` in both planes, the layout a resident segment has.
    fn tone_planes(hz: f32, frames: usize, amplitude: f32) -> Vec<f32> {
        let mut planar = vec![0.0f32; frames * 2];
        for frame in 0..frames {
            let value = amplitude * (std::f32::consts::TAU * hz * frame as f32 / 48_000.0).sin();
            planar[frame] = value;
            planar[frames + frame] = value;
        }
        planar
    }

    /// The frequency of a spectrum's loudest bin.
    fn peak_hz(spectrum: &[f32]) -> f32 {
        let peak = spectrum
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("a finite bin"))
            .map(|(index, _)| index)
            .expect("at least one bin");
        peak as f32 * 48_000.0 / spectrum::WINDOW as f32
    }

    #[test]
    fn given_a_resident_segment_with_a_1khz_tone_when_the_spectrum_is_taken_then_the_curve_peaks_at_1khz(
    ) {
        let frames = 48_000;
        let segment = Resident::new(
            0,
            0,
            frames,
            Arc::from(tone_planes(1_000.0, frames, 0.5)),
            None,
        );
        let (original, generated) = segment_spectra(&segment, 24_000).expect("a spectrum");
        assert!(
            (peak_hz(&original) - 1_000.0).abs() < 25.0,
            "peak at {} Hz",
            peak_hz(&original)
        );
        assert!(
            generated.is_none(),
            "a segment with no rendered side has no second curve"
        );
    }

    #[test]
    fn given_a_rendered_resident_segment_when_the_spectrum_is_taken_then_each_curve_is_its_own_plane(
    ) {
        let frames = 48_000;
        let segment = Resident::new(
            0,
            0,
            frames,
            Arc::from(tone_planes(1_000.0, frames, 0.5)),
            Some(Arc::from(tone_planes(8_000.0, frames, 0.5))),
        );
        let (original, generated) = segment_spectra(&segment, 24_000).expect("a spectrum");
        let generated = generated.expect("a rendered side");
        assert!((peak_hz(&original) - 1_000.0).abs() < 25.0);
        assert!((peak_hz(&generated) - 8_000.0).abs() < 50.0);
    }

    #[test]
    fn given_a_playhead_inside_a_segment_when_the_window_is_taken_then_it_is_centred_on_it_and_inside(
    ) {
        let frames = 48_000;
        let second = Resident::new(
            1,
            96_000,
            frames,
            Arc::from(tone_planes(1_000.0, frames, 0.5)),
            None,
        );
        let window = window_from_segment(
            second.start(),
            frames,
            96_000 + 10_000,
            |offset, channel| second.dry(offset, channel),
        )
        .expect("a window");
        assert_eq!(window.len(), spectrum::WINDOW * 2);
        let offset = 10_000 - spectrum::WINDOW / 2;
        assert_eq!(window[0], second.dry(offset, 0));
        assert_eq!(
            window[window.len() - 1],
            second.dry(offset + spectrum::WINDOW - 1, 1)
        );
    }

    #[test]
    fn given_a_playhead_at_a_segment_end_when_the_window_is_taken_then_it_stays_inside_the_segment()
    {
        let frames = 48_000;
        let segment = Resident::new(
            0,
            0,
            frames,
            Arc::from(tone_planes(1_000.0, frames, 0.5)),
            None,
        );
        let window = window_from_segment(segment.start(), frames, frames - 1, |offset, channel| {
            segment.dry(offset, channel)
        })
        .expect("a window");
        assert_eq!(window[0], segment.dry(frames - spectrum::WINDOW, 0));
        assert_eq!(
            window[window.len() - 1],
            segment.dry(frames - 1, 1),
            "the window must end on the segment's last frame"
        );
    }

    #[test]
    fn given_a_segment_shorter_than_a_window_when_the_spectrum_is_taken_then_there_is_none() {
        let frames = spectrum::WINDOW - 1;
        let segment = Resident::new(
            0,
            0,
            frames,
            Arc::from(tone_planes(1_000.0, frames, 0.5)),
            None,
        );
        assert!(segment_spectra(&segment, 0).is_none());
    }

    #[test]
    fn given_a_resident_segment_when_the_spectrum_window_is_taken_then_it_allocates_one_window() {
        let frames = 48_000;
        let segment = Resident::new(
            0,
            0,
            frames,
            Arc::from(tone_planes(1_000.0, frames, 0.5)),
            None,
        );
        let (window, allocated) = crate::alloc_counter::measure(|| {
            window_from_segment(segment.start(), frames, 0, |offset, channel| {
                segment.dry(offset, channel)
            })
            .expect("a window")
        });
        println!("spectrum window: {allocated} bytes");
        assert_eq!(window.len(), spectrum::WINDOW * 2);
        assert!(
            allocated <= spectrum::WINDOW * 2 * std::mem::size_of::<f32>(),
            "{allocated} bytes for one window"
        );
    }

    #[test]
    fn given_a_video_url_with_radio_on_when_asked_then_the_mix_is_requested() {
        assert!(radio_requested("https://youtu.be/dQw4w9WgXcQ", true));
        assert!(!radio_requested("https://youtu.be/dQw4w9WgXcQ", false));
    }

    #[test]
    fn given_a_playlist_url_when_asked_for_radio_then_it_is_not_a_seed() {
        // A playlist URL is already a queue; seeding radio from it would replace that.
        assert!(!radio_requested(
            "https://www.youtube.com/playlist?list=PL1",
            true
        ));
        assert!(!radio_requested("song.mp3", true));
    }

    #[test]
    fn given_a_loading_model_past_the_budget_when_the_start_is_decided_then_it_keeps_waiting() {
        // The reported bug: "generating stuck at 0, no GPU, no blue waveform, yet playback
        // started". A cold load measured 23.9 s on this box (16.9 s weights + 2.5 s warmup
        // + the first window), so a 30 s wall-clock timer fired while the weights were still
        // loading. Loading is not slowness: the window must wait, however long it takes.
        assert!(
            !should_start(0, 96_000, true, true, 31.0),
            "a model still loading must not start playback on the original"
        );
        assert!(
            !should_start(0, 96_000, true, true, 5_000.0),
            "there is no timeout on being not ready yet"
        );
    }

    #[test]
    fn given_a_loaded_renderer_behind_the_budget_when_deciding_then_it_starts_on_the_original() {
        // The case the budget exists for: a CPU box at RTF 7.7 will never fill the
        // prebuffer, and silence would be worse than the original.
        assert!(should_start(0, 96_000, true, false, 31.0));
        assert!(
            !should_start(0, 96_000, true, false, 10.0),
            "inside the budget it keeps waiting for the generated band"
        );
    }

    #[test]
    fn given_a_frontier_at_the_prebuffer_when_deciding_then_it_starts_already_enhanced() {
        // The normal path, and the one the product promises: the first note is generated.
        assert!(should_start(96_000, 96_000, true, true, 0.0));
        assert!(should_start(120_000, 96_000, true, true, 0.0));
    }

    #[test]
    fn given_no_frontier_when_deciding_then_it_starts_at_once() {
        // A full-band file or a cache hit has nothing to generate, so there is nothing to
        // wait for and no model is loading.
        assert!(should_start(0, 96_000, false, false, 0.0));
    }

    #[test]
    fn given_a_worker_device_when_asked_then_only_cpu_is_the_slow_case() {
        assert!(is_cpu_device("cpu"));
        assert!(!is_cpu_device("cuda"));
        assert!(!is_cpu_device("mps"));
        assert!(!is_cpu_device("none"));
        assert!(!is_cpu_device("rendering"));
    }

    #[test]
    fn given_the_transport_sections_when_listed_then_they_draw_in_this_order() {
        use TransportSection::*;
        assert_eq!(
            TRANSPORT_SECTIONS,
            [
                Heading,
                Toggle,
                Queue,
                Url,
                Spectrum,
                Waveform,
                KeyHint,
                Playback,
                Progress,
                RenderStatus,
                Live,
                Sink,
                Metadata,
                Disclaimer,
                StreamErrors,
                CpuWarning,
                Error,
            ]
        );
    }

    #[test]
    fn given_the_transport_sections_when_counted_then_each_appears_exactly_once() {
        for section in TRANSPORT_SECTIONS {
            let count = TRANSPORT_SECTIONS
                .iter()
                .filter(|other| **other == section)
                .count();
            assert_eq!(count, 1, "{section:?} is drawn {count} times");
        }
    }

    #[test]
    fn given_a_policy_when_labelled_then_the_line_states_the_preset_latency_and_dropouts() {
        let policy = flacify_core::live_policy(
            flacify_core::Preset::Two,
            &flacify_core::Probe::from_rtf(0.8),
            1_024,
        );
        let line = live_line(&policy, 3, None);
        assert!(line.contains("preset 2 granules"), "{line}");
        assert!(line.contains("latency 2.58 s"), "{line}");
        assert!(
            line.contains("window 1.28 + buffer 1.28 + device 0.02"),
            "{line}"
        );
        assert!(line.contains("dropouts 3"), "{line}");
        assert!(
            !line.contains("RTF"),
            "no ratio before it is measured: {line}"
        );

        let measured = live_line(&policy, 0, Some(0.74));
        assert!(measured.contains("dropouts 0"), "{measured}");
        assert!(measured.ends_with("RTF 0.74"), "{measured}");
    }

    /// A sink route at a fixed latency, with the correction at a known value.
    fn sink_route(fallback: Option<&str>) -> SinkRoute {
        let stats = Arc::new(crate::sink::SinkStats::default());
        SinkRoute {
            from: "BlackHole 2ch".to_string(),
            to: "External Headphones".to_string(),
            fallback: fallback.map(str::to_string),
            cutoff_hz: 15_200,
            ring_capacity_frames: crate::sink::MIN_RING_FRAMES,
            latency_secs: 2.78,
            stats,
            stop: Arc::new(AtomicBool::new(false)),
        }
    }

    #[test]
    fn given_a_sink_route_when_labelled_then_the_line_states_route_latency_and_resyncs() {
        let line = sink_line(&sink_route(None), 4);
        assert!(
            line.contains("BlackHole 2ch → External Headphones"),
            "{line}"
        );
        assert!(line.contains("latency 2.78 s"), "{line}");
        assert!(line.contains("dropouts 4"), "{line}");
        assert!(line.contains("resyncs 0"), "{line}");
        assert!(line.contains("capture signal"), "{line}");
        assert!(!line.contains("(default)"), "nothing fell back: {line}");
    }

    #[test]
    fn given_a_fallback_route_when_labelled_then_the_panel_says_the_default_plays() {
        let line = sink_line(&sink_route(Some("no output matches \"AirPods\"")), 0);
        assert!(line.contains("(default)"), "{line}");
    }

    #[test]
    fn given_a_capture_of_digital_silence_when_labelled_then_the_panel_says_capture_silent() {
        let route = sink_route(None);
        let silent = vec![0.0f32; crate::sink::SILENT_FRAMES * 2];
        let ring = crate::live::Ring::new(crate::sink::MIN_RING_FRAMES, 2);
        let head = std::sync::atomic::AtomicUsize::new(0);
        crate::sink::write_capture(&ring, &head, &route.stats, &silent);

        let line = sink_line(&route, 0);
        assert!(line.contains("capture silent"), "{line}");
        assert!(
            SILENT_HINT.contains("Microphone"),
            "the hint names the permission"
        );
    }

    #[test]
    fn given_a_correction_when_labelled_then_the_line_states_it_and_the_fill() {
        let route = sink_route(None);
        let line = sink_correction_line(&route);
        assert!(line.contains("drift +0 ppm"), "{line}");
        assert!(
            line.contains(&format!(
                "ring fill 0 of {} frames",
                route.ring_capacity_frames
            )),
            "{line}"
        );
        assert!(line.contains("0 dropped"), "{line}");
    }
}
