//! The window: one track, one switch, and the truth about what has been generated.
//!
//! Playback starts immediately; the renderer fills a frontier behind it. The switch is
//! only enabled where generated audio actually exists, so it can never claim something
//! the file does not have — and once the frontier is comfortably ahead it flips itself,
//! because the point of the product is hearing the difference, not finding the button.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use eframe::egui;

use crate::device::AudioEngine;
use crate::frontier::Frontier;
use crate::render_status::RenderStatus;
use crate::session::{Playback, Session};
use crate::spectrum;

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

pub struct PlayerApp {
    /// One worker for the whole window, so a dropped file does not reload the model.
    session: Session,
    engine: AudioEngine,
    frontier: Option<Arc<Frontier>>,
    /// Present only when this window is watching a background render.
    status: Option<Arc<RenderStatus>>,
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
}

/// No more waveform bars than this; a pixel is narrower than a bar worth drawing.
const WAVEFORM_MAX_BARS: usize = 512;

impl PlayerApp {
    pub fn new(session: Session, playback: Playback) -> Self {
        let engine = playback.engine;
        let (waveform, waveform_bar_frames) = waveform_bars(
            &engine.display_original,
            engine.device_channels,
            engine.device_rate,
            WAVEFORM_MAX_BARS,
        );
        // One second of slack before the switch is offered, so a toggle cannot land on
        // the edge of what exists and fall back mid-ramp.
        let promote_margin = engine.device_rate as usize;
        let prebuffer_frames = (engine.device_rate as usize).saturating_mul(2);
        Self {
            session,
            engine,
            frontier: playback.frontier,
            status: playback.status,
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
        }
    }

    /// Replace what is playing, keeping the worker. A failed load leaves the current
    /// file alone and says why, instead of stopping playback with nothing to show.
    fn load(&mut self, path: &Path) {
        let mut playback = match self.session.open(path) {
            Ok(playback) => playback,
            Err(error) => {
                self.error = Some(format!("could not play {}: {error:#}", path.display()));
                return;
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
        self.info = playback.info;
        self.path = playback.path;
        self.promote_margin = self.engine.device_rate as usize;
        self.prebuffer_frames = (self.engine.device_rate as usize).saturating_mul(2);
        let (waveform, bar_frames) = waveform_bars(
            &self.engine.display_original,
            self.engine.device_channels,
            self.engine.device_rate,
            WAVEFORM_MAX_BARS,
        );
        self.waveform = waveform;
        self.waveform_bar_frames = bar_frames;
        self.started = false;
        self.opened_at = std::time::Instant::now();
        self.title_dirty = true;
        self.error = None;
    }

    fn ready_frames(&self) -> usize {
        match &self.frontier {
            Some(frontier) => frontier.ready_frames(),
            None => self.engine.controls.frames,
        }
    }

    fn draw_transport(&mut self, ui: &mut egui::Ui) {
        let rate = self.engine.device_rate;
        let controls = Arc::clone(&self.engine.controls);
        let position = controls.position_secs(rate);
        let duration = self.info.audio_secs.max(0.001);
        let prepared_frames = self.ready_frames();

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

        // The switch. Enabled only where generated audio exists.
        let generated_here =
            prepared_frames >= controls.position_frames() + self.promote_margin.min(48_000);
        let live = controls.is_enhanced();
        let label = if !controls.has_enhanced {
            "Full-band source".to_string()
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
            .add_enabled(controls.has_enhanced && generated_here, button)
            .clicked()
        {
            controls.toggle();
        }

        ui.add_space(10.0);
        self.draw_spectrum(ui);

        ui.add_space(6.0);
        self.draw_waveform(ui);

        ui.add_space(6.0);
        ui.label(
            egui::RichText::new(
                "space toggles   ·   p plays   ·   ←/→ seek 5 s   ·   drop a file   ·   esc quits",
            )
            .weak()
            .size(11.0),
        );

        ui.add_space(10.0);
        ui.horizontal(|ui| {
            let playing = controls.is_playing();
            if ui.button(if playing { "Pause" } else { "Play" }).clicked() {
                controls.set_playing(!playing);
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
        });

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

        ui.add_space(8.0);
        self.draw_render_status(ui, prepared_frames);

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
                "{} · cutoff {} Hz · {} · out {} Hz × {} ch{}",
                self.info.device,
                self.info.cutoff_hz,
                source,
                rate,
                self.engine.device_channels,
                matched
            ))
            .weak()
            .size(11.0),
        );
        ui.label(
            egui::RichText::new("the top band is generated, not recovered")
                .weak()
                .size(11.0),
        );

        let errors = self
            .engine
            .stream_errors
            .load(std::sync::atomic::Ordering::Relaxed);
        if errors > 0 {
            ui.label(
                egui::RichText::new(format!("device stream errors: {errors}"))
                    .weak()
                    .size(11.0),
            );
        }

        if let Some(error) = &self.error {
            ui.add_space(8.0);
            ui.colored_label(egui::Color32::from_rgb(220, 90, 90), error);
        }
    }

    /// The two things the frontier cannot say on its own: that the render stopped, and
    /// that it is not going to catch up. Both otherwise look the same from here — an
    /// empty generated curve and a switch that never lights.
    fn draw_render_status(&self, ui: &mut egui::Ui, prepared_frames: usize) {
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
    /// Start once there is something worth hearing, or after a patience budget so a
    /// slow machine still gets the original rather than silence.
    fn maybe_start(&mut self) {
        if self.started {
            return;
        }
        let ready = self.ready_frames();
        let patient = self.opened_at.elapsed().as_secs_f64() > 30.0;
        if ready >= self.prebuffer_frames || patient || self.frontier.is_none() {
            self.started = true;
            self.engine.controls.set_playing(true);
        }
    }
}

impl PlayerApp {
    /// The whole track as per-bar RMS, with the render frontier marked: bars the renderer
    /// has reached are bright, the rest are dim. Click or drag anywhere to seek; this is
    /// the transport, not a picture.
    fn draw_waveform(&self, ui: &mut egui::Ui) {
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
                    if !controls.has_enhanced || index * self.waveform_bar_frames < prepared {
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
        let height = 108.0;
        let (response, painter) = ui.allocate_painter(
            egui::vec2(ui.available_width(), height),
            egui::Sense::hover(),
        );
        let rect = response.rect;
        painter.rect_filled(rect, 3.0, egui::Color32::from_rgb(17, 19, 23));

        let channels = self.engine.device_channels;
        let position = self.engine.controls.position_frames();
        let Some(original) = spectrum::analyse(&self.engine.display_original, channels, position)
        else {
            return;
        };
        let generated = match &self.engine.display_generated {
            Some(buffer) => spectrum::analyse(buffer, channels, position),
            None => self.frontier.as_ref().and_then(|frontier| {
                window_from_frontier(frontier, position)
                    .and_then(|window| spectrum::analyse(&window, 2, spectrum::WINDOW / 2))
            }),
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
    let peak = out.iter().copied().fold(0.0f32, f32::max);
    if peak > 1e-6 {
        for value in &mut out {
            *value = (*value / peak).clamp(0.0, 1.0);
        }
    }
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

        // The window takes another file without restarting the app, which is what lets
        // the resident worker pay for itself: the second file skips the weight load.
        let dropped = ctx.input(|input| {
            input
                .raw
                .dropped_files
                .iter()
                .find_map(|file| file.path.clone())
        });
        if let Some(path) = dropped {
            self.load(&path);
        }

        self.maybe_start();
        if let Some(engine) = Some(&self.engine) {
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
            });
            if controls.is_playing() {
                ctx.request_repaint_after(Duration::from_millis(33));
            } else {
                ctx.request_repaint_after(Duration::from_millis(250));
            }
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
}
