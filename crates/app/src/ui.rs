//! The window: one track, one switch, and the truth about what has been generated.
//!
//! Playback starts immediately; the renderer fills a frontier behind it. The switch is
//! only enabled where generated audio actually exists, so it can never claim something
//! the file does not have — and once the frontier is comfortably ahead it flips itself,
//! because the point of the product is hearing the difference, not finding the button.

use std::sync::Arc;
use std::time::Duration;

use eframe::egui;

use crate::device::AudioEngine;
use crate::frontier::Frontier;
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
}

pub struct PlayerApp {
    engine: AudioEngine,
    frontier: Option<Arc<Frontier>>,
    info: TrackInfo,
    error: Option<String>,
    promote_margin: usize,
    /// Playback starts when this many frames of generated audio exist, so the first
    /// note is already the product and the frontier stays ahead from there.
    prebuffer_frames: usize,
    started: bool,
    opened_at: std::time::Instant,
}

impl PlayerApp {
    pub fn new(engine: AudioEngine, frontier: Option<Arc<Frontier>>, info: TrackInfo) -> Self {
        // One second of slack before the switch is offered, so a toggle cannot land on
        // the edge of what exists and fall back mid-ramp.
        let promote_margin = engine.device_rate as usize;
        let prebuffer_frames = (engine.device_rate as usize).saturating_mul(2);
        Self {
            engine,
            frontier,
            info,
            error: None,
            promote_margin,
            prebuffer_frames,
            started: false,
            opened_at: std::time::Instant::now(),
        }
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
        ui.label(
            egui::RichText::new("space toggles   ·   p plays   ·   ←/→ seek 5 s   ·   esc quits")
                .weak()
                .size(11.0),
        );

        ui.add_space(16.0);
        let mut scrub = position;
        if ui
            .add(
                egui::Slider::new(&mut scrub, 0.0..=duration)
                    .show_value(false)
                    .trailing_fill(true),
            )
            .changed()
        {
            controls.seek_secs(scrub, rate);
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

        ui.add_space(12.0);
        ui.horizontal(|ui| {
            let playing = controls.is_playing();
            if ui.button(if playing { "Pause" } else { "Play" }).clicked() {
                controls.set_playing(!playing);
            }
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

        ui.add_space(10.0);
        let source = if self.info.from_cache {
            "cached"
        } else {
            "rendering now"
        };
        ui.label(
            egui::RichText::new(format!(
                "{} · cutoff {} Hz · {} · out {} Hz × {} ch",
                self.info.device, self.info.cutoff_hz, source, rate, self.engine.device_channels
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

pub fn run(app: PlayerApp) -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([560.0, 380.0])
            .with_resizable(false)
            .with_title("flacify"),
        ..Default::default()
    };
    eframe::run_native("flacify", options, Box::new(|_cc| Ok(Box::new(app))))
}
