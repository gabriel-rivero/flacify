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

/// What the window needs to know about the track, none of it expensive.
pub struct TrackInfo {
    pub name: String,
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
        ui.label(egui::RichText::new(&self.info.name).size(16.0));
        ui.add_space(10.0);

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

        ui.add_space(8.0);
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
