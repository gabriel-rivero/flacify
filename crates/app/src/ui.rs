//! The window: one toggle, one transport, nothing else.
//!
//! The expensive work happens on a thread that this UI only watches through atomics;
//! `update` polls a channel, starts the audio device on the main thread when the
//! buffers are ready, and then draws. While playing it asks eframe for 30 fps, and
//! when nothing is happening it does not repaint at all.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use eframe::egui;

use crate::device::AudioEngine;

/// Everything the model produced, ready to play.
pub struct Prepared {
    pub track_name: String,
    /// Interleaved stereo at 48 kHz, same length as `enhanced`.
    pub original: Vec<f32>,
    pub enhanced: Option<Vec<f32>>,
    pub cutoff_hz: u32,
    pub render_secs: f64,
    pub audio_secs: f64,
    pub peak_before: f32,
    pub limited_samples: usize,
    pub device: String,
    /// True when the audio came from the cache instead of a fresh render.
    pub from_cache: bool,
    pub cache_bytes: u64,
}

/// Progress the worker thread publishes while it runs.
pub struct Progress {
    pub done: Arc<AtomicUsize>,
    total: AtomicUsize,
    pub status: Arc<Mutex<String>>,
}

impl Progress {
    pub fn new(total: usize) -> Self {
        Self {
            done: Arc::new(AtomicUsize::new(0)),
            total: AtomicUsize::new(total),
            status: Arc::new(Mutex::new("starting".to_string())),
        }
    }

    /// The total is only known once the file is decoded, which happens on the same
    /// thread that reports progress.
    pub fn set_total(&self, total: usize) {
        self.total.store(total, Ordering::Relaxed);
    }

    pub fn total(&self) -> usize {
        self.total.load(Ordering::Relaxed)
    }

    pub fn set_status(&self, text: impl Into<String>) {
        if let Ok(mut guard) = self.status.lock() {
            *guard = text.into();
        }
    }

    pub fn set_done(&self, done: usize) {
        self.done.store(done, Ordering::Relaxed);
    }
}

enum Stage {
    Working,
    Ready(Box<Prepared>),
    Failed(String),
}

pub struct PlayerApp {
    receiver: Receiver<Result<Prepared, String>>,
    progress: Arc<Progress>,
    stage: Stage,
    engine: Option<AudioEngine>,
    last_error: Option<String>,
}

impl PlayerApp {
    pub fn new(receiver: Receiver<Result<Prepared, String>>, progress: Arc<Progress>) -> Self {
        Self {
            receiver,
            progress,
            stage: Stage::Working,
            engine: None,
            last_error: None,
        }
    }

    fn poll(&mut self) {
        match self.receiver.try_recv() {
            Ok(Ok(prepared)) => {
                self.stage = Stage::Ready(Box::new(prepared));
            }
            Ok(Err(message)) => self.stage = Stage::Failed(message),
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                if !matches!(self.stage, Stage::Ready(_) | Stage::Failed(_)) {
                    self.stage = Stage::Failed("the worker thread stopped without a result".into());
                }
            }
        }
    }

    fn draw_working(&self, ui: &mut egui::Ui) {
        let status = self
            .progress
            .status
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default();
        ui.add_space(12.0);
        ui.heading("Preparing");
        ui.label(status);
        let total = self.progress.total();
        let fraction = if total == 0 {
            0.0
        } else {
            self.progress.done.load(Ordering::Relaxed) as f32 / total as f32
        };
        ui.add(
            egui::ProgressBar::new(fraction)
                .desired_width(f32::INFINITY)
                .text(format!("{:.0}%", fraction * 100.0)),
        );
    }

    fn draw_player(&self, ui: &mut egui::Ui) {
        let Stage::Ready(prepared) = &self.stage else {
            return;
        };

        ui.add_space(8.0);
        ui.label(egui::RichText::new(&prepared.track_name).size(15.0));
        ui.add_space(10.0);

        if let Some(engine) = &self.engine {
            let controls = Arc::clone(&engine.controls);
            let enhanced = controls.is_enhanced();
            let label = if !controls.has_enhanced {
                "Full-band source"
            } else if enhanced {
                "Hearing: generated top"
            } else {
                "Hearing: original"
            };
            let button = egui::Button::new(egui::RichText::new(label).size(22.0))
                .min_size(egui::vec2(360.0, 64.0));
            if ui.add_enabled(controls.has_enhanced, button).clicked() {
                controls.toggle();
            }
            ui.add_space(6.0);
            ui.label(
                egui::RichText::new("space toggles   ·   p plays and pauses   ·   esc quits")
                    .weak()
                    .size(11.0),
            );

            ui.add_space(14.0);
            let position = controls.position_secs(engine.device_rate);
            let duration = prepared.audio_secs.max(0.001);
            // The playhead is this slider's value, so dragging it *is* the seek.
            let mut scrub = position;
            if ui
                .add(
                    egui::Slider::new(&mut scrub, 0.0..=duration)
                        .show_value(false)
                        .trailing_fill(true),
                )
                .changed()
            {
                controls.seek_secs(scrub, engine.device_rate);
            }
            ui.label(
                egui::RichText::new(format!(
                    "{:02}:{:02} / {:02}:{:02}   ·   drag to seek, arrows jump 5 s",
                    (position as u64) / 60,
                    (position as u64) % 60,
                    (duration as u64) / 60,
                    (duration as u64) % 60
                ))
                .weak()
                .size(11.0),
            );

            ui.add_space(8.0);
            let playing = controls.is_playing();
            if ui.button(if playing { "Pause" } else { "Play" }).clicked() {
                controls.set_playing(!playing);
            }
            ui.label(
                egui::RichText::new(format!(
                    "model on {} · output {} Hz × {} ch",
                    prepared.device, engine.device_rate, engine.device_channels
                ))
                .weak()
                .size(11.0),
            );
            if !playing && position >= duration - 0.05 {
                ui.label(
                    egui::RichText::new("end of track — press play to restart from the top").weak(),
                );
            }
        }

        ui.add_space(12.0);
        ui.label(
            egui::RichText::new(format!(
                "cutoff {} Hz · rendered in {:.0} s for {:.0} s of audio (RTF {:.2})",
                prepared.cutoff_hz,
                prepared.render_secs,
                prepared.audio_secs,
                prepared.render_secs / prepared.audio_secs.max(0.001)
            ))
            .weak()
            .size(11.0),
        );
        if prepared.limited_samples > 0 {
            ui.label(
                egui::RichText::new(format!(
                    "peaks limited: {:.2} -> full scale, {} samples above the knee",
                    prepared.peak_before, prepared.limited_samples
                ))
                .weak()
                .size(11.0),
            );
        }
        if prepared.cache_bytes > 0 {
            ui.label(
                egui::RichText::new(format!(
                    "cache: {:.0} MB on disk",
                    prepared.cache_bytes as f64 / 1e6
                ))
                .weak()
                .size(11.0),
            );
        }
        ui.label(
            egui::RichText::new("the top band is generated, not recovered")
                .weak()
                .size(11.0),
        );

        if let Some(error) = &self.last_error {
            ui.add_space(8.0);
            ui.colored_label(egui::Color32::from_rgb(220, 90, 90), error);
        }
    }
}

impl eframe::App for PlayerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll();
        if matches!(self.stage, Stage::Ready(_))
            && self.engine.is_none()
            && self.last_error.is_none()
        {
            if let Stage::Ready(prepared) = &self.stage {
                match AudioEngine::start(&prepared.original, prepared.enhanced.as_deref()) {
                    Ok(engine) => {
                        engine.controls.set_playing(true);
                        self.engine = Some(engine);
                    }
                    Err(error) => {
                        tracing::error!(%error, "could not open the output device");
                        self.last_error = Some(format!("{error:#}"));
                    }
                }
            }
        }

        if let Some(engine) = &self.engine {
            let controls = &engine.controls;
            ctx.input(|input| {
                if input.key_pressed(egui::Key::Space) {
                    controls.toggle();
                }
                if input.key_pressed(egui::Key::P) {
                    controls.set_playing(!controls.is_playing());
                }
                let rate = engine.device_rate;
                let backwards = controls.position_frames().saturating_sub(5 * rate as usize);
                if input.key_pressed(egui::Key::ArrowLeft) {
                    controls.seek_frames(backwards);
                }
                if input.key_pressed(egui::Key::ArrowRight) {
                    controls.seek_frames(controls.position_frames() + 5 * rate as usize);
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
        } else {
            ctx.request_repaint_after(Duration::from_millis(100));
        }

        egui::CentralPanel::default().show(ctx, |ui| match &self.stage {
            Stage::Working => self.draw_working(ui),
            Stage::Failed(message) => {
                ui.heading("Could not start");
                ui.colored_label(egui::Color32::from_rgb(220, 90, 90), message);
            }
            Stage::Ready(_) => self.draw_player(ui),
        });
    }
}

pub fn run(app: PlayerApp) -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([520.0, 320.0])
            .with_resizable(false)
            .with_title("flacify"),
        ..Default::default()
    };
    eframe::run_native("flacify", options, Box::new(|_cc| Ok(Box::new(app))))
}
