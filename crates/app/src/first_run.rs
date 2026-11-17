//! The first run: provision the sidecar from inside the window.
//!
//! `flacify play` used to stop with "run `flacify setup`" on a machine without the model.
//! A released binary should not require a terminal, so when the install is missing this
//! window reports the size before it starts, runs the same `setup::ensure` on a thread,
//! shows which step is running, and hands the same window to the player when the file is
//! ready. The CLI command stays for scripted installs and for `--smoke`, which cannot
//! click through a dialog.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use eframe::egui;

use crate::session::Session;
use crate::setup::{self, Paths, Progress};
use crate::ui;

/// What the setup thread reports, in a form the window can read without a lock on the
/// download path: atomics for the hot fields, a mutex for the two strings.
#[derive(Default)]
struct SetupState {
    step: Mutex<Option<String>>,
    fraction: AtomicU32,
    done: AtomicBool,
    failure: Mutex<Option<String>>,
}

impl SetupState {
    fn report(&self, step: &str, fraction: f32) {
        if let Ok(mut current) = self.step.lock() {
            *current = Some(step.to_string());
        }
        self.fraction
            .store(fraction.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    }

    fn finish(&self) {
        self.done.store(true, Ordering::Release);
    }

    fn fail(&self, message: String) {
        if let Ok(mut failure) = self.failure.lock() {
            *failure = Some(message);
        }
        self.finish();
    }

    fn snapshot(&self) -> (Option<String>, f32, bool, Option<String>) {
        let step = self.step.lock().ok().and_then(|step| step.clone());
        let fraction = f32::from_bits(self.fraction.load(Ordering::Relaxed));
        let done = self.done.load(Ordering::Acquire);
        let failure = self.failure.lock().ok().and_then(|failure| failure.clone());
        (step, fraction, done, failure)
    }
}

pub struct FirstRun {
    paths: Paths,
    file: PathBuf,
    splice: bool,
    state: Arc<SetupState>,
    started: bool,
    /// Set once the install is ready and the file has been opened.
    player: Option<ui::PlayerApp>,
    error: Option<String>,
}

impl FirstRun {
    pub fn new(paths: Paths, file: PathBuf, splice: bool) -> Self {
        Self {
            paths,
            file,
            splice,
            state: Arc::new(SetupState::default()),
            started: false,
            player: None,
            error: None,
        }
    }

    /// Run the download on its own thread; the window stays responsive and can be closed.
    fn start(&mut self) {
        self.state = Arc::new(SetupState::default());
        self.started = true;
        self.error = None;
        let state = Arc::clone(&self.state);
        let paths = self.paths.clone();
        std::thread::spawn(move || {
            let reporter = Arc::clone(&state);
            let progress = Progress::new(move |step, fraction| reporter.report(step, fraction));
            match setup::ensure(&paths, progress) {
                Ok(()) => state.finish(),
                Err(error) => state.fail(format!("{error:#}")),
            }
        });
    }

    fn total_gb() -> f64 {
        setup::plan().iter().map(|step| step.bytes).sum::<u64>() as f64 / 1e9
    }

    fn step_label(id: &str) -> String {
        setup::plan()
            .iter()
            .find(|step| step.id == id)
            .map(|step| step.label.to_string())
            .unwrap_or_else(|| id.to_string())
    }

    /// Turn a finished install into the player, once.
    fn open_player(&mut self) {
        if self.player.is_some() || self.error.is_some() {
            return;
        }
        match Session::new(&self.paths, self.splice)
            .and_then(|session| session.open(&self.file).map(|playback| (session, playback)))
        {
            Ok((session, playback)) => self.player = Some(ui::PlayerApp::new(session, playback)),
            Err(error) => self.error = Some(format!("{error:#}")),
        }
    }
}

impl eframe::App for FirstRun {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        if let Some(player) = &mut self.player {
            player.update(ctx, frame);
            return;
        }

        let (step, fraction, done, failure) = self.state.snapshot();
        if done && failure.is_none() {
            self.open_player();
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(12.0);
            ui.label(egui::RichText::new("flacify").size(20.0).strong());
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new(format!(
                    "The model is installed once: about {:.1} GB of Python, PyTorch and \
                     weights. Nothing is uploaded, and the app only uses it while it runs.",
                    Self::total_gb()
                ))
                .weak()
                .size(12.0),
            );
            ui.add_space(14.0);

            if let Some(failure) = failure {
                ui.colored_label(
                    egui::Color32::from_rgb(220, 90, 90),
                    format!("setup failed: {failure}"),
                );
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new(
                        "Retry skips the steps that already finished; a dropped connection \
                         restarts the step it was on.",
                    )
                    .weak()
                    .size(11.0),
                );
                ui.add_space(10.0);
                if ui.button("Retry").clicked() {
                    self.start();
                }
            } else if let Some(error) = &self.error {
                ui.label("The install finished, but the file could not be opened:");
                ui.colored_label(egui::Color32::from_rgb(220, 90, 90), error);
            } else if !self.started {
                if ui
                    .add(egui::Button::new("Download and set up").min_size(egui::vec2(220.0, 34.0)))
                    .clicked()
                {
                    self.start();
                }
                ui.add_space(10.0);
                ui.label(
                    egui::RichText::new("or run `flacify setup` from a terminal")
                        .weak()
                        .size(11.0),
                );
            } else if done {
                ui.label("Setup complete; opening the file…");
            } else {
                let label = step
                    .as_deref()
                    .map(Self::step_label)
                    .unwrap_or_else(|| "starting".to_string());
                ui.label(format!("Installing: {label}"));
                ui.add_space(6.0);
                ui.add(
                    egui::ProgressBar::new(fraction)
                        .desired_width(f32::INFINITY)
                        .text(format!("{:.0}%", fraction * 100.0)),
                );
                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new(
                        "You can close this window; the next run picks up from the steps \
                         that finished.",
                    )
                    .weak()
                    .size(11.0),
                );
            }

            ui.add_space(14.0);
            if ui.button("Quit").clicked() {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        });

        if self.player.is_some() {
            ctx.request_repaint();
        } else {
            ctx.request_repaint_after(Duration::from_millis(120));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn given_a_report_when_snapshotted_then_the_step_and_fraction_survive() {
        let state = SetupState::default();
        state.report("python", 0.25);
        let (step, fraction, done, failure) = state.snapshot();
        assert_eq!(step.as_deref(), Some("python"));
        assert!((fraction - 0.25).abs() < 1e-6);
        assert!(!done);
        assert!(failure.is_none());
    }

    #[test]
    fn given_a_failure_when_snapshotted_then_it_is_done_and_named() {
        let state = SetupState::default();
        state.fail("no network".to_string());
        let (_, _, done, failure) = state.snapshot();
        assert!(done);
        assert_eq!(failure.as_deref(), Some("no network"));
    }

    #[test]
    fn given_a_finished_report_when_snapshotted_then_it_is_done_without_an_error() {
        let state = SetupState::default();
        state.report("weights", 1.0);
        state.finish();
        let (_, fraction, done, failure) = state.snapshot();
        assert!((fraction - 1.0).abs() < 1e-6);
        assert!(done);
        assert!(failure.is_none());
    }

    #[test]
    fn given_the_install_plan_when_summed_then_the_window_can_state_its_size() {
        assert!(FirstRun::total_gb() > 0.0);
    }
}
