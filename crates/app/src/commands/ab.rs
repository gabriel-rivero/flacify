//! `flacify ab`: the blind trials over a real render, and the report they produce.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use flacify_core::SAMPLE_RATE;

use crate::ab::{self, LevelCheck, Plan, Position, ReportMeta, Side};
use crate::device;
use crate::render;
use crate::session::Session;
use crate::setup;

/// Silence between a trial's two excerpts, so the boundary is audible.
const EXCERPT_GAP: Duration = Duration::from_millis(400);

/// How long past the excerpt's own length the device may take before the run gives up.
const PLAY_PATIENCE_SECS: f64 = 4.0;

pub(crate) fn run(file: &Path, trials: u32, seed: u64, export: Option<&Path>) -> Result<()> {
    let paths = setup::Paths::discover()?;
    if !paths.is_ready() {
        bail!("the model is not installed yet; run `flacify setup` once");
    }
    let session = Session::new(&paths, false)?;
    let sides = load_sides(&session, file)?;
    let excerpt_frames = (ab::EXCERPT_SECS * f64::from(SAMPLE_RATE)).round() as usize;
    let mut plan = Plan::new(seed, trials, sides.frames, excerpt_frames)?;

    let levels = measure_levels(&plan, &sides, excerpt_frames);
    let file_label = file.display().to_string();
    let meta = ReportMeta {
        file: &file_label,
        device: "(not opened)",
        model: &sides.model,
        sample_rate: SAMPLE_RATE,
    };
    let report_path = destination(file, export);

    // The gate runs before the first excerpt: a run whose sides are already unequal would
    // spend the operator's time producing a score that measures loudness.
    if let Err(gate) = ab::level_gate(&levels) {
        write_report(&report_path, &ab::markdown(&meta, &plan, &levels))?;
        return Err(gate);
    }
    match ab::worst_level_diff(&levels) {
        Some(worst) => println!("level     matched, worst {worst:.2} dB"),
        None => println!("level     matched"),
    }
    println!(
        "plan      {} trials of {:.1} s, seed {seed}",
        plan.trials().len(),
        ab::EXCERPT_SECS
    );
    println!();

    let (original, generated) = sides.interleaved();
    let engine = device::AudioEngine::start(&original, Some(&generated))?;
    // The blind test owns the switch (D8): the enhanced side never appears on its own, and
    // every switch from here is the trial's.
    engine.controls.hold_promotion();
    engine.controls.use_enhanced.store(false, Ordering::Relaxed);
    let device_name = engine.device_name.clone();
    let meta = ReportMeta {
        device: &device_name,
        ..meta
    };

    run_trials(&engine, &mut plan)?;
    drop(engine);

    write_report(&report_path, &ab::markdown(&meta, &plan, &levels))?;
    match plan.score() {
        Some(score) => println!(
            "scored    {} of {} correct, p = {}, 95% CI {:.1}% to {:.1}%",
            score.correct,
            score.answered,
            ab::p_label(score.p_value),
            score.ci_low * 100.0,
            score.ci_high * 100.0
        ),
        None => println!("scored    no trial was answered"),
    }
    Ok(())
}

/// The file's two sides, whole and channel-major at 48 kHz.
struct Sides {
    original: Vec<f32>,
    generated: Vec<f32>,
    frames: usize,
    /// The accelerator the render used, or the one the cache entry recorded.
    model: String,
}

impl Sides {
    /// Interleaved stereo, which is what the device opens with.
    fn interleaved(&self) -> (Vec<f32>, Vec<f32>) {
        (
            render::interleave_stereo(&self.original, self.frames),
            render::interleave_stereo(&self.generated, self.frames),
        )
    }
}

/// Decode the file and render or load its generated side, once.
fn load_sides(session: &Session, file: &Path) -> Result<Sides> {
    let loaded = render::load(file, session.cache().root())?;
    if loaded.analysis.looks_full_band() {
        bail!(
            "{} is already full-band: there is no generated side to compare",
            file.display()
        );
    }
    let frames = loaded.track.frames;
    let cancel = AtomicBool::new(false);
    let (rendered, from_cache) = render::enhance_or_cached(
        &loaded.track,
        &loaded.source_hash,
        render::Job {
            session: session.worker(),
            cutoff_hz: Some(loaded.analysis.cutoff_hz),
            seed: render::seed_for(&loaded.track),
            splice_low_band: false,
        },
        render::Store {
            cache: session.cache(),
            model_id: session.model_id(),
        },
        &cancel,
        &mut |_| {},
    )?;
    ensure!(
        rendered.planar.len() == frames * 2,
        "the render returned {} samples for a track of {frames} frames",
        rendered.planar.len()
    );
    if from_cache {
        eprintln!("cache hit: nothing to render");
    }
    Ok(Sides {
        original: loaded.track.planar,
        generated: rendered.planar,
        frames,
        model: rendered.device,
    })
}

/// Measure every excerpt on both sides, before a single one is played.
fn measure_levels(plan: &Plan, sides: &Sides, excerpt_frames: usize) -> Vec<LevelCheck> {
    plan.trials()
        .iter()
        .map(|trial| {
            let original = excerpt(
                &sides.original,
                sides.frames,
                trial.excerpt_start,
                excerpt_frames,
            );
            let generated = excerpt(
                &sides.generated,
                sides.frames,
                trial.excerpt_start,
                excerpt_frames,
            );
            ab::measure_level(trial, &original, &generated)
        })
        .collect()
}

/// `[left, right]` of one excerpt out of a channel-major stereo buffer.
fn excerpt(planar: &[f32], frames: usize, start: usize, len: usize) -> [&[f32]; 2] {
    debug_assert!(start + len <= frames);
    [
        &planar[start..start + len],
        &planar[frames + start..frames + start + len],
    ]
}

/// Play each trial's two excerpts in its order and record what the operator says.
///
/// The truth is never printed here: the operator learns it from the report, after the
/// last trial.
fn run_trials(engine: &device::AudioEngine, plan: &mut Plan) -> Result<()> {
    let total = plan.trials().len();
    let mut input = io::stdin().lock();
    let mut output = io::stdout();
    for index in 0..total {
        let (start, order) = {
            let trial = &plan.trials()[index];
            (trial.excerpt_start, trial.order)
        };
        println!("trial {}/{total}", index + 1);
        play_excerpt(engine, start, plan.excerpt_frames(), order.first(), "first")?;
        std::thread::sleep(EXCERPT_GAP);
        play_excerpt(
            engine,
            start,
            plan.excerpt_frames(),
            order.second(),
            "second",
        )?;
        let answer = ask(&mut input, &mut output)?;
        plan.record(index as u32, answer)?;
        println!();
    }
    Ok(())
}

/// Play one side of one excerpt and wait for it to finish.
fn play_excerpt(
    engine: &device::AudioEngine,
    start: usize,
    frames: usize,
    side: Side,
    which: &str,
) -> Result<()> {
    let controls = &engine.controls;
    let start = at_device_rate(start, engine.device_rate);
    let frames = at_device_rate(frames, engine.device_rate);
    println!("playing   the {which} excerpt");
    controls
        .use_enhanced
        .store(side == Side::Generated, Ordering::Relaxed);
    controls.seek_frames(start);
    controls.set_playing(true);

    let patience =
        Duration::from_secs_f64(frames as f64 / f64::from(engine.device_rate) + PLAY_PATIENCE_SECS);
    let deadline = Instant::now() + patience;
    while controls.position_frames() < start + frames {
        if Instant::now() >= deadline {
            controls.set_playing(false);
            let played = controls.position_frames().saturating_sub(start);
            bail!("the device played {played} frames of {frames} before the excerpt timed out");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    controls.set_playing(false);
    Ok(())
}

/// Ask for one answer, re-prompting until the line is one of the two the prompt offers.
fn ask(input: &mut impl io::BufRead, output: &mut impl Write) -> Result<Position> {
    loop {
        write!(
            output,
            "answer    which was the generated one? 1 = first, 2 = second: "
        )?;
        output.flush()?;
        let mut line = String::new();
        if input.read_line(&mut line)? == 0 {
            bail!("the input ended before the last trial was answered");
        }
        if let Some(answer) = ab::parse_answer(&line) {
            return Ok(answer);
        }
        writeln!(output, "answer 1 (first) or 2 (second)")?;
    }
}

/// A 48 kHz frame index at the device's rate. The mixer's playhead counts device frames.
fn at_device_rate(frame_48k: usize, device_rate: u32) -> usize {
    (frame_48k as u64 * u64::from(device_rate) / u64::from(SAMPLE_RATE)) as usize
}

fn destination(file: &Path, export: Option<&Path>) -> PathBuf {
    match export {
        Some(path) => path.to_path_buf(),
        None => {
            let mut name = file
                .file_name()
                .map(|name| name.to_os_string())
                .unwrap_or_default();
            name.push(".flacify-ab.md");
            file.with_file_name(name)
        }
    }
}

fn write_report(path: &Path, text: &str) -> Result<()> {
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))?;
    println!("report    {}", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn given_a_file_without_an_export_when_asked_for_the_destination_then_it_sits_beside_it() {
        assert_eq!(
            destination(Path::new("/music/a.flac"), None),
            PathBuf::from("/music/a.flac.flacify-ab.md")
        );
    }

    #[test]
    fn given_an_export_when_asked_for_the_destination_then_it_is_used_as_given() {
        assert_eq!(
            destination(
                Path::new("/music/a.flac"),
                Some(Path::new("/tmp/report.md"))
            ),
            PathBuf::from("/tmp/report.md")
        );
    }

    #[test]
    fn given_a_48_khz_frame_when_converted_then_it_lands_at_the_device_rate() {
        assert_eq!(at_device_rate(0, 44_100), 0);
        assert_eq!(at_device_rate(48_000, 44_100), 44_100);
        assert_eq!(at_device_rate(288_000, 44_100), 264_600);
        assert_eq!(at_device_rate(288_000, 48_000), 288_000);
    }

    #[test]
    fn given_a_channel_major_buffer_when_one_excerpt_is_taken_then_both_channels_come_from_it() {
        // Two frames, then the right channel's two frames: [L0, L1, R0, R1].
        let planar = [1.0f32, 2.0, 10.0, 20.0];
        let [left, right] = excerpt(&planar, 2, 1, 1);
        assert_eq!(left, [2.0]);
        assert_eq!(right, [20.0]);
    }

    #[test]
    fn given_an_answer_then_a_bad_line_when_asked_then_the_prompt_comes_again() {
        let mut input = io::Cursor::new(b"maybe\n2\n");
        let mut output = Vec::new();
        let answer = ask(&mut input, &mut output).expect("an answer");
        assert_eq!(answer, Position::Second);
        let text = String::from_utf8(output).expect("utf-8");
        assert_eq!(text.matches("which was the generated one").count(), 2);
        assert!(text.contains("answer 1 (first) or 2 (second)"), "{text}");
    }

    #[test]
    fn given_no_more_input_when_asked_then_the_run_fails_instead_of_guessing() {
        let mut input = io::Cursor::new(b"");
        let mut output = Vec::new();
        let error = ask(&mut input, &mut output)
            .expect_err("no answer")
            .to_string();
        assert!(error.contains("input ended"), "{error}");
    }
}
