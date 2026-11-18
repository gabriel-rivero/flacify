//! `flacify play`: the window, and the headless smoke run beside it.

use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::Instant;

use anyhow::{Context, Result};
use flacify_core::{Preset, Probe, SAMPLE_RATE};

use crate::first_run;
use crate::live;
use crate::session::{needs_setup, Playback, Session};
use crate::setup;
use crate::ui;

/// What `play` was asked for, one field per flag group.
pub(crate) struct Request<'a> {
    pub file: &'a Path,
    pub smoke: bool,
    pub splice: bool,
    /// Present when `--live` was asked for.
    pub live: Option<LiveOptions>,
    /// Seconds a smoke run listens before it reports.
    pub watch_secs: f64,
}

/// The two numbers `--live` needs before a policy exists: which window, and what the
/// machine's compute ratio is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LiveOptions {
    pub preset: Preset,
    pub rtf: f64,
}

impl LiveOptions {
    /// The preset `--window` names and the ratio `--rtf` gives, checked before any work.
    pub fn parse(window: &str, rtf: f64) -> Result<Self> {
        let preset = Preset::parse(window)
            .with_context(|| format!("--window {window}: the presets are 1, 2, 4 and 8"))?;
        anyhow::ensure!(
            rtf.is_finite() && rtf > 0.0,
            "--rtf {rtf} is not a positive compute ratio"
        );
        Ok(Self { preset, rtf })
    }
}

/// Refuse flag pairs this build cannot serve, rather than ignoring one of them.
fn check_flags(request: &Request<'_>) -> Result<()> {
    anyhow::ensure!(
        !(request.live.is_some() && request.splice),
        "--splice is not wired into the live path; drop one of the two"
    );
    Ok(())
}

pub(crate) fn run(request: Request<'_>) -> Result<()> {
    check_flags(&request)?;
    let paths = setup::Paths::discover()?;
    let session = Session::new(&paths, request.splice)?;
    let started = Instant::now();

    // A full-band file and a cache hit never need the model, so they play on a machine
    // without the 6 GB install. Only a file that has to render asks for setup.
    let opened = match request.live {
        Some(live) => session.open_live(request.file, live.preset, &Probe::from_rtf(live.rtf)),
        None => session.open(request.file),
    };
    let mut playback = match opened {
        Ok(playback) => playback,
        Err(error) if needs_setup(&error) => {
            if request.smoke {
                anyhow::bail!("{error}. A headless run cannot install it.");
            }
            return ui::run(first_run::FirstRun::new(
                paths,
                request.file.to_path_buf(),
                request.splice,
            ))
            .map_err(|error| anyhow::anyhow!("{error}"));
        }
        Err(error) => return Err(error),
    };

    if !request.smoke {
        return ui::run(ui::PlayerApp::new(session, playback))
            .map_err(|error| anyhow::anyhow!("{error}"));
    }
    let result = if playback.live.is_some() {
        report_live(&mut playback, request.watch_secs)
    } else {
        report(&playback, started, request.watch_secs)
    };
    playback.cancel();
    result
}

/// Report what a headless live run saw.
///
/// The preset, the policy's frames and latency, the renderer's own measured ratio, and the
/// counters the panel prints: everything a CI machine or a person on the phone can check
/// without a window. It plays while it watches, so the report is of the audio path and not
/// of a simulation of it.
fn report_live(playback: &mut Playback, watch_secs: f64) -> Result<()> {
    let live = playback
        .live
        .as_ref()
        .context("a live playback reports live counters")?;
    let engine = &playback.engine;
    let policy = &live.policy;
    let stats = live.source.stats();
    let rate = f64::from(engine.device_rate);

    engine.controls.set_playing(true);
    let started = Instant::now();
    let mut peak_ahead = 0usize;
    while started.elapsed().as_secs_f64() < watch_secs {
        std::thread::sleep(std::time::Duration::from_millis(100));
        let position = engine.controls.position_frames();
        peak_ahead = peak_ahead.max(live.source.occupied(position));
    }

    let position = engine.controls.position_frames();
    println!(
        "preset    {} window = {} granules, {} frames",
        policy.preset.label(),
        policy.preset.granules(),
        policy.window_frames
    );
    println!(
        "buffer    {} frames of wet audio ({}), {} frames of dry behind it",
        policy.buffer_frames,
        secs(policy.buffer_frames),
        policy.buffer_frames + policy.window_frames + live::block_frames(policy)
    );
    println!(
        "latency   {:.2} s = window {:.2} + buffer {:.2} + device {:.2}",
        policy.total_latency_secs,
        policy.model_latency_secs,
        policy.buffer_latency_secs,
        policy.device_latency_secs
    );
    println!(
        "resident  {} bytes in the rings",
        live.source.resident_bytes()
    );
    println!(
        "device    \"{}\" {} Hz x {} ch",
        engine.device_name, engine.device_rate, engine.device_channels
    );
    match stats.measured_rtf() {
        Some(rtf) => println!(
            "measured  RTF {rtf:.2} over the last {} blocks",
            stats.blocks_rendered()
        ),
        None => println!(
            "measured  RTF not yet: {} blocks timed of 3",
            stats.blocks_rendered()
        ),
    }
    println!(
        "dropouts  {} over {} starved frames",
        stats.dropouts(),
        stats.starved_frames()
    );
    println!(
        "played    {position} frames ({:.2} s of audio) in {:.1} s of watching",
        position as f64 / rate,
        started.elapsed().as_secs_f64()
    );
    println!("peak      {peak_ahead} frames resident ahead of the playhead");
    let switch = if !engine.controls.has_enhanced() {
        "none".to_string()
    } else if engine.controls.is_enhanced() {
        "generated".to_string()
    } else {
        "original".to_string()
    };
    println!("switch    {switch}");
    if let Some(reason) = engine.controls.enhanced_blocked() {
        println!("blocked   {reason}");
    }
    let producer = playback.live.as_ref().map_or("none", |live| {
        if live.is_producer_finished() {
            "ended"
        } else {
            "running"
        }
    });
    println!("producer  {producer}");
    if let Some(live) = playback.live.as_mut() {
        live.reap()?;
    }
    if position == 0 {
        anyhow::bail!("the device never pulled audio");
    }
    Ok(())
}

/// A frame count as the seconds it is worth, for the lines that state a duration.
fn secs(frames: usize) -> String {
    format!("{:.2} s", frames as f64 / f64::from(SAMPLE_RATE))
}

/// Report what a headless run saw.
///
/// Playback starts once the frontier is two seconds ahead, so the first note is already
/// the generated signal and the frontier stays ahead from there (it runs at about 1.28x
/// realtime once the model is loaded). A patience budget catches the slow machine that
/// never gets there: it starts on the original and the automatic promotion does the rest.
fn report(playback: &Playback, started: Instant, watch_secs: f64) -> Result<()> {
    let engine = &playback.engine;
    let info = &playback.info;
    let rate = engine.device_rate as usize;

    let prebuffer = 2 * rate;
    let patience = 30.0;
    while started.elapsed().as_secs_f64() < patience {
        let ready = match &playback.frontier {
            Some(frontier) => frontier.ready_frames(),
            None => usize::MAX,
        };
        if ready >= prebuffer {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let first_audio = started.elapsed();
    engine.controls.set_playing(true);

    // Then watch for a few seconds: the toggle should become live, and promotion should
    // fire on its own.
    let mut live: Option<f64> = None;
    let mut promoted: Option<f64> = None;
    let mut peak_ready = 0usize;
    let watch = watch_secs;
    let until = started.elapsed().as_secs_f64() + watch;
    while started.elapsed().as_secs_f64() < until {
        std::thread::sleep(std::time::Duration::from_millis(100));
        let ready = match &playback.frontier {
            Some(frontier) => frontier.ready_frames(),
            None => (info.audio_secs * rate as f64) as usize,
        };
        peak_ready = peak_ready.max(ready);
        let position = engine.controls.position_frames();
        if live.is_none() && ready > position + rate {
            live = Some(started.elapsed().as_secs_f64());
        }
        if promoted.is_none() && engine.controls.promoted.load(Ordering::Relaxed) {
            promoted = Some(started.elapsed().as_secs_f64());
        }
    }

    let frames = engine.controls.position_frames();
    println!(
        "first     {:.2} s to first audio (prebuffer included)",
        first_audio.as_secs_f64()
    );
    println!(
        "device    \"{}\" {} Hz x {} ch",
        engine.device_name, engine.device_rate, engine.device_channels
    );
    println!(
        "played    {frames} frames ({:.2} s of audio) in {:.0} s of watching",
        frames as f64 / f64::from(engine.device_rate),
        watch
    );
    println!(
        "dropouts  {} frames of silence ({:.3} s) from segment misses",
        engine.controls.underrun_frames(),
        engine.controls.underrun_frames() as f64 / f64::from(engine.device_rate)
    );
    println!("model     {}", info.device);
    println!(
        "generated {:.1} s ready of {:.1} s",
        peak_ready as f64 / f64::from(engine.device_rate),
        info.audio_secs
    );
    if let Some(failure) = playback.status.as_ref().and_then(|status| status.failure()) {
        println!("render    failed: {failure}");
    }
    match live {
        Some(at) => println!("live      toggle offered at {at:.1} s"),
        None => println!("live      frontier never got a second ahead"),
    }
    match promoted {
        Some(at) => println!("promoted  automatically at {at:.1} s"),
        None => println!("promoted  not yet"),
    }
    println!(
        "switch    {}",
        if !engine.controls.has_enhanced() {
            "none (full-band source)"
        } else if engine.controls.is_enhanced() {
            "generated"
        } else {
            "original"
        }
    );
    if frames == 0 {
        anyhow::bail!("the device never pulled audio");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(live: Option<LiveOptions>, splice: bool) -> Request<'static> {
        Request {
            file: Path::new("song.mp3"),
            smoke: true,
            splice,
            live,
            watch_secs: 4.0,
        }
    }

    #[test]
    fn given_live_and_splice_when_checked_then_the_pair_is_refused() {
        let live = Some(LiveOptions::parse("8", 0.8).expect("a live request"));
        let error = check_flags(&request(live, true)).expect_err("the live path does not splice");
        assert!(format!("{error}").contains("--splice"), "{error}");
        assert!(check_flags(&request(live, false)).is_ok());
        assert!(check_flags(&request(None, true)).is_ok());
    }

    #[test]
    fn given_a_preset_and_a_ratio_when_parsed_then_the_request_carries_both() {
        assert_eq!(
            LiveOptions::parse("2", 0.62).expect("valid"),
            LiveOptions {
                preset: Preset::Two,
                rtf: 0.62
            }
        );
    }

    #[test]
    fn given_a_window_that_is_not_a_preset_when_parsed_then_it_names_the_four_choices() {
        let error = LiveOptions::parse("3", 0.8).expect_err("3 is not a preset");
        let message = format!("{error:#}");
        assert!(message.contains("1, 2, 4 and 8"), "{message}");
    }

    #[test]
    fn given_a_ratio_that_cannot_size_a_buffer_when_parsed_then_it_is_refused() {
        for rtf in [0.0, -1.0, f64::NAN] {
            let error =
                LiveOptions::parse("8", rtf).expect_err("a ratio must be a positive number");
            assert!(format!("{error:#}").contains("positive"), "{rtf}");
        }
    }
}
