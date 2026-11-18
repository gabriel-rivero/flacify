//! `flacify play`: the window, and the headless smoke run beside it.

use std::path::Path;
use std::time::Instant;

use anyhow::Result;

use crate::first_run;
use crate::session::{needs_setup, Playback, Session};
use crate::setup;
use crate::ui;

pub(crate) fn run(file: &Path, smoke: bool, splice: bool) -> Result<()> {
    let paths = setup::Paths::discover()?;
    let session = Session::new(&paths, splice)?;
    let started = Instant::now();

    // A full-band file and a cache hit never need the model, so they play on a machine
    // without the 6 GB install. Only a file that has to render asks for setup.
    let mut playback = match session.open(file) {
        Ok(playback) => playback,
        Err(error) if needs_setup(&error) => {
            if smoke {
                anyhow::bail!("{error}. A headless run cannot install it.");
            }
            return ui::run(first_run::FirstRun::new(paths, file.to_path_buf(), splice))
                .map_err(|error| anyhow::anyhow!("{error}"));
        }
        Err(error) => return Err(error),
    };

    if smoke {
        let result = report(&playback, started);
        playback.cancel();
        result
    } else {
        ui::run(ui::PlayerApp::new(session, playback)).map_err(|error| anyhow::anyhow!("{error}"))
    }
}

/// Report what a headless run saw.
///
/// Playback starts once the frontier is two seconds ahead, so the first note is already
/// the generated signal and the frontier stays ahead from there (it runs at about 1.28x
/// realtime once the model is loaded). A patience budget catches the slow machine that
/// never gets there: it starts on the original and the automatic promotion does the rest.
fn report(playback: &Playback, started: Instant) -> Result<()> {
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
    let watch = 4.0;
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
        if promoted.is_none()
            && engine
                .controls
                .promoted
                .load(std::sync::atomic::Ordering::Relaxed)
        {
            promoted = Some(started.elapsed().as_secs_f64());
        }
    }

    let frames = engine.controls.position_frames();
    println!(
        "first     {:.2} s to first audio (prebuffer included)",
        first_audio.as_secs_f64()
    );
    println!(
        "device    {} Hz x {} ch",
        engine.device_rate, engine.device_channels
    );
    println!(
        "played    {frames} frames ({:.2} s of audio) in {:.0} s of watching",
        frames as f64 / f64::from(engine.device_rate),
        watch
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
