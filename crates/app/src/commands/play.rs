//! `flacify play`: the window, and the headless smoke run beside it.

use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use flacify_core::{Preset, Probe, SAMPLE_RATE};

use crate::first_run;
use crate::live;
use crate::playlist;
use crate::session::{needs_setup, Playback, Session};
use crate::setup;
use crate::ui;
use crate::youtube;

/// What `play` was asked for, one field per flag group.
pub(crate) struct Request<'a> {
    /// `None` opens the window with no track, so one can be dropped onto it.
    pub file: Option<&'a Path>,
    pub smoke: bool,
    pub splice: bool,
    /// Expand a YouTube video into its radio mix.
    pub radio: bool,
    /// Play one video even when the URL names a playlist.
    pub single: bool,
    /// Present when `--live` was asked for.
    pub live: Option<LiveOptions>,
    /// Seconds a smoke run listens before it reports.
    pub watch_secs: f64,
    /// Where to start, when `--seek` was given.
    pub seek_secs: Option<f64>,
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
    anyhow::ensure!(
        !(request.radio && request.single),
        "--radio and --single ask for different queues; pick one"
    );
    if request.radio || request.single {
        let Some(file) = request.file else {
            anyhow::bail!("--radio and --single need a URL");
        };
        let is_url = youtube::parse(&file.to_string_lossy()).is_some();
        anyhow::ensure!(
            is_url,
            "{} needs a URL, not {}",
            if request.radio { "--radio" } else { "--single" },
            file.display()
        );
    }
    Ok(())
}

/// Open the window with nothing loaded, so a file, folder or playlist can be dropped on it.
pub(crate) fn run_gui() -> Result<()> {
    let paths = setup::Paths::discover()?;
    ui::run_empty(paths, false).map_err(|error| anyhow::anyhow!("{error}"))
}

/// The queue a command line names: local inputs through the queue module, URLs through
/// yt-dlp. A single file is a queue of one, so the window always speaks queue.
fn build_queue(
    input: &Path,
    downloads: &Path,
    radio: bool,
    single: bool,
) -> Result<playlist::Queue> {
    let text = input.to_string_lossy();
    match youtube::parse(&text) {
        Some(_) => playlist::Queue::from_url(&text, downloads, radio, single),
        None => playlist::Queue::from_local(&[input.to_path_buf()], downloads),
    }
}

pub(crate) fn run(request: Request<'_>) -> Result<()> {
    check_flags(&request)?;
    let Some(input) = request.file else {
        anyhow::ensure!(
            !(request.smoke || request.live.is_some()),
            "--smoke and --live need a file"
        );
        return run_gui();
    };
    let paths = setup::Paths::discover()?;
    let downloads = paths.downloads_dir();

    // The whole input becomes a queue before the window opens, so the first track's
    // fetch has the same cold-start cost a single `play <url>` always had.
    let queue = build_queue(input, &downloads, request.radio, request.single)?;
    if request.live.is_some() {
        anyhow::ensure!(
            queue.len() == 1,
            "--live plays one track, but this names {}; drop --live",
            queue.len()
        );
    }
    let entry = queue
        .current()
        .context("a queue always has a current entry")?;
    let first = playlist::resolve(entry, &downloads, None)?;
    if queue.len() > 1 {
        eprintln!("queue     {} ({})", queue.len(), queue.origin());
    }
    if entry.local_path().is_none() {
        eprintln!("source    {} ({})", first.label(), queue.indicator());
    }
    let file = first.path.as_path();
    let session = Session::new(&paths, request.splice)?;
    let started = Instant::now();

    // A full-band file and a cache hit never need the model, so they play on a machine
    // without the 6 GB install. Only a file that has to render asks for setup.
    let opened = match request.live {
        Some(live) => session.open_live(file, live.preset, &Probe::from_rtf(live.rtf)),
        None => session.open(file),
    };
    let mut playback = match opened {
        Ok(playback) => playback,
        Err(error) if needs_setup(&error) => {
            if request.smoke {
                anyhow::bail!("{error}. A headless run cannot install it.");
            }
            return ui::run(first_run::FirstRun::with_queue(
                paths,
                queue,
                request.splice,
            ))
            .map_err(|error| anyhow::anyhow!("{error}"));
        }
        Err(error) => return Err(error),
    };

    if !request.smoke {
        return ui::run(ui::PlayerApp::with_queue(session, playback, queue))
            .map_err(|error| anyhow::anyhow!("{error}"));
    }
    let result = if playback.live.is_some() {
        report_live(&mut playback, request.watch_secs)
    } else {
        report(&playback, started, request.watch_secs, request.seek_secs)
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
/// What a `--seek` cost, in the terms the acceptance run records.
struct SeekOutcome {
    target_secs: f64,
    waited_secs: f64,
    starved_frames: u64,
    landed: bool,
}

/// The line a seek prints, separated so the landed and starved cases are testable.
fn seek_line(seek: &SeekOutcome, rate: u32) -> String {
    format!(
        "seek      to {:.1} s {} after {:.2} s, starved {} frames ({:.3} s)",
        seek.target_secs,
        if seek.landed {
            "landed"
        } else {
            "did not land"
        },
        seek.waited_secs,
        seek.starved_frames,
        seek.starved_frames as f64 / f64::from(rate)
    )
}

/// Move the playhead and wait for audio to resume there.
///
/// A seek into an unrendered segment holds the playhead and plays silence until the
/// prefetch catches up, so waiting for the position to reach the target is exactly the
/// measurement: how long the listener hears nothing, and how much of it was counted.
fn seek_to(playback: &Playback, seconds: f64, rate: u32) -> SeekOutcome {
    let target = (seconds.max(0.0) * f64::from(rate)) as usize;
    let starved_before = playback.engine.controls.underrun_frames();
    let started = Instant::now();
    playback.engine.controls.seek_frames(target);
    let deadline = Duration::from_secs(10);
    // `seek_frames` moves the counter immediately; the audio is there only once the
    // playhead has advanced past the target, which needs a fill that found a segment.
    let mut landed = false;
    while started.elapsed() < deadline {
        if playback.engine.controls.position_frames() > target {
            landed = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    SeekOutcome {
        target_secs: seconds.max(0.0),
        waited_secs: started.elapsed().as_secs_f64(),
        starved_frames: playback
            .engine
            .controls
            .underrun_frames()
            .saturating_sub(starved_before),
        landed,
    }
}

fn report(
    playback: &Playback,
    started: Instant,
    watch_secs: f64,
    seek_secs: Option<f64>,
) -> Result<()> {
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

    // After playback has started, so the playhead can actually move: a seek before the
    // first fill only moves a counter and measures nothing.
    let seek = seek_secs.map(|seconds| seek_to(playback, seconds, engine.device_rate));

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
    if let Some(seek) = seek {
        println!("{}", seek_line(&seek, engine.device_rate));
    }
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
            file: Some(Path::new("song.mp3")),
            smoke: true,
            splice,
            radio: false,
            single: false,
            live,
            watch_secs: 4.0,
            seek_secs: None,
        }
    }

    #[test]
    fn given_a_seek_that_landed_when_reported_then_it_states_the_wait_and_the_silence() {
        let seek = SeekOutcome {
            target_secs: 3300.0,
            waited_secs: 0.184,
            starved_frames: 4_800,
            landed: true,
        };
        let line = seek_line(&seek, 48_000);
        assert!(line.contains("to 3300.0 s landed after 0.18 s"), "{line}");
        assert!(line.contains("starved 4800 frames (0.100 s)"), "{line}");
    }

    #[test]
    fn given_a_seek_that_never_landed_when_reported_then_it_says_so() {
        let seek = SeekOutcome {
            target_secs: 55.0,
            waited_secs: 10.0,
            starved_frames: 480_000,
            landed: false,
        };
        let line = seek_line(&seek, 48_000);
        assert!(
            line.starts_with("seek      to 55.0 s did not land"),
            "{line}"
        );
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

    #[test]
    fn given_radio_and_single_when_checked_then_the_pair_is_refused() {
        let mut request = request(None, false);
        request.radio = true;
        request.single = true;
        let error = check_flags(&request).expect_err("they ask for different queues");
        assert!(format!("{error}").contains("pick one"), "{error}");
    }

    #[test]
    fn given_radio_or_single_with_a_path_when_checked_then_a_url_is_required() {
        for (radio, single) in [(true, false), (false, true)] {
            let mut request = request(None, false);
            request.radio = radio;
            request.single = single;
            let error = check_flags(&request).expect_err("a URL is required");
            assert!(format!("{error}").contains("needs a URL"), "{error}");
        }
    }

    #[test]
    fn given_radio_with_a_url_when_checked_then_it_is_accepted() {
        let mut request = request(None, false);
        request.radio = true;
        request.file = Some(Path::new("https://youtu.be/dQw4w9WgXcQ"));
        assert!(check_flags(&request).is_ok());
    }
}
