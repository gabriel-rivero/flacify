//! S2: play a tone into a loopback device and read it back, to measure clock drift.
//!
//! The sink has two clocks — the device it captures from and the device it plays to — and
//! nothing in `cpal` keeps them together. This binary answers the two questions before the
//! sink is trusted with them:
//!
//! 1. Can a terminal-launched process open the loopback for capture at all, or does macOS
//!    refuse it with a TCC prompt the app cannot satisfy?
//! 2. How far apart do the two clocks run over ten minutes, and does the captured stream
//!    drop blocks while it happens?
//!
//! It is a probe, not the sink: no model, no rings, no correction. The tone is generated in
//! the output callback and a phase discontinuity in the input counts as a dropped block.
//!
//!     cargo run --release --bin sink_probe -- --from "BlackHole 2ch" --to "BlackHole 2ch" --seconds 600

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

/// 440 Hz at -26 dBFS: audible enough to see, quiet enough to leave running.
const TONE_HZ: f32 = 440.0;
const AMPLITUDE: f32 = 0.05;

macro_rules! out {
    ($buf:expr, $($arg:tt)*) => {{
        let line = format!($($arg)*);
        println!("{line}");
        $buf.push(line);
    }};
}

fn main() -> Result<(), String> {
    let mut report: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    let mut from = None;
    let mut to = None;
    let mut seconds = 600.0f64;
    let mut log: Option<std::path::PathBuf> = None;
    let mut silent = false;
    let mut play: Option<std::path::PathBuf> = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--from" => from = Some(args.next().ok_or("--from wants a device name")?),
            "--to" => to = Some(args.next().ok_or("--to wants a device name")?),
            "--log" => log = Some(args.next().ok_or("--log wants a path")?.into()),
            // Two different devices drift against each other whether or not anything is
            // playing, so a clock measurement does not have to make a sound.
            "--silent" => silent = true,
            "--play" => play = Some(args.next().ok_or("--play wants a wav path")?.into()),
            "--seconds" => {
                seconds = args
                    .next()
                    .ok_or("--seconds wants a number")?
                    .parse()
                    .map_err(|error| format!("--seconds: {error}"))?
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    let to = to.ok_or("--to is required")?;
    let is_feeder = play.is_some();

    let host = cpal::default_host();
    // A feeder only plays, so it needs no capture device and no microphone permission.
    let from = match (from, is_feeder) {
        (Some(from), _) => from,
        (None, true) => to.clone(),
        (None, false) => return Err("--from is required".to_string()),
    };
    let input_device = if is_feeder {
        None
    } else {
        Some(find(&host, &from, true)?)
    };
    let output_device = find(&host, &to, false)?;
    out!(report, "from      \"{from}\" (capture)");
    out!(report, "to        \"{to}\" (playback)");

    let input_config = match &input_device {
        Some(device) => Some(
            device
                .default_input_config()
                .map_err(|error| format!("\"{from}\" input configuration: {error}"))?,
        ),
        None => None,
    };
    let output_config = output_device
        .default_output_config()
        .map_err(|error| format!("\"{to}\" output configuration: {error}"))?;
    if output_config.channels() != 2 {
        return Err(format!(
            "playback must be stereo; \"{to}\" is {} channels",
            output_config.channels()
        ));
    }
    // The sink needs stereo, but the probe also reads a mono microphone: it exists to
    // answer "may this process capture at all", and a mono device answers that fine.
    let rate = match &input_config {
        Some(config) => config.sample_rate(),
        None => output_config.sample_rate(),
    };
    let in_channels = input_config
        .as_ref()
        .map(|config| config.channels() as usize)
        .unwrap_or(2);
    match &input_config {
        Some(config) => out!(
            report,
            "format    in {} Hz x{} ch ({:?}), out {} Hz x{} ch",
            config.sample_rate(),
            config.channels(),
            config.sample_format(),
            output_config.sample_rate(),
            output_config.channels()
        ),
        None => out!(
            report,
            "format    playing only, out {} Hz x{} ch",
            output_config.sample_rate(),
            output_config.channels()
        ),
    }
    if output_config.sample_rate() != rate {
        return Err("the two devices report different rates; set both to 48 kHz".to_string());
    }

    // A feeder for the sink: ffmpeg's AudioToolbox output will not start on this macOS,
    // and something has to put music into the loopback.
    let audio: Option<Arc<Vec<f32>>> = match &play {
        Some(path) => Some(Arc::new(read_wav(path)?)),
        None => None,
    };
    if let Some(samples) = &audio {
        out!(
            report,
            "playing   {} frames from {}",
            samples.len() / 2,
            play.as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_default()
        );
    }
    let cursor = Arc::new(AtomicU64::new(0));

    let running = Arc::new(AtomicBool::new(true));
    let in_frames = Arc::new(AtomicU64::new(0));
    let out_frames = Arc::new(AtomicU64::new(0));
    let drops = Arc::new(AtomicU64::new(0));
    let peak = Arc::new(AtomicU64::new(0));

    let input_stream: Option<cpal::Stream> = match (&input_device, &input_config) {
        (None, _) => None,
        (Some(device), Some(config)) => Some({
            let in_frames = Arc::clone(&in_frames);
            let drops = Arc::clone(&drops);
            let peak = Arc::clone(&peak);
            let mut last: Option<f32> = None;
            // One period at 440 Hz is 109 samples, so neighbouring samples differ by less
            // than this. A dropped block shows up as a jump several times larger.
            let jump = (2.0 * std::f32::consts::PI * TONE_HZ / rate as f32).sin() * AMPLITUDE * 4.0;
            device
                .build_input_stream(
                    &config.clone().into(),
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        let mut block_peak: f32 = 0.0;
                        for frame in data.chunks(in_channels) {
                            let sample = frame[0];
                            if let Some(previous) = last {
                                if (sample - previous).abs() > jump {
                                    drops.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                            last = Some(sample);
                            block_peak = block_peak.max(sample.abs());
                        }
                        peak.fetch_max(block_peak.to_bits() as u64, Ordering::Relaxed);
                        in_frames.fetch_add((data.len() / in_channels) as u64, Ordering::Relaxed);
                    },
                    |error| eprintln!("input stream error: {error}"),
                    None,
                )
                .map_err(|error| {
                    format!(
                        "opening \"{from}\" for capture: {error}\n\
                     If this is a permissions failure, macOS refused the microphone to this \
                     process; a signed .app needs NSMicrophoneUsageDescription."
                    )
                })?
        }),
        (Some(_), None) => None,
    };

    let output_stream = {
        let out_frames = Arc::clone(&out_frames);
        let running = Arc::clone(&running);
        let audio = audio.clone();
        let cursor = Arc::clone(&cursor);
        let mut phase = 0.0f32;
        let step = 2.0 * std::f32::consts::PI * TONE_HZ / rate as f32;
        output_device
            .build_output_stream(
                &output_config.clone().into(),
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    if !running.load(Ordering::Relaxed) || silent {
                        data.fill(0.0);
                        out_frames.fetch_add((data.len() / 2) as u64, Ordering::Relaxed);
                        return;
                    }
                    match &audio {
                        Some(samples) => {
                            let mut at = cursor.load(Ordering::Relaxed) as usize;
                            for frame in data.chunks_mut(2) {
                                if at + 1 < samples.len() {
                                    frame[0] = samples[at];
                                    frame[1] = samples[at + 1];
                                    at += 2;
                                } else {
                                    frame.fill(0.0);
                                }
                            }
                            cursor.store(at as u64, Ordering::Relaxed);
                        }
                        None => {
                            for frame in data.chunks_mut(2) {
                                let sample = phase.sin() * AMPLITUDE;
                                frame.fill(sample);
                                phase += step;
                                if phase > std::f32::consts::TAU {
                                    phase -= std::f32::consts::TAU;
                                }
                            }
                        }
                    }
                    out_frames.fetch_add((data.len() / 2) as u64, Ordering::Relaxed);
                },
                |error| eprintln!("output stream error: {error}"),
                None,
            )
            .map_err(|error| format!("opening \"{to}\" for playback: {error}"))?
    };

    if let Some(stream) = &input_stream {
        stream
            .play()
            .map_err(|error| format!("starting capture: {error}"))?;
    }
    output_stream
        .play()
        .map_err(|error| format!("starting playback: {error}"))?;

    // The first seconds are dominated by stream start latency, not by clock rate, so the
    // drift is measured between two later samples rather than from the start.
    let started = Instant::now();
    let settle = Duration::from_secs_f64(seconds.min(5.0));
    let mut baseline: Option<(Duration, u64, u64)> = None;
    let mut last_report = Duration::ZERO;
    while started.elapsed() < Duration::from_secs_f64(seconds) {
        std::thread::sleep(Duration::from_millis(200));
        let elapsed = started.elapsed();
        if baseline.is_none() && elapsed >= settle {
            baseline = Some((
                elapsed,
                in_frames.load(Ordering::Relaxed),
                out_frames.load(Ordering::Relaxed),
            ));
        }
        if elapsed.as_secs() >= last_report.as_secs() + 60 {
            last_report = elapsed;
            out!(
                report,
                "  {:>4} s  in {}  out {}",
                elapsed.as_secs(),
                in_frames.load(Ordering::Relaxed),
                out_frames.load(Ordering::Relaxed)
            );
        }
    }
    running.store(false, Ordering::Relaxed);
    let elapsed = started.elapsed();
    drop(input_stream);
    drop(output_stream);

    let captured = in_frames.load(Ordering::Relaxed);
    let played = out_frames.load(Ordering::Relaxed);
    let expected = elapsed.as_secs_f64() * f64::from(rate);
    let peak_value = f32::from_bits(peak.load(Ordering::Relaxed) as u32);
    let (base_elapsed, base_in, base_out) = baseline.unwrap_or((Duration::ZERO, 0, 0));
    let window = elapsed.saturating_sub(base_elapsed);
    out!(report, "");
    out!(report, "elapsed   {:.3} s", elapsed.as_secs_f64());
    out!(
        report,
        "played    {played} frames ({:.0} expected)",
        expected
    );
    if is_feeder {
        out!(
            report,
            "fed       {played} frames into \"{to}\" in {:.1} s",
            elapsed.as_secs_f64()
        );
        if let Some(path) = log {
            std::fs::write(&path, report.join("\n") + "\n")
                .map_err(|error| format!("writing {}: {error}", path.display()))?;
        }
        return Ok(());
    }
    out!(
        report,
        "captured  {captured} frames ({:.0} expected), {:.1} dBFS peak{}",
        expected,
        20.0 * (peak_value.max(1e-9)).log10(),
        if peak_value < 1e-6 {
            "  <- digital silence: the app that launched this has no Microphone permission, \
             or nothing is routed into the loopback. System Settings > Privacy & Security > \
             Microphone."
        } else {
            ""
        }
    );
    let in_ppm = ppm(captured - base_in, window, rate);
    let out_ppm = ppm(played - base_out, window, rate);
    out!(report,
        "drift     capture {in_ppm:+.1} ppm, playback {out_ppm:+.1} ppm, relative {:+.1} ppm          over {:.0} s",
        in_ppm - out_ppm,
        window.as_secs_f64()
    );
    out!(
        report,
        "drops     {} discontinuities in the captured stream",
        drops.load(Ordering::Relaxed)
    );
    if let Some(path) = log {
        std::fs::write(&path, report.join("\n") + "\n")
            .map_err(|error| format!("writing {}: {error}", path.display()))?;
    }
    Ok(())
}

/// One 48 kHz stereo WAV, read whole: a feeder holds the whole file anyway.
fn read_wav(path: &std::path::Path) -> Result<Vec<f32>, String> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|error| format!("opening {}: {error}", path.display()))?;
    let spec = reader.spec();
    if spec.sample_rate != 48_000 || spec.channels != 2 {
        return Err(format!(
            "{} is {} Hz x{} ch; the loopback is 48 kHz stereo",
            path.display(),
            spec.sample_rate,
            spec.channels
        ));
    }
    let samples: Result<Vec<f32>, _> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|sample| sample.map(|value| f32::from(value) / 32_768.0))
            .collect(),
        hound::SampleFormat::Float => reader.samples::<f32>().collect(),
    };
    samples.map_err(|error| format!("reading {}: {error}", path.display()))
}

fn ppm(frames: u64, elapsed: Duration, rate: u32) -> f64 {
    let expected = elapsed.as_secs_f64() * f64::from(rate);
    if expected <= 0.0 {
        return 0.0;
    }
    (frames as f64 / expected - 1.0) * 1e6
}

fn find(host: &cpal::Host, name: &str, input: bool) -> Result<cpal::Device, String> {
    let devices: Vec<cpal::Device> = if input {
        host.input_devices()
            .map_err(|error| format!("listing input devices: {error}"))?
            .collect()
    } else {
        host.output_devices()
            .map_err(|error| format!("listing output devices: {error}"))?
            .collect()
    };
    let matches: Vec<(cpal::Device, String)> = devices
        .into_iter()
        .filter_map(|device| {
            device
                .description()
                .ok()
                .map(|description| (device, description.name().to_string()))
        })
        .collect();
    if let Some((device, _)) = matches
        .iter()
        .find(|(_, device_name)| device_name.eq_ignore_ascii_case(name))
    {
        return Ok(device.clone());
    }
    let available: Vec<&str> = matches.iter().map(|(_, name)| name.as_str()).collect();
    Err(format!(
        "no device named \"{name}\"; available: {}",
        available.join(", ")
    ))
}
