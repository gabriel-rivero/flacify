//! `flacify sink`: capture one device, enhance it, and play it on another.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use flacify_core::{Probe, SAMPLE_RATE};

use crate::commands::play::LiveOptions;
use crate::device::{self, DeviceList, Direction, Resolution};
use crate::session::{Playback, Session};
use crate::setup;
use crate::sink::{self, CaptureState, SinkStats};
use crate::ui;

/// What `sink` was asked for, one field per flag group.
pub(crate) struct Request<'a> {
    pub from: &'a str,
    pub to: &'a str,
    /// The preset `--window` names and the ratio `--rtf` gives.
    pub live: LiveOptions,
    pub smoke: bool,
    /// Seconds a smoke run listens before it reports.
    pub watch_secs: f64,
}

/// The two names, decided without opening either device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Route {
    /// The capture device, which must be the one named.
    pub from: String,
    /// The output to ask for, or `None` for the default when the name matched nothing.
    pub to: Option<String>,
    /// Why the output is not the device that was named, when it is not.
    pub fallback: Option<String>,
}

pub(crate) fn run(request: Request<'_>) -> Result<()> {
    let devices = device::list().context("listing the devices")?;
    let route = resolve_route(&devices, request.from, request.to)?;
    let paths = setup::Paths::discover()?;
    let session = Session::new(&paths, false)?;
    let mut playback = session.open_sink(
        &route.from,
        route.to.as_deref(),
        route.fallback.clone(),
        request.live.preset,
        &Probe::from_rtf(request.live.rtf),
    )?;
    if !request.smoke {
        return ui::run(ui::PlayerApp::new(session, playback))
            .map_err(|error| anyhow::anyhow!("{error}"));
    }
    let result = report(&mut playback, request.watch_secs);
    playback.cancel();
    result
}

/// What the two names mean.
///
/// The input must be the device that was named: a capture that fell back to the built-in
/// microphone is not the stream the user routed, so a miss is an error. The output may fall
/// back to the default with the reason stated, because silence would be worse than the
/// wrong headphones.
pub(crate) fn resolve_route(devices: &DeviceList, from: &str, to: &str) -> Result<Route> {
    let default_input = devices
        .inputs
        .iter()
        .find(|info| info.is_default)
        .map(|info| info.name.as_str());
    let from = match device::resolve(Some(from), &devices.inputs, default_input) {
        Resolution::Selected(name) => name,
        // `resolve` only answers with the default when nothing was named, which for a
        // named input is an empty name and not a choice.
        Resolution::Default { reason } => anyhow::bail!(
            "no capture device was named ({reason}); name the loopback your player is \
             routed to, for example --from \"BlackHole 2ch\""
        ),
        Resolution::Missing {
            requested,
            available,
        } => anyhow::bail!(
            "{}",
            device::missing_message(Direction::Input, &requested, &available)
        ),
    };

    match device::resolve(Some(to), &devices.outputs, None) {
        Resolution::Selected(name) => Ok(Route {
            from,
            to: Some(name),
            fallback: None,
        }),
        Resolution::Missing {
            requested,
            available,
        } => {
            if requested.is_empty() {
                anyhow::bail!(
                    "no output device was named; pass --to \"<device>\" from `flacify devices`"
                );
            }
            let listed = if available.is_empty() {
                "(none)".to_string()
            } else {
                available.join(", ")
            };
            let reason = format!(
                "no output device matches \"{requested}\", so the default plays; the system \
                 lists: {listed}"
            );
            tracing::warn!(reason = %reason, "falling back to the default output");
            Ok(Route {
                from,
                to: None,
                fallback: Some(reason),
            })
        }
        Resolution::Default { reason } => Ok(Route {
            from,
            to: None,
            fallback: Some(reason),
        }),
    }
}

/// Report what a headless sink run saw.
///
/// The same block `play --live --smoke` prints, with the sink's own lines: the route, the
/// band the preprocessor was built for, the drift correction, and the resyncs it needed.
fn report(playback: &mut Playback, watch_secs: f64) -> Result<()> {
    let live = playback
        .live
        .as_ref()
        .context("a live playback reports live counters")?;
    let route = playback
        .sink
        .as_ref()
        .context("the sink reports the route it opened")?;
    let engine = &playback.engine;
    let policy = &live.policy;
    let stats = live.source.stats();
    let sink_stats = Arc::clone(&route.stats);
    let rate = f64::from(engine.device_rate);
    let policy_latency = policy.total_latency_secs;
    let capture_latency = route.latency_secs - policy_latency;

    engine.controls.set_playing(true);
    let started = Instant::now();
    let mut peak = 0usize;
    while started.elapsed().as_secs_f64() < watch_secs {
        std::thread::sleep(std::time::Duration::from_millis(100));
        let position = engine.controls.position_frames();
        peak = peak.max(live.source.occupied(position));
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
        policy.buffer_frames + policy.window_frames + crate::live::block_frames(policy)
    );
    println!(
        "latency   {:.2} s = window {:.2} + buffer {:.2} + device {:.2} + capture ring {:.2}",
        route.latency_secs,
        policy.model_latency_secs,
        policy.buffer_latency_secs,
        policy.device_latency_secs,
        capture_latency
    );
    println!("route     \"{}\" -> \"{}\"", route.from, route.to);
    if let Some(reason) = &route.fallback {
        println!("output    {reason}");
    }
    println!(
        "band      cutoff {} Hz, measured on the first second of capture",
        route.cutoff_hz
    );
    println!("capture   {}", capture_line(&sink_stats));
    if sink_stats.capture_state() == CaptureState::Silent {
        println!("silent    {}", sink::SILENT_HINT);
    }
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
    println!("resyncs   {}", sink_stats.resyncs());
    println!(
        "played    {position} frames ({:.2} s of audio) in {:.1} s of watching",
        position as f64 / rate,
        started.elapsed().as_secs_f64()
    );
    println!("peak      {peak} frames resident ahead of the playhead");
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
    let producer = if live.is_producer_finished() {
        "ended"
    } else {
        "running"
    };
    println!("producer  {producer}");
    if let Some(live) = playback.live.as_mut() {
        live.reap()?;
    }
    if position == 0 {
        anyhow::bail!("the device never pulled audio");
    }
    // A run that never carried signal is a refused microphone, not a quiet passage: a
    // report that says "0 dropouts" beside an empty capture is worse than an error.
    anyhow::ensure!(
        sink_stats.has_carried_signal(),
        "the capture never carried signal: {}",
        sink::SILENT_HINT
    );
    Ok(())
}

/// The capture line: the state first, because a silent capture makes every other number on
/// the report meaningless.
fn capture_line(stats: &SinkStats) -> String {
    let state = match stats.capture_state() {
        CaptureState::Signal => "signal",
        CaptureState::Silent => "silent",
    };
    format!(
        "{state} · fill {} of {} frames, correction {:+.0} ppm (peak {:.0}), {} dropped",
        stats.fill_frames(),
        sink::RING_FRAMES,
        stats.correction_ppm(),
        stats.peak_correction_ppm(),
        stats.dropped_frames()
    )
}

/// A frame count as the seconds it is worth, for the lines that state a duration.
fn secs(frames: usize) -> String {
    format!("{:.2} s", frames as f64 / f64::from(SAMPLE_RATE))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DeviceInfo;

    fn device(name: &str, is_default: bool) -> DeviceInfo {
        DeviceInfo {
            name: name.to_string(),
            channels: 2,
            default_sample_rate: 48_000,
            is_default,
        }
    }

    fn devices() -> DeviceList {
        DeviceList {
            inputs: vec![
                device("BlackHole 2ch", true),
                device("MacBook Pro Microphone", false),
            ],
            outputs: vec![
                device("External Headphones", true),
                device("BlackHole 2ch", false),
            ],
        }
    }

    #[test]
    fn given_both_names_when_resolved_then_the_route_is_the_named_pair() {
        let route = resolve_route(&devices(), "blackhole", "headphones").expect("a route");
        assert_eq!(
            route,
            Route {
                from: "BlackHole 2ch".to_string(),
                to: Some("External Headphones".to_string()),
                fallback: None,
            }
        );
    }

    #[test]
    fn given_a_capture_name_no_device_carries_when_resolved_then_it_is_an_error_not_a_fallback() {
        let error = resolve_route(&devices(), "Loopback 9ch", "External Headphones")
            .expect_err("a capture must be the device that was named");
        let message = format!("{error:#}");
        assert!(message.contains("no input device matches"), "{message}");
        assert!(message.contains("BlackHole 2ch"), "{message}");
    }

    #[test]
    fn given_a_capture_name_that_is_ambiguous_when_resolved_then_it_lists_the_candidates() {
        let mut list = devices();
        list.inputs.push(device("BlackHole 16ch", false));
        let error = resolve_route(&list, "blackhole", "External Headphones")
            .expect_err("an ambiguous capture is not a choice");
        let message = format!("{error:#}");
        assert!(message.contains("BlackHole 16ch"), "{message}");
    }

    #[test]
    fn given_an_empty_capture_name_when_resolved_then_the_default_is_refused() {
        let error = resolve_route(&devices(), "  ", "External Headphones")
            .expect_err("the input never falls back");
        let message = format!("{error:#}");
        assert!(message.contains("no capture device was named"), "{message}");
    }

    #[test]
    fn given_an_empty_output_name_when_resolved_then_it_asks_for_one_rather_than_falling_back() {
        let error =
            resolve_route(&devices(), "BlackHole 2ch", " ").expect_err("no output was named");
        let message = format!("{error:#}");
        assert!(message.contains("no output device was named"), "{message}");
    }

    #[test]
    fn given_an_output_name_no_device_carries_when_resolved_then_the_default_plays_with_why() {
        let route = resolve_route(&devices(), "BlackHole 2ch", "AirPods").expect("a route");
        assert_eq!(route.from, "BlackHole 2ch");
        assert_eq!(route.to, None, "the default answers the miss");
        let reason = route.fallback.expect("the fallback is stated");
        assert!(reason.contains("AirPods"), "{reason}");
        assert!(reason.contains("External Headphones"), "{reason}");
    }

    #[test]
    fn given_a_capture_that_never_carried_signal_when_reported_then_it_says_silent() {
        let stats = SinkStats::default();
        let line = capture_line(&stats);
        assert!(line.starts_with("signal"), "{line}");

        // Five seconds of zeros, as a refused microphone permission delivers.
        let silent = vec![0.0f32; crate::sink::SILENT_FRAMES * 2];
        let ring = crate::live::Ring::new(crate::sink::RING_FRAMES, 2);
        let head = std::sync::atomic::AtomicUsize::new(0);
        crate::sink::write_capture(&ring, &head, &stats, &silent);
        assert_eq!(stats.capture_state(), CaptureState::Silent);
        assert!(!stats.has_carried_signal());
        let line = capture_line(&stats);
        assert!(line.starts_with("silent"), "{line}");
        assert!(
            sink::SILENT_HINT.contains("Microphone"),
            "the hint must name the permission: {}",
            sink::SILENT_HINT
        );
    }
}
