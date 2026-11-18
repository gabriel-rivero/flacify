//! What to do with the measured speed of the machine.
//!
//! The probe answers one question: how many seconds of audio does one second of
//! wall clock produce? Everything else is arithmetic, and the arithmetic is the
//! reason "buffer enough for the song" does not work: above real time a buffer can
//! only cover part of a track, and filling that part costs exactly as long as the
//! part it covers.
//!
//! The live path asks the same probe a second question, which is what a window
//! costs in latency, memory and headroom while the playhead is already running.

use crate::{GRANULE, SAMPLE_RATE};

/// Sustained timings from the hardware probe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Probe {
    /// Median seconds of compute per second of audio.
    pub rtf_p50: f64,
    /// 95th percentile, which is what dropouts are made of.
    pub rtf_p95: f64,
    /// Last 60 s of a continuous run, i.e. after the GPU has had time to throttle.
    pub rtf_sustained: f64,
}

impl Probe {
    /// A probe built from a single measured number, for tests and `--rtf` overrides.
    pub fn from_rtf(rtf: f64) -> Self {
        Self {
            rtf_p50: rtf,
            rtf_p95: rtf,
            rtf_sustained: rtf,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Comfortably faster than real time (sustained RTF <= 0.7).
    Realtime,
    /// Faster on average, but jitter will sometimes starve the buffer.
    Marginal,
    /// Slower than real time: pre-render the part of the track that playback needs.
    PreRender,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub mode: Mode,
    /// Seconds of audio to render before starting playback.
    pub prebuffer_secs: f64,
    /// Estimated seconds from pressing play to the first sample.
    pub time_to_first_audio_secs: f64,
    /// For `PreRender`, how long until playback can start for a track of this length.
    pub estimated_wait_secs: Option<f64>,
    pub warning: Option<Warning>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Warning {
    pub headline: String,
    pub detail: String,
    pub actions: [Action; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Pre-render what is needed, then play.
    Continue,
    /// Play the original now, enhance later.
    PlayOriginal,
    /// Render to cache without playing.
    EnhanceOnly,
}

/// Decide what playback should do, given a measured probe.
///
/// `window_secs` is the model window in seconds; `track_secs` is the length of the
/// file about to play, when it is known.
pub fn plan(probe: &Probe, window_secs: f64, track_secs: Option<f64>) -> Plan {
    let rtf = probe.rtf_sustained.max(1e-6);
    let windows_of_prebuffer = if rtf <= 0.4 {
        1.0
    } else if rtf <= 0.7 {
        2.0
    } else if rtf <= 1.0 {
        3.0
    } else {
        // Above real time, the prebuffer is whatever fraction of the track is needed
        // to survive to the end.
        let track = track_secs.unwrap_or(180.0).max(0.0);
        return pre_render(probe, track, window_secs);
    };

    let prebuffer_secs = windows_of_prebuffer * window_secs;
    let ttfa = prebuffer_secs * rtf;
    let mode = if rtf <= 0.7 {
        Mode::Realtime
    } else {
        Mode::Marginal
    };

    let warning = if mode == Mode::Marginal {
        Some(Warning {
            headline: "This machine is close to real time".to_string(),
            detail: format!(
                "Rendering at {:.2}x real time. Playback starts after about {:.1} s \
                 and may fall back to the original if the machine slows down.",
                rtf, ttfa
            ),
            actions: [Action::Continue, Action::PlayOriginal, Action::EnhanceOnly],
        })
    } else {
        None
    };

    Plan {
        mode,
        prebuffer_secs,
        time_to_first_audio_secs: ttfa,
        estimated_wait_secs: None,
        warning,
    }
}

fn pre_render(probe: &Probe, track_secs: f64, window_secs: f64) -> Plan {
    let rtf = probe.rtf_sustained.max(1e-6);
    // A buffer of B seconds of audio covers B * rtf seconds of rendering, and
    // playback drains it at (1 - 1/rtf) per second. Solving for the whole track gives
    // B = T(r-1)/r. Filling that buffer costs B * rtf = T(r-1) seconds of wall clock,
    // which is a full render minus exactly one track length: the saving over rendering
    // everything up front never exceeds the track's own duration.
    let prebuffer_secs = (track_secs * (rtf - 1.0) / rtf).max(window_secs);
    let wait = prebuffer_secs * rtf;
    Plan {
        mode: Mode::PreRender,
        prebuffer_secs,
        time_to_first_audio_secs: wait,
        estimated_wait_secs: Some(wait),
        warning: Some(Warning {
            headline: "This machine is slower than real time".to_string(),
            detail: format!(
                "Rendering at {:.2}x real time. A {} track needs {:.1} s of audio \
                 rendered before playback can start, which takes about {:.0} s at this \
                 speed; after that it plays through without falling behind.",
                rtf,
                duration_label(track_secs),
                prebuffer_secs,
                wait
            ),
            actions: [Action::Continue, Action::PlayOriginal, Action::EnhanceOnly],
        }),
    }
}

fn duration_label(secs: f64) -> String {
    if secs >= 60.0 {
        format!("{:.0} minute", (secs / 60.0).round())
    } else {
        format!("{:.0} second", secs.round())
    }
}

/// The model windows the live path offers, counted in granules.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Preset {
    One,
    Two,
    Four,
    Eight,
}

impl Preset {
    /// Every preset, shortest window first.
    pub const ALL: [Preset; 4] = [Preset::One, Preset::Two, Preset::Four, Preset::Eight];

    /// How many granules of 0.64 s the window holds.
    pub fn granules(self) -> usize {
        match self {
            Preset::One => 1,
            Preset::Two => 2,
            Preset::Four => 4,
            Preset::Eight => 8,
        }
    }

    /// The model window in frames.
    pub fn frames(self) -> usize {
        self.granules() * GRANULE
    }

    /// The window as the UI states it.
    pub fn label(self) -> &'static str {
        match self {
            Preset::One => "0.64 s",
            Preset::Two => "1.28 s",
            Preset::Four => "2.56 s",
            Preset::Eight => "5.12 s",
        }
    }

    /// Read a preset from a `--window` argument.
    pub fn parse(text: &str) -> Option<Preset> {
        match text {
            "1" => Some(Preset::One),
            "2" => Some(Preset::Two),
            "4" => Some(Preset::Four),
            "8" => Some(Preset::Eight),
            _ => None,
        }
    }
}

/// Headroom over the p95 render ratio, so one call slower than p95 still leaves audio.
const BUFFER_SAFETY: f64 = 1.25;
const CHANNELS: usize = 2;
const BYTES_PER_SAMPLE: usize = 4;

/// What one window of live playback costs on this machine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LivePolicy {
    pub preset: Preset,
    pub window_frames: usize,
    /// Frames of buffer in the wet ring; the dry ring also carries the alignment delay.
    pub buffer_frames: usize,
    /// The window the model must see before it can emit, which floors the latency.
    pub model_latency_secs: f64,
    pub buffer_latency_secs: f64,
    pub device_latency_secs: f64,
    /// Model plus buffer plus device: the number the UI prints.
    pub total_latency_secs: f64,
    /// Wet buffer, dry buffer and the dry side's alignment delay. The model's input
    /// window is not counted.
    pub resident_bytes: usize,
    /// Sustained RTF below 1.0.
    pub is_real_time: bool,
    /// A single worst-case call cannot drain more than the buffer holds.
    pub is_worst_case_real_time: bool,
}

/// Decide what one preset costs on this machine.
///
/// `device_buffer_frames` is the device's own buffer, which the UI adds to the latency
/// it prints. `probe` must be measured at this preset's window: fixed per-call cost is
/// a larger share of a short window.
pub fn live_policy(preset: Preset, probe: &Probe, device_buffer_frames: usize) -> LivePolicy {
    let window_frames = preset.frames();
    let window_secs = frames_to_secs(window_frames);
    let rtf_p95 = probe.rtf_p95.max(1e-6);
    let rtf_sustained = probe.rtf_sustained.max(1e-6);

    // One call drains the ring by a window's worth times the render ratio, so the
    // buffer covers the p95 call and not the median one. Floored at one window so a
    // slow first call cannot starve playback immediately.
    let worst_case_frames = window_frames as f64 * rtf_p95 * BUFFER_SAFETY;
    let buffer_frames = (worst_case_frames.ceil() as usize).max(window_frames);
    let buffer_secs = frames_to_secs(buffer_frames);
    let device_secs = frames_to_secs(device_buffer_frames);
    // The dry side is delayed by the model window so it lines up with the wet side.
    let dry_frames = buffer_frames + window_frames;

    LivePolicy {
        preset,
        window_frames,
        buffer_frames,
        model_latency_secs: window_secs,
        buffer_latency_secs: buffer_secs,
        device_latency_secs: device_secs,
        total_latency_secs: window_secs + buffer_secs + device_secs,
        resident_bytes: (buffer_frames + dry_frames) * CHANNELS * BYTES_PER_SAMPLE,
        is_real_time: rtf_sustained < 1.0,
        is_worst_case_real_time: rtf_p95 < 1.0,
    }
}

fn frames_to_secs(frames: usize) -> f64 {
    frames as f64 / SAMPLE_RATE as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_machine_gets_no_warning() {
        let plan = plan(&Probe::from_rtf(0.3), 5.12, Some(180.0));
        assert_eq!(plan.mode, Mode::Realtime);
        assert!(plan.warning.is_none());
        assert!((plan.prebuffer_secs - 5.12).abs() < 1e-9);
        assert!((plan.time_to_first_audio_secs - 1.536).abs() < 1e-9);
    }

    #[test]
    fn given_comfortable_hardware_when_planned_then_one_window_is_enough() {
        let plan = plan(&Probe::from_rtf(0.4), 5.12, Some(180.0));
        assert_eq!(plan.mode, Mode::Realtime);
        assert!((plan.prebuffer_secs - 5.12).abs() < 1e-9);
    }

    #[test]
    fn given_marginal_hardware_when_planned_then_it_warns_and_deepens_the_buffer() {
        let plan = plan(&Probe::from_rtf(0.9), 5.12, Some(180.0));
        assert_eq!(plan.mode, Mode::Marginal);
        assert!((plan.prebuffer_secs - 15.36).abs() < 1e-9);
        let warning = plan.warning.expect("marginal hardware must warn");
        assert!(warning.detail.contains("0.90"));
    }

    #[test]
    fn given_slow_hardware_when_planned_then_the_wait_matches_the_arithmetic() {
        // 180 s at 2x real time needs 90 s of audio rendered before playback,
        // which takes 180 s to produce. Half the track for half the wait.
        let plan = plan(&Probe::from_rtf(2.0), 5.12, Some(180.0));
        assert_eq!(plan.mode, Mode::PreRender);
        assert!((plan.prebuffer_secs - 90.0).abs() < 1e-9);
        assert!((plan.estimated_wait_secs.unwrap() - 180.0).abs() < 1e-9);
    }

    #[test]
    fn given_a_very_slow_machine_when_planned_then_the_wait_is_one_render_minus_a_track() {
        // At 8.7x (the measured CPU RTF) the prebuffer has to cover 7.7/8.7 of the
        // track, 159.3 s of the 180, and filling it takes 159.3 * 8.7 = 1386 s. That
        // is a full render (180 * 8.7 = 1566 s) minus exactly one track length, so
        // there is almost nothing left to save.
        let plan = plan(&Probe::from_rtf(8.7), 5.12, Some(180.0));
        let prebuffer = plan.prebuffer_secs;
        let wait = plan.estimated_wait_secs.unwrap();
        assert!(
            (prebuffer - 180.0 * 7.7 / 8.7).abs() < 1e-9,
            "prebuffer {prebuffer}"
        );
        assert!((wait - 180.0 * 7.7).abs() < 1e-9, "wait {wait}");
        assert!((wait - (180.0 * 8.7 - 180.0)).abs() < 1e-9);
    }

    #[test]
    fn given_slow_hardware_when_warned_then_it_separates_audio_from_wall_clock() {
        // The warning used to print the wait as if it were the amount of audio. At
        // 8.7x those are 159.3 s of audio and 1386 s of waiting, so confusing them
        // overstates what must be rendered by a factor of the RTF.
        let plan = plan(&Probe::from_rtf(8.7), 5.12, Some(180.0));
        let detail = &plan.warning.unwrap().detail;
        assert!(detail.contains("159.3 s of audio"), "{detail}");
        assert!(detail.contains("1386 s"), "{detail}");
    }

    #[test]
    fn given_no_track_length_when_slow_then_it_assumes_a_three_minute_track() {
        let plan = plan(&Probe::from_rtf(1.5), 5.12, None);
        assert_eq!(plan.mode, Mode::PreRender);
        assert!(plan.estimated_wait_secs.unwrap() > 0.0);
    }

    /// What the M5 measures on the file path: faster than real time, with some jitter.
    const M5_PROBE: Probe = Probe {
        rtf_p50: 0.55,
        rtf_p95: 0.70,
        rtf_sustained: 0.78,
    };
    const DEVICE_FRAMES: usize = 512;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn given_a_fixed_probe_when_tabled_then_every_number_is_pinned() {
        // p95 below 0.8 keeps the safety factor under the one-window floor, so the
        // buffer and the window are equal at every preset.
        assert_eq!(
            Preset::ALL,
            [Preset::One, Preset::Two, Preset::Four, Preset::Eight]
        );
        let expected = [
            (Preset::One, 30_720usize, 30_720usize, 0.64, 737_280usize),
            (Preset::Two, 61_440, 61_440, 1.28, 1_474_560),
            (Preset::Four, 122_880, 122_880, 2.56, 2_949_120),
            (Preset::Eight, 245_760, 245_760, 5.12, 5_898_240),
        ];
        let device_secs = DEVICE_FRAMES as f64 / SAMPLE_RATE as f64;
        println!(
            "{:<8} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>10} {:>4} {:>5}",
            "preset",
            "window",
            "buffer",
            "model",
            "buffer",
            "device",
            "total",
            "resident",
            "rt",
            "worst"
        );
        for (preset, window_frames, buffer_frames, window_secs, resident_bytes) in expected {
            let policy = live_policy(preset, &M5_PROBE, DEVICE_FRAMES);
            let real_time = if policy.is_real_time { "yes" } else { "no" };
            let worst_case = if policy.is_worst_case_real_time {
                "yes"
            } else {
                "no"
            };
            println!(
                "{:<8} {:>7} {:>7} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>10} {:>4} {:>5}",
                policy.preset.label(),
                policy.window_frames,
                policy.buffer_frames,
                policy.model_latency_secs,
                policy.buffer_latency_secs,
                policy.device_latency_secs,
                policy.total_latency_secs,
                policy.resident_bytes,
                real_time,
                worst_case,
            );
            assert_eq!(policy.window_frames, window_frames);
            assert_eq!(policy.buffer_frames, buffer_frames);
            assert_close(policy.model_latency_secs, window_secs);
            assert_close(policy.buffer_latency_secs, window_secs);
            assert_close(policy.device_latency_secs, device_secs);
            assert_close(policy.total_latency_secs, window_secs * 2.0 + device_secs);
            assert_eq!(policy.resident_bytes, resident_bytes);
            assert!(policy.is_real_time);
            assert!(policy.is_worst_case_real_time);
        }
    }

    #[test]
    fn given_the_four_presets_when_planned_then_latency_rises_with_the_window() {
        let mut previous = 0.0;
        for preset in Preset::ALL {
            let policy = live_policy(preset, &M5_PROBE, DEVICE_FRAMES);
            assert!(
                policy.total_latency_secs > previous,
                "{} did not raise the latency: {}",
                preset.label(),
                policy.total_latency_secs
            );
            previous = policy.total_latency_secs;
        }
    }

    #[test]
    fn given_a_cpu_class_probe_when_planned_then_neither_check_passes() {
        // 3x real time cannot sustain playback, and a p95 call drains three windows.
        // The 1.25 safety factor makes the buffer 3.75 windows, which covers the
        // drain but does not make playback keep up.
        let policy = live_policy(Preset::Eight, &Probe::from_rtf(3.0), DEVICE_FRAMES);
        assert!(!policy.is_real_time);
        assert!(!policy.is_worst_case_real_time);
        assert_close(policy.buffer_latency_secs, 5.12 * 3.0 * 1.25);
    }

    #[test]
    fn given_a_fast_probe_when_planned_then_both_checks_pass() {
        let policy = live_policy(Preset::Four, &Probe::from_rtf(0.5), DEVICE_FRAMES);
        assert!(policy.is_real_time);
        assert!(policy.is_worst_case_real_time);
    }

    #[test]
    fn given_a_jittery_probe_when_planned_then_only_the_sustained_check_passes() {
        // The average keeps up but a p95 call drains 1.4 windows, so the UI has to arm
        // passthrough even though the sustained ratio says real time.
        let probe = Probe {
            rtf_p50: 0.70,
            rtf_p95: 1.40,
            rtf_sustained: 0.90,
        };
        let policy = live_policy(Preset::Two, &probe, DEVICE_FRAMES);
        assert!(policy.is_real_time);
        assert!(!policy.is_worst_case_real_time);
    }

    #[test]
    fn given_eight_granules_when_planned_then_resident_bytes_is_both_rings_and_the_delay() {
        let policy = live_policy(Preset::Eight, &Probe::from_rtf(0.6), 0);
        assert_eq!(
            policy.resident_bytes,
            (2 * policy.buffer_frames + policy.window_frames) * 2 * 4
        );
        assert_eq!(policy.resident_bytes, 5_898_240);
    }

    #[test]
    fn given_a_window_argument_when_parsed_then_only_the_four_presets_are_accepted() {
        for preset in Preset::ALL {
            assert_eq!(Preset::parse(&preset.granules().to_string()), Some(preset));
        }
        for text in ["3", "0", "eight", "", " 8", "6", "1.28"] {
            assert_eq!(Preset::parse(text), None, "{text:?}");
        }
    }

    #[test]
    fn given_a_slow_but_real_time_probe_when_planned_then_the_buffer_rounds_up() {
        // 30_720 * 0.813 * 1.25 = 31_219.2, so the buffer is 31_220 frames and not the
        // one-window floor. This is where the ceil, not the floor, sets the depth.
        let probe = Probe {
            rtf_p50: 0.70,
            rtf_p95: 0.813,
            rtf_sustained: 0.78,
        };
        let policy = live_policy(Preset::One, &probe, 0);
        assert_eq!(policy.buffer_frames, 31_220);
        assert!(policy.buffer_frames > policy.window_frames);
    }

    #[test]
    fn given_exactly_real_time_when_planned_then_neither_check_passes() {
        let exactly = live_policy(Preset::Two, &Probe::from_rtf(1.0), 0);
        assert!(!exactly.is_real_time);
        assert!(!exactly.is_worst_case_real_time);
        let under = live_policy(Preset::Two, &Probe::from_rtf(0.999), 0);
        assert!(under.is_real_time);
        assert!(under.is_worst_case_real_time);
    }

    #[test]
    fn given_a_larger_device_buffer_when_planned_then_the_total_rises_by_that_much() {
        let without = live_policy(Preset::Two, &M5_PROBE, 0);
        let with = live_policy(Preset::Two, &M5_PROBE, 4_800);
        assert_close(with.device_latency_secs, 0.1);
        assert_close(with.total_latency_secs - without.total_latency_secs, 0.1);
    }

    #[test]
    fn given_each_preset_when_measured_then_the_window_is_its_granules() {
        let expected = [
            (Preset::One, 1),
            (Preset::Two, 2),
            (Preset::Four, 4),
            (Preset::Eight, 8),
        ];
        for (preset, granules) in expected {
            assert_eq!(preset.granules(), granules);
            assert_eq!(preset.frames(), granules * 30_720);
            assert_eq!(preset.frames(), granules * GRANULE);
            let policy = live_policy(preset, &M5_PROBE, DEVICE_FRAMES);
            assert_eq!(policy.window_frames, granules * GRANULE);
            assert_eq!(
                preset.label(),
                format!("{:.2} s", policy.model_latency_secs)
            );
        }
    }

    #[test]
    fn given_a_zero_or_nan_probe_when_planned_then_the_numbers_stay_finite() {
        for probe in [Probe::from_rtf(0.0), Probe::from_rtf(f64::NAN)] {
            let policy = live_policy(Preset::One, &probe, DEVICE_FRAMES);
            assert!(policy.total_latency_secs.is_finite());
            assert_eq!(policy.buffer_frames, GRANULE);
            assert!(policy.is_real_time);
            assert!(policy.is_worst_case_real_time);
        }
    }
}
