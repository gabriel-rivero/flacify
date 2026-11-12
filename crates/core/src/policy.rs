//! What to do with the measured speed of the machine.
//!
//! The probe answers one question: how many seconds of audio does one second of
//! wall clock produce? Everything else is arithmetic, and the arithmetic is the
//! reason "buffer enough for the song" does not work: above real time a buffer can
//! only cover part of a track, and filling that part costs exactly as long as the
//! part it covers.

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
}
