//! The audio device side: list what the machine has, open one by name, and pull audio
//! through it.
//!
//! The output callback does exactly one thing per buffer: ask the mixer for interleaved f32,
//! convert to whatever the device wants, and get out of the way. No allocation, no lock, no
//! logging: every buffer it needs was allocated before the stream started, and the UI talks
//! to it through atomics only. A stream that fails is rebuilt once by a supervisor thread,
//! which is the only code that replaces the stream after it starts.
//!
//! If the output does not run at 48 kHz (many do not), both buffers are resampled once here,
//! before playback, so the callback never has to. Segment playback has no buffers to resample,
//! so it is offered only on a 48 kHz stereo device. Capture never resamples: the sink's clock
//! correction is a later decision (K1).

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::Sample as _;
use flacify_core::dsp::resample::Resampler;

use crate::frontier::Frontier;
use crate::live;
use crate::player::{Controls, Mixer};
use crate::segments::Reader;

/// Bigger than any buffer a device asked us for in practice; the callback clamps to it
/// rather than allocating, and counts the buffers that do not fit.
pub const MAX_FRAMES: usize = 16_384;
/// The device buffer playback asks for first. The live policy adds it to the latency it
/// prints; S3 replaces that estimate with the device's own figure.
pub const PREFERRED_BUFFER_FRAMES: u32 = 1024;

/// A device and the configuration it defaults to, as much as a human needs to pick it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    pub name: String,
    pub channels: usize,
    pub default_sample_rate: u32,
    pub is_default: bool,
}

/// Everything the host offers, split the way the app consumes it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeviceList {
    pub inputs: Vec<DeviceInfo>,
    pub outputs: Vec<DeviceInfo>,
}

/// What a requested name means, given what the host actually has.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Resolution {
    /// The one device that answers to the request.
    Selected(String),
    /// Nothing was requested, so the system default is the answer.
    Default { reason: String },
    /// No device, or more than one, answers to the request. `available` holds the
    /// candidates when the request was ambiguous and the whole list when it was not;
    /// `requested` is empty when nothing was asked for and there was no default either.
    Missing {
        requested: String,
        available: Vec<String>,
    },
}

/// Which side of the device the caller is asking about.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Direction {
    Input,
    Output,
}

impl Direction {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Direction::Input => "input",
            Direction::Output => "output",
        }
    }
}

/// What a name means for the opener: a device, or the reason there is none.
enum Choice {
    Device(cpal::Device),
    Missing(String),
}

/// The device a name means, decided without touching hardware.
///
/// An exact name wins, then a unique case-insensitive substring (so `BlackHole 2ch` finds
/// `BlackHole 2ch (Virtual)`), and anything else is `Missing` with what it could have meant.
pub fn resolve(
    requested: Option<&str>,
    available: &[DeviceInfo],
    default: Option<&str>,
) -> Resolution {
    let Some(requested) = requested.map(str::trim).filter(|name| !name.is_empty()) else {
        return match default {
            Some(name) => Resolution::Default {
                reason: format!("no device was named, so the default \"{name}\" is used"),
            },
            None => Resolution::Missing {
                requested: String::new(),
                available: device_names(available),
            },
        };
    };
    let wanted = requested.to_lowercase();

    // An exact name first: a device whose name is a prefix of another must still mean itself.
    if let Some(found) = available
        .iter()
        .find(|info| info.name.to_lowercase() == wanted)
    {
        return Resolution::Selected(found.name.clone());
    }

    let candidates: Vec<&DeviceInfo> = available
        .iter()
        .filter(|info| info.name.to_lowercase().contains(&wanted))
        .collect();
    match candidates.as_slice() {
        [only] => Resolution::Selected(only.name.clone()),
        [] => Resolution::Missing {
            requested: requested.to_string(),
            available: device_names(available),
        },
        many => Resolution::Missing {
            requested: requested.to_string(),
            available: many.iter().map(|info| info.name.clone()).collect(),
        },
    }
}

fn device_names(available: &[DeviceInfo]) -> Vec<String> {
    available.iter().map(|info| info.name.clone()).collect()
}

/// The error for a name that matched nothing, or matched more than one thing. Public
/// inside the crate so a command that must not fall back can reuse the wording.
pub(crate) fn missing_message(
    direction: Direction,
    requested: &str,
    available: &[String],
) -> String {
    let kind = direction.label();
    let listed = if available.is_empty() {
        "(none)".to_string()
    } else {
        available.join(", ")
    };
    if requested.is_empty() {
        format!("no default {kind} device is available; the system lists: {listed}")
    } else {
        format!("no {kind} device matches \"{requested}\"; the system lists: {listed}")
    }
}

/// `1 channel`, `2 channels`: a count the sentence can agree with.
fn channels_label(channels: usize) -> String {
    if channels == 1 {
        "1 channel".to_string()
    } else {
        format!("{channels} channels")
    }
}

/// Stream errors and over-long device buffers, for the whole process.
static STREAM_ERRORS: AtomicU64 = AtomicU64::new(0);
static BUFFER_OVERFLOWS: AtomicU64 = AtomicU64::new(0);

/// What the audio path has had to report since this process started.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub stream_errors: u64,
    pub buffer_overflows: u64,
}

/// The counters as they stand now.
pub fn counters() -> Counters {
    Counters {
        stream_errors: STREAM_ERRORS.load(Ordering::Relaxed),
        buffer_overflows: BUFFER_OVERFLOWS.load(Ordering::Relaxed),
    }
}

/// One line for the UI when the stream has not been clean, or `None` when it has.
pub fn stream_warning(name: &str, counters: Counters) -> Option<String> {
    if counters.stream_errors == 0 && counters.buffer_overflows == 0 {
        return None;
    }
    Some(format!(
        "device \"{name}\": {} stream errors, {} buffers over {MAX_FRAMES} frames zero-filled",
        counters.stream_errors, counters.buffer_overflows
    ))
}

/// Every device the host offers in both directions.
pub fn list() -> Result<DeviceList> {
    let host = cpal::default_host();
    Ok(DeviceList {
        inputs: enumerate(&host, Direction::Input)?
            .into_iter()
            .map(|(info, _)| info)
            .collect(),
        outputs: enumerate(&host, Direction::Output)?
            .into_iter()
            .map(|(info, _)| info)
            .collect(),
    })
}

/// The inventory as text: one line per device, with what it defaults to and which one is
/// the default. `flacify devices` and the diagnose report both print this.
pub fn describe(list: &DeviceList) -> String {
    let width = list
        .inputs
        .iter()
        .chain(list.outputs.iter())
        .map(|info| info.name.len() + 2)
        .max()
        .unwrap_or(0);
    let mut text = String::new();
    for (heading, devices) in [("inputs", &list.inputs), ("outputs", &list.outputs)] {
        let _ = writeln!(text, "{heading}");
        if devices.is_empty() {
            let _ = writeln!(text, "  (none)");
            continue;
        }
        for info in devices {
            let name = format!("\"{}\"", info.name);
            let suffix = if info.is_default { "  (default)" } else { "" };
            let _ = writeln!(
                text,
                "  {name:<width$}  {} Hz x {}{suffix}",
                info.default_sample_rate,
                channels_label(info.channels)
            );
        }
    }
    text
}

/// The device a name means, or why nothing matched. The caller decides what a miss means,
/// because playback and capture want opposite answers.
fn choose(host: &cpal::Host, direction: Direction, requested: Option<&str>) -> Result<Choice> {
    let found = enumerate(host, direction)?;
    let available: Vec<DeviceInfo> = found.iter().map(|(info, _)| info.clone()).collect();
    // Read once: the name `resolve` needs and the handle the answer needs are one query.
    let default = default_device(host, direction);
    let default_name = default.as_ref().and_then(device_name);
    match resolve(requested, &available, default_name.as_deref()) {
        Resolution::Selected(name) => found
            .into_iter()
            .find(|(info, _)| info.name.eq_ignore_ascii_case(&name))
            .map(|(_, device)| Choice::Device(device))
            .with_context(|| format!("opening \"{name}\"")),
        Resolution::Default { reason } => {
            tracing::debug!(%reason, "opening the default device");
            let device =
                default.with_context(|| format!("no default {} device", direction.label()))?;
            Ok(Choice::Device(device))
        }
        Resolution::Missing {
            requested,
            available,
        } => Ok(Choice::Missing(missing_message(
            direction, &requested, &available,
        ))),
    }
}

/// The name cpal reports for a device, or `None` when the host will not say.
fn device_name(device: &cpal::Device) -> Option<String> {
    device
        .description()
        .ok()
        .map(|description| description.name().to_string())
}

fn default_device(host: &cpal::Host, direction: Direction) -> Option<cpal::Device> {
    match direction {
        Direction::Input => host.default_input_device(),
        Direction::Output => host.default_output_device(),
    }
}

/// Every device in `direction`, with the configuration each defaults to and the device
/// handle the opener needs.
fn enumerate(host: &cpal::Host, direction: Direction) -> Result<Vec<(DeviceInfo, cpal::Device)>> {
    let default_name = default_device(host, direction).and_then(|device| device_name(&device));
    let devices: Vec<cpal::Device> = match direction {
        Direction::Input => host
            .input_devices()
            .context("listing the input devices")?
            .collect(),
        Direction::Output => host
            .output_devices()
            .context("listing the output devices")?
            .collect(),
    };
    let mut listed = Vec::with_capacity(devices.len());
    for device in devices {
        let name = match device_name(&device) {
            Some(name) => name,
            None => {
                tracing::warn!("a device would not say its name; skipping it");
                continue;
            }
        };
        let format = match direction {
            Direction::Input => device
                .default_input_config()
                .map(|config| (config.channels(), config.sample_rate())),
            Direction::Output => device
                .default_output_config()
                .map(|config| (config.channels(), config.sample_rate())),
        };
        let (channels, rate) = match format {
            Ok(format) => format,
            Err(error) => {
                tracing::warn!(%error, device = %name, "no default configuration; skipping it");
                continue;
            }
        };
        let is_default = default_name.as_deref() == Some(name.as_str());
        let info = DeviceInfo {
            name,
            channels: channels as usize,
            default_sample_rate: rate,
            is_default,
        };
        listed.push((info, device));
    }
    Ok(listed)
}

pub struct AudioEngine {
    /// The live stream. [`Drop`] takes it so a restart cannot outlive the window.
    stream: Arc<Mutex<Option<cpal::Stream>>>,
    pub controls: Arc<Controls>,
    /// The device that was actually opened, so the UI can say which one it is.
    pub device_name: String,
    pub device_rate: u32,
    pub device_channels: usize,
    /// The signals themselves, shared with the mixer (an `Arc` clone, not a copy), so
    /// the window can draw a spectrum of what is playing. Empty on the segmented path,
    /// which holds no whole-track buffer.
    pub display_original: Arc<[f32]>,
    pub display_generated: Option<Arc<[f32]>>,
}

impl AudioEngine {
    /// `original` and `enhanced` are interleaved stereo at 48 kHz, same length.
    pub fn start(original: &[f32], enhanced: Option<&[f32]>) -> Result<Self> {
        let output = Output::open_named(None)?;
        let prepared = prepare(original, output.rate, output.channels)?;
        let generated = match enhanced {
            Some(buffer) => Some(prepare(buffer, output.rate, output.channels)?),
            None => None,
        };
        let (mixer, controls) = Mixer::new(prepared, generated, output.channels, output.rate);
        output.start(mixer, controls)
    }

    /// Play now, generate behind: the frontier must be 48 kHz stereo, because a
    /// progressive render cannot be resampled per chunk without seams.
    pub fn start_progressive(original: &[f32], frontier: Arc<Frontier>) -> Result<Self> {
        let output = Output::open_named(None)?;
        if output.rate != flacify_core::SAMPLE_RATE || output.channels != 2 {
            anyhow::bail!(
                "{}",
                streaming_refusal(
                    "progressive playback",
                    &output.name,
                    output.rate,
                    output.channels
                )
            );
        }
        let prepared = prepare(original, output.rate, output.channels)?;
        let (mixer, controls) =
            Mixer::progressive(prepared, frontier, output.channels, output.rate);
        output.start(mixer, controls)
    }

    /// Play a complete cache entry from its segments: nothing is resampled here, so the
    /// device has to be 48 kHz stereo.
    pub fn start_segmented(reader: Arc<Reader>) -> Result<Self> {
        let output = Output::open_named(None)?;
        if output.rate != flacify_core::SAMPLE_RATE || output.channels != 2 {
            anyhow::bail!(
                "{}",
                streaming_refusal(
                    "segment playback",
                    &output.name,
                    output.rate,
                    output.channels
                )
            );
        }
        let (mixer, controls) = Mixer::segmented(reader, output.rate);
        output.start(mixer, controls)
    }

    /// Play a live feed: the rings are the source, so nothing is resampled per chunk and
    /// the device must take 48 kHz stereo.
    pub fn start_live(source: Arc<live::Live>) -> Result<Self> {
        Self::start_live_on(source, None)
    }

    /// Play a live feed on the named output, or on the default when the name resolves to
    /// nothing: the sink names its output, and a miss plays through the default rather than
    /// silent. The caller states the reason it fell back.
    pub fn start_live_on(source: Arc<live::Live>, requested: Option<&str>) -> Result<Self> {
        let output = Output::open_named(requested)?;
        if output.rate != flacify_core::SAMPLE_RATE || output.channels != 2 {
            anyhow::bail!(
                "{}",
                streaming_refusal("live playback", &output.name, output.rate, output.channels)
            );
        }
        let (mixer, controls) = Mixer::live(source, output.rate);
        output.start(mixer, controls)
    }
}

/// Why a path that cannot resample per chunk refuses this device.
fn streaming_refusal(what: &str, name: &str, rate: u32, channels: usize) -> String {
    format!(
        "{what} needs a 48 kHz stereo device; \"{name}\" is {rate} Hz with {}",
        channels_label(channels)
    )
}

impl Drop for AudioEngine {
    fn drop(&mut self) {
        // Taking the stream drops it here, so closing the window stops playback now rather
        // than when the restart thread notices the channel closed.
        if let Ok(mut held) = self.stream.lock() {
            held.take();
        }
    }
}

/// The device we found, before deciding what to feed it.
struct Output {
    device: cpal::Device,
    config: cpal::StreamConfig,
    format: cpal::SampleFormat,
    rate: u32,
    channels: usize,
    name: String,
}

impl Output {
    /// The named device, or the default when nothing is named. A name that resolves to
    /// nothing falls back to the default and says why; a machine with no default is an error.
    fn open_named(requested: Option<&str>) -> Result<Self> {
        let host = cpal::default_host();
        let device = match choose(&host, Direction::Output, requested)? {
            Choice::Device(device) => device,
            Choice::Missing(problem) => {
                // Playback through the default beats silence, but never silently: the
                // reason names what was asked for and what this machine has.
                tracing::warn!(reason = %problem, "falling back to the default device");
                default_device(&host, Direction::Output).context("no default output device")?
            }
        };
        let name = device_name(&device).unwrap_or_else(|| "(unnamed)".to_string());
        let default = device
            .default_output_config()
            .with_context(|| format!("asking \"{name}\" for its default configuration"))?;

        // Prefer a stereo 48 kHz configuration when the device offers one: that is what
        // lets playback start before the render finishes. A 96 kHz default is common and
        // would otherwise force the slow path for no product reason.
        let preferred = device
            .supported_output_configs()
            .ok()
            .and_then(|mut configs| {
                configs
                    .find(|range| {
                        range.channels() == 2
                            && range.min_sample_rate() <= flacify_core::SAMPLE_RATE
                            && range.max_sample_rate() >= flacify_core::SAMPLE_RATE
                    })
                    .map(|range| range.with_sample_rate(flacify_core::SAMPLE_RATE))
            });
        let supported = preferred.unwrap_or(default);
        let rate = supported.sample_rate();
        let channels = supported.channels() as usize;
        let format = supported.sample_format();
        Ok(Self {
            device,
            config: supported.into(),
            format,
            rate,
            channels,
            name,
        })
    }

    fn start(self, mixer: Mixer, controls: Arc<Controls>) -> Result<AudioEngine> {
        let display_original = mixer.display_original().unwrap_or_default();
        let display_generated = mixer.display_generated();
        let (stream, failures, config) = open_with_fallback(
            &self.device,
            self.config.clone(),
            self.format,
            &mixer,
            self.channels,
        )?;
        stream.play().context("starting playback")?;

        let slot = Arc::new(Mutex::new(Some(stream)));
        {
            let slot = Arc::clone(&slot);
            let name = self.name.clone();
            let device = self.device.clone();
            let format = self.format;
            let channels = self.channels;
            // The mixer is a handle on the same signals and controls, so a rebuilt stream
            // resumes at the playhead rather than at the start.
            let spare = mixer.clone();
            std::thread::spawn(move || {
                let mut rebuild = move || {
                    let (sender, receiver) = mpsc::channel();
                    let stream = open(&device, &config, format, spare.clone(), channels, sender)?;
                    Ok((stream, receiver))
                };
                supervise(failures, slot, name, &mut rebuild);
            });
        }

        Ok(AudioEngine {
            stream: slot,
            controls,
            device_name: self.name,
            device_rate: self.rate,
            device_channels: self.channels,
            display_original,
            display_generated,
        })
    }
}

/// What to do about a stream that failed while it was running.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Restart {
    /// The first failure is usually a device change; rebuild once and keep the playhead.
    Retry,
    /// A second failure is the device, not a glitch. Report it and stop.
    Stop,
}

/// One retry, then a report: a device that fails twice does not come back by asking again.
pub fn restart_after(failures: u64) -> Restart {
    if failures <= 1 {
        Restart::Retry
    } else {
        Restart::Stop
    }
}

/// Rebuild a failed stream once, and say so when that does not work either.
fn supervise(
    mut failures: Receiver<String>,
    slot: Arc<Mutex<Option<cpal::Stream>>>,
    name: String,
    rebuild: &mut dyn FnMut() -> Result<(cpal::Stream, Receiver<String>)>,
) {
    let mut seen = 0u64;
    while let Ok(error) = failures.recv() {
        // The window closed: `Drop` took the stream, so there is nothing to rebuild.
        let is_running = slot.lock().map(|held| held.is_some()).unwrap_or(false);
        if !is_running {
            break;
        }
        seen += 1;
        match restart_after(seen) {
            Restart::Retry => {
                tracing::warn!(
                    %error,
                    device = %name,
                    "the output stream failed; rebuilding it once"
                );
                let (stream, receiver) = match rebuild() {
                    Ok(rebuilt) => rebuilt,
                    Err(error) => {
                        tracing::error!(
                            %error,
                            device = %name,
                            "the output stream could not be rebuilt"
                        );
                        break;
                    }
                };
                if let Err(error) = stream.play() {
                    tracing::error!(
                        %error,
                        device = %name,
                        "the rebuilt output stream will not start"
                    );
                    break;
                }
                let previous = match slot.lock() {
                    Ok(mut held) => held.replace(stream),
                    Err(_) => break,
                };
                // Dropped outside the lock: stopping a device waits for its callback, and
                // that wait does not belong under a lock the owner also takes.
                drop(previous);
                failures = receiver;
            }
            Restart::Stop => {
                tracing::error!(
                    %error,
                    device = %name,
                    "the output stream failed twice; stopping"
                );
                break;
            }
        }
    }
}

/// Whether progressive playback can work here: chunks are 48 kHz stereo, and
/// resampling them one second at a time would put a seam at every boundary.
pub fn progressive_supported() -> bool {
    match Output::open_named(None) {
        Ok(output) => output.rate == flacify_core::SAMPLE_RATE && output.channels == 2,
        Err(_) => false,
    }
}

/// A capture device and the configuration it defaults to.
///
/// `build_input_stream` starts capture; the caller keeps the stream and drops it to stop.
/// The sink is the caller: capture is never resampled, and a device that cannot do 48 kHz
/// stereo is refused here rather than converted twice.
pub struct Input {
    device: cpal::Device,
    config: cpal::StreamConfig,
    format: cpal::SampleFormat,
    rate: u32,
    channels: usize,
    name: String,
}

#[allow(dead_code)]
impl Input {
    /// The named device, or the default. Live mode feeds 48 kHz stereo into the rings, so a
    /// device that cannot do that is refused here with its actual format rather than
    /// resampled: the sink's correction is a rate trim, not a format converter.
    pub fn open_named(requested: Option<&str>) -> Result<Self> {
        let host = cpal::default_host();
        let device = match choose(&host, Direction::Input, requested)? {
            Choice::Device(device) => device,
            // Recording the built-in microphone because the loopback was misspelled is not
            // a fallback anyone wants, so a capture name that resolves to nothing stops here.
            Choice::Missing(problem) => return Err(anyhow::anyhow!(problem)),
        };
        let name = device_name(&device).unwrap_or_else(|| "(unnamed)".to_string());
        let supported = device
            .default_input_config()
            .with_context(|| format!("asking \"{name}\" for its default input configuration"))?;
        let rate = supported.sample_rate();
        let channels = supported.channels() as usize;
        if rate != flacify_core::SAMPLE_RATE || channels != 2 {
            anyhow::bail!(
                "live capture needs a 48 kHz stereo input; \"{name}\" is {rate} Hz with {}",
                channels_label(channels)
            );
        }
        Ok(Self {
            format: supported.sample_format(),
            config: supported.into(),
            device,
            rate,
            channels,
            name,
        })
    }

    /// Start capture at the device's default configuration: interleaved f32 blocks reach
    /// `on_block` until the returned stream is dropped, and `on_error` is called when the
    /// stream fails. The conversion buffer is allocated here, once: the callback neither
    /// allocates nor grows it, and a block longer than [`MAX_FRAMES`] is truncated and
    /// counted instead of reallocated.
    pub fn build_input_stream<F, E>(&self, on_block: F, on_error: E) -> Result<cpal::Stream>
    where
        F: FnMut(&[f32]) + Send + 'static,
        E: FnMut() + Send + 'static,
    {
        match self.format {
            cpal::SampleFormat::I8 => self.build::<i8, F, E>(on_block, on_error),
            cpal::SampleFormat::I16 => self.build::<i16, F, E>(on_block, on_error),
            cpal::SampleFormat::I32 => self.build::<i32, F, E>(on_block, on_error),
            cpal::SampleFormat::I64 => self.build::<i64, F, E>(on_block, on_error),
            cpal::SampleFormat::U8 => self.build::<u8, F, E>(on_block, on_error),
            cpal::SampleFormat::U16 => self.build::<u16, F, E>(on_block, on_error),
            cpal::SampleFormat::U32 => self.build::<u32, F, E>(on_block, on_error),
            cpal::SampleFormat::U64 => self.build::<u64, F, E>(on_block, on_error),
            cpal::SampleFormat::F32 => self.build::<f32, F, E>(on_block, on_error),
            cpal::SampleFormat::F64 => self.build::<f64, F, E>(on_block, on_error),
            other => anyhow::bail!("the input device wants {other}, which is not wired up yet"),
        }
    }

    fn build<T, F, E>(&self, mut on_block: F, mut on_error: E) -> Result<cpal::Stream>
    where
        T: cpal::SizedSample,
        f32: cpal::FromSample<T>,
        F: FnMut(&[f32]) + Send + 'static,
        E: FnMut() + Send + 'static,
    {
        let mut block = vec![0.0f32; MAX_FRAMES * self.channels];
        let stream = self.device.build_input_stream(
            &self.config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                let len = data.len().min(block.len());
                for (sample, converted) in data[..len].iter().zip(block[..len].iter_mut()) {
                    *converted = f32::from_sample(*sample);
                }
                if len < data.len() {
                    BUFFER_OVERFLOWS.fetch_add(1, Ordering::Relaxed);
                }
                on_block(&block[..len]);
            },
            move |error| {
                STREAM_ERRORS.fetch_add(1, Ordering::Relaxed);
                tracing::error!(%error, "input stream error");
                on_error();
            },
            None,
        )?;
        stream.play().context("starting capture")?;
        Ok(stream)
    }

    pub fn sample_rate(&self) -> u32 {
        self.rate
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

/// The small buffer first, then whatever the device wants.
///
/// A fixed buffer size is a request, not a guarantee: WASAPI in shared mode and some ALSA
/// configurations refuse it. Returns the stream, its error channel, and the configuration
/// that worked, so a rebuild asks for exactly the same thing.
fn open_with_fallback(
    device: &cpal::Device,
    mut config: cpal::StreamConfig,
    format: cpal::SampleFormat,
    mixer: &Mixer,
    channels: usize,
) -> Result<(cpal::Stream, Receiver<String>, cpal::StreamConfig)> {
    let mut first_error = None;
    for buffer_size in [
        cpal::BufferSize::Fixed(PREFERRED_BUFFER_FRAMES),
        cpal::BufferSize::Default,
    ] {
        config.buffer_size = buffer_size;
        let (failures, receiver) = mpsc::channel();
        match open(device, &config, format, mixer.clone(), channels, failures) {
            Ok(stream) => return Ok((stream, receiver, config.clone())),
            Err(error) => {
                tracing::warn!(%error, "output stream refused a {:?} buffer", buffer_size);
                first_error = first_error.or(Some(error));
            }
        }
    }
    let error = first_error.unwrap_or_else(|| anyhow::anyhow!("no output stream"));
    Err(anyhow::anyhow!("{error:#}"))
}

/// One attempt with one sample format.
fn open(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    format: cpal::SampleFormat,
    mixer: Mixer,
    channels: usize,
    failures: Sender<String>,
) -> Result<cpal::Stream> {
    match format {
        cpal::SampleFormat::I8 => build::<i8>(device, config, mixer, channels, failures),
        cpal::SampleFormat::I16 => build::<i16>(device, config, mixer, channels, failures),
        cpal::SampleFormat::I32 => build::<i32>(device, config, mixer, channels, failures),
        cpal::SampleFormat::I64 => build::<i64>(device, config, mixer, channels, failures),
        cpal::SampleFormat::U8 => build::<u8>(device, config, mixer, channels, failures),
        cpal::SampleFormat::U16 => build::<u16>(device, config, mixer, channels, failures),
        cpal::SampleFormat::U32 => build::<u32>(device, config, mixer, channels, failures),
        cpal::SampleFormat::U64 => build::<u64>(device, config, mixer, channels, failures),
        cpal::SampleFormat::F32 => build::<f32>(device, config, mixer, channels, failures),
        cpal::SampleFormat::F64 => build::<f64>(device, config, mixer, channels, failures),
        other => anyhow::bail!("the output device wants {other}, which is not wired up yet"),
    }
}

fn build<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut mixer: Mixer,
    channels: usize,
    failures: Sender<String>,
) -> Result<cpal::Stream>
where
    T: cpal::SizedSample + cpal::FromSample<f32>,
{
    let mut scratch = vec![0.0f32; MAX_FRAMES * channels.max(1)];
    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            fill_device_buffer(&mut mixer, &mut scratch, data);
        },
        move |error| {
            STREAM_ERRORS.fetch_add(1, Ordering::Relaxed);
            // `send` allocates, which is why this is the error callback and not the data
            // one. A send that fails means a supervisor has already given up.
            let _ = failures.send(error.to_string());
        },
        None,
    )?;
    Ok(stream)
}

/// One device buffer from the mixer.
///
/// A device that asks for more than [`MAX_FRAMES`] gets silence for the excess, and the
/// buffer is counted so the truncation is visible instead of guessed at.
fn fill_device_buffer<T>(mixer: &mut Mixer, scratch: &mut [f32], data: &mut [T])
where
    T: cpal::SizedSample + cpal::FromSample<f32>,
{
    let len = data.len().min(scratch.len());
    mixer.fill(&mut scratch[..len]);
    for (out, sample) in data.iter_mut().zip(scratch[..len].iter()) {
        *out = T::from_sample(*sample);
    }
    if len < data.len() {
        BUFFER_OVERFLOWS.fetch_add(1, Ordering::Relaxed);
        for out in data.iter_mut().skip(len) {
            *out = T::from_sample(0.0f32);
        }
    }
}

/// Resample once if needed, then map stereo onto the device's channel count.
fn prepare(interleaved: &[f32], device_rate: u32, device_channels: usize) -> Result<Vec<f32>> {
    let resampled = if device_rate == flacify_core::SAMPLE_RATE {
        interleaved.to_vec()
    } else {
        resample_stereo(interleaved, flacify_core::SAMPLE_RATE, device_rate)?
    };
    Ok(remap_channels(&resampled, 2, device_channels))
}

fn resample_stereo(interleaved: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    let divisor = gcd(from, to);
    let resampler = Resampler::new((to / divisor) as usize, (from / divisor) as usize)?;
    let frames = interleaved.len() / 2;
    let mut left = Vec::with_capacity(frames);
    let mut right = Vec::with_capacity(frames);
    for frame in 0..frames {
        left.push(f64::from(interleaved[frame * 2]));
        right.push(f64::from(interleaved[frame * 2 + 1]));
    }
    let left = resampler.process(&left);
    let right = resampler.process(&right);
    if left.len() != right.len() {
        anyhow::bail!(
            "resampling returned {} frames on the left and {} on the right",
            left.len(),
            right.len()
        );
    }
    let mut out = Vec::with_capacity(left.len() * 2);
    for frame in 0..left.len() {
        out.push(left[frame] as f32);
        out.push(right[frame] as f32);
    }
    Ok(out)
}

fn remap_channels(interleaved: &[f32], from: usize, to: usize) -> Vec<f32> {
    if from == to {
        return interleaved.to_vec();
    }
    let frames = interleaved.len() / from.max(1);
    let mut out = vec![0.0f32; frames * to];
    for frame in 0..frames {
        match to {
            1 => {
                let sum: f32 = (0..from)
                    .map(|channel| interleaved[frame * from + channel])
                    .sum();
                out[frame] = sum / from as f32;
            }
            _ => {
                for channel in 0..to.min(from) {
                    out[frame * to + channel] = interleaved[frame * from + channel];
                }
            }
        }
    }
    out
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a.max(1)
    } else {
        gcd(b, a % b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(name: &str, channels: usize, is_default: bool) -> DeviceInfo {
        DeviceInfo {
            name: name.to_string(),
            channels,
            default_sample_rate: 48_000,
            is_default,
        }
    }

    /// The counters are process-wide, so the tests that read them take turns.
    static COUNTER_TURN: Mutex<()> = Mutex::new(());

    fn spoken(list: &[&str]) -> Vec<DeviceInfo> {
        list.iter().map(|name| device(name, 2, false)).collect()
    }

    #[test]
    fn given_an_exact_name_in_another_case_when_resolved_then_that_device_is_selected() {
        let available = spoken(&["BlackHole 2ch (Virtual)", "External Headphones"]);
        assert_eq!(
            resolve(
                Some("external headphones"),
                &available,
                Some("BlackHole 2ch (Virtual)")
            ),
            Resolution::Selected("External Headphones".to_string())
        );
    }

    #[test]
    fn given_a_prefix_of_one_name_when_resolved_then_the_exact_name_still_wins() {
        let available = spoken(&["BlackHole 2ch", "BlackHole 2ch (Virtual)"]);
        assert_eq!(
            resolve(Some("BlackHole 2ch"), &available, None),
            Resolution::Selected("BlackHole 2ch".to_string())
        );
    }

    #[test]
    fn given_a_substring_only_one_device_contains_when_resolved_then_it_is_selected() {
        let available = spoken(&["External Headphones", "BlackHole 2ch (Virtual)"]);
        assert_eq!(
            resolve(Some("blackhole 2ch"), &available, None),
            Resolution::Selected("BlackHole 2ch (Virtual)".to_string())
        );
    }

    #[test]
    fn given_a_substring_two_devices_contain_when_resolved_then_it_is_missing_with_the_candidates()
    {
        let available = spoken(&["BlackHole 2ch", "BlackHole 16ch", "External Headphones"]);
        assert_eq!(
            resolve(Some("blackhole"), &available, Some("External Headphones")),
            Resolution::Missing {
                requested: "blackhole".to_string(),
                available: vec!["BlackHole 2ch".to_string(), "BlackHole 16ch".to_string()],
            }
        );
    }

    #[test]
    fn given_a_name_no_device_carries_when_resolved_then_it_is_missing_with_the_whole_list() {
        let available = spoken(&["BlackHole 2ch", "External Headphones"]);
        assert_eq!(
            resolve(Some("AirPods"), &available, Some("External Headphones")),
            Resolution::Missing {
                requested: "AirPods".to_string(),
                available: vec![
                    "BlackHole 2ch".to_string(),
                    "External Headphones".to_string()
                ],
            }
        );
    }

    #[test]
    fn given_no_name_and_a_default_when_resolved_then_the_default_is_named_in_the_reason() {
        let available = spoken(&["BlackHole 2ch", "External Headphones"]);
        let resolution = resolve(None, &available, Some("External Headphones"));
        match resolution {
            Resolution::Default { reason } => {
                assert!(reason.contains("External Headphones"), "{reason}");
            }
            other => panic!("expected the default, got {other:?}"),
        }
        assert_eq!(
            resolve(Some("   "), &available, Some("External Headphones")),
            resolve(None, &available, Some("External Headphones"))
        );
    }

    #[test]
    fn given_no_name_and_no_default_when_resolved_then_it_is_missing() {
        let available = spoken(&["BlackHole 2ch"]);
        assert_eq!(
            resolve(None, &available, None),
            Resolution::Missing {
                requested: String::new(),
                available: vec!["BlackHole 2ch".to_string()],
            }
        );
    }

    #[test]
    fn given_a_missing_name_when_the_message_is_built_then_it_names_the_request_and_the_list() {
        let message = missing_message(
            Direction::Output,
            "AirPods",
            &[
                "BlackHole 2ch".to_string(),
                "External Headphones".to_string(),
            ],
        );
        assert_eq!(
            message,
            "no output device matches \"AirPods\"; the system lists: BlackHole 2ch, External Headphones"
        );
    }

    #[test]
    fn given_no_request_and_no_default_when_the_message_is_built_then_it_says_there_is_none() {
        assert_eq!(
            missing_message(Direction::Input, "", &[]),
            "no default input device is available; the system lists: (none)"
        );
    }

    #[test]
    fn given_one_or_more_channels_when_labelled_then_the_word_agrees_with_the_count() {
        assert_eq!(channels_label(1), "1 channel");
        assert_eq!(channels_label(2), "2 channels");
    }

    #[test]
    fn given_a_buffer_longer_than_max_frames_when_filled_then_the_excess_is_silent_and_counted() {
        let _turn = COUNTER_TURN
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let before = counters().buffer_overflows;
        let (mut mixer, controls) = Mixer::new(vec![0.5f32; 8], None, 2, 48_000);
        controls.set_playing(true);
        let mut scratch = vec![0.0f32; 4];
        let mut data = vec![9.0f32; 8];

        fill_device_buffer(&mut mixer, &mut scratch, &mut data);

        assert_eq!(data, vec![0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(counters().buffer_overflows, before + 1);
    }

    #[test]
    fn given_a_buffer_within_max_frames_when_filled_then_audio_is_converted_and_nothing_is_counted()
    {
        let _turn = COUNTER_TURN
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let before = counters().buffer_overflows;
        let (mut mixer, controls) = Mixer::new(vec![0.5f32; 8], None, 2, 48_000);
        controls.set_playing(true);
        let mut scratch = vec![0.0f32; 8];
        let mut data = vec![0.0f32; 4];

        fill_device_buffer(&mut mixer, &mut scratch, &mut data);

        assert_eq!(data, vec![0.5, 0.5, 0.5, 0.5]);
        assert_eq!(counters().buffer_overflows, before);
    }

    #[test]
    fn given_one_stream_failure_when_asked_then_it_is_retried_and_a_second_stops() {
        assert_eq!(restart_after(1), Restart::Retry);
        assert_eq!(restart_after(2), Restart::Stop);
        assert_eq!(restart_after(9), Restart::Stop);
    }

    #[test]
    fn given_a_closed_engine_when_an_error_arrives_then_nothing_is_rebuilt() {
        // `None` in the slot is what `Drop` leaves behind: the window is gone, so a restart
        // would play into a closed track with nothing to show it.
        let slot = Arc::new(Mutex::new(None));
        let (sender, receiver) = mpsc::channel();
        sender
            .send("the device vanished".to_string())
            .expect("sending");
        let mut rebuilds = 0;

        supervise(
            receiver,
            Arc::clone(&slot),
            "External Headphones".to_string(),
            &mut || {
                rebuilds += 1;
                anyhow::bail!("a closed engine must not reach this")
            },
        );

        assert_eq!(rebuilds, 0);
    }

    #[test]
    fn given_a_44k1_mono_device_when_refused_then_the_error_names_the_device_and_its_format() {
        assert_eq!(
            streaming_refusal("progressive playback", "Built-in Output", 44_100, 1),
            "progressive playback needs a 48 kHz stereo device; \"Built-in Output\" is 44100 Hz with 1 channel"
        );
        assert_eq!(
            streaming_refusal("segment playback", "Built-in Output", 44_100, 1),
            "segment playback needs a 48 kHz stereo device; \"Built-in Output\" is 44100 Hz with 1 channel"
        );
        assert_eq!(
            streaming_refusal("live playback", "Built-in Output", 44_100, 2),
            "live playback needs a 48 kHz stereo device; \"Built-in Output\" is 44100 Hz with 2 channels"
        );
    }

    #[test]
    fn given_a_clean_stream_when_asked_for_the_warning_then_there_is_none() {
        assert_eq!(stream_warning("BlackHole 2ch", Counters::default()), None);
    }

    #[test]
    fn given_a_stream_error_when_asked_for_the_warning_then_it_names_the_device_and_the_counts() {
        let counters = Counters {
            stream_errors: 2,
            buffer_overflows: 1,
        };
        let warning = stream_warning("BlackHole 2ch", counters).expect("a warning");
        assert!(warning.contains("BlackHole 2ch"), "{warning}");
        assert!(warning.contains("2 stream errors"), "{warning}");
        assert!(warning.contains("1 buffers over 16384 frames"), "{warning}");
    }

    #[test]
    fn given_a_list_when_described_then_each_device_shows_its_format_and_the_default() {
        let list = DeviceList {
            inputs: vec![device("BlackHole 2ch", 2, true)],
            outputs: vec![device("External Headphones", 2, false)],
        };
        let text = describe(&list);
        assert!(text.contains("inputs"), "{text}");
        assert!(text.contains("\"BlackHole 2ch\""), "{text}");
        assert!(text.contains("48000 Hz x 2 channels  (default)"), "{text}");
        assert!(text.contains("\"External Headphones\""), "{text}");
        assert_eq!(text.matches("(default)").count(), 1, "{text}");
    }

    #[test]
    fn given_no_devices_when_described_then_both_sides_say_none() {
        let text = describe(&DeviceList::default());
        assert_eq!(text.matches("(none)").count(), 2, "{text}");
    }

    #[test]
    fn given_a_mono_device_when_remapped_then_channels_are_averaged() {
        let stereo = vec![1.0, 0.0, 0.5, 0.5];
        assert_eq!(remap_channels(&stereo, 2, 1), vec![0.5, 0.5]);
    }

    #[test]
    fn given_a_four_channel_device_when_remapped_then_the_extra_channels_are_silent() {
        let stereo = vec![0.25, 0.5, 0.75, 1.0];
        assert_eq!(
            remap_channels(&stereo, 2, 4),
            vec![0.25, 0.5, 0.0, 0.0, 0.75, 1.0, 0.0, 0.0]
        );
    }

    #[test]
    fn given_a_44k1_device_when_resampled_then_the_length_follows_the_ratio() {
        let frames = 4410;
        let interleaved: Vec<f32> = (0..frames).flat_map(|_| [0.5f32, -0.5]).collect();
        let out = resample_stereo(&interleaved, 44_100, 48_000).unwrap();
        assert_eq!(out.len(), 4800 * 2);
        assert!((out[1000] - 0.5).abs() < 1e-3);
    }
}
