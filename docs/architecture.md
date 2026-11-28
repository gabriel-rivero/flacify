# Architecture

Status: current for v1.0, checked against the code 2026-11-28. Where the ADRs described
a plan the code has since replaced (ADR-002's ONNX Runtime, for one), this page records
what the code does and names the ADR that changed it; the decisions themselves are in
[`decisions.md`](decisions.md).

## Overview

```
                 ┌──────────── control plane (egui, atomics, channels) ────────────┐
                 │  toggle · seek · gain · window size · probe state · meters      │
                 └───────▲──────────────────────────────▲─────────────────────────┘
                         │                              │
 source ──▶ decode ──▶ preprocess ──▶ window ──▶ infer ──▶ overlap-add ──▶ mixer ──▶ device
 (file,     (symphonia  (cutoff +  (5.12 s   (Python   (cosine xfade,   (A/B xfade)   (cpal)
  capture)   → f32 48k)  lowpass)    frames)   sidecar)  loudness match)
                              │
                              └──▶ render cache (segmented FLAC, background)
```

Five stages, one queue, one clock. The two axes that vary are the source (file vs
live capture) and the buffer policy (render-ahead vs bounded). Everything else is
identical, which is the whole point of the rewrite.

## Crates

| crate | owns | depends on |
|---|---|---|
| `crates/core` | windows, overlap-add, xfade, delay line, buffer policy, cutoff detector, loudness, state machine, the `Upscaler` trait and its mocks | `realfft`, `thiserror` — no I/O |
| `crates/infer` | the `Model` trait (one window in, one out), the deterministic noise, the `Model`-driven `Upscaler`, and the Python sidecar client that serves a window | `flacify-core`, `thiserror` |
| `crates/app` (bin) | symphonia decode, `cpal` output, cache, probe, egui UI, CLI | the two above, plus the crates in [`dependencies.md`](dependencies.md) |

Two traits, two boundaries, and the split is deliberate. `Upscaler` lives in `core`
because the pipeline's own tests need mock implementations and must run with no model;
`Model` lives in `infer` because it is the shape of the window the backend accepts, and
nothing in `core` may know about tensors.

`crates/infer` names no inference runtime on purpose. ADR-010 put the model in a Python
sidecar the app installs, so the crate holds a client for `tools/worker.py` and a trait
the backend sits behind; the ONNX path ADR-002 planned is not in the tree. Swapping the
backend is one implementation of `Model`, not a new dependency in `core`.

`core` must compile and test with no GPU, no audio device, and no model. That is
what makes the project developable on a laptop, in CI, and by someone who has not
downloaded 3.3 GB.

Resist adding crates. `codec/`, `cache/`, `probe/`, `ui/` are modules inside `app`
until something else needs them.

## Stages

### 1. Decode

`symphonia` to `f32`, planar, **48 kHz** — the model's native rate. Resample once,
at the source, with the crate's own polyphase resampler
(`crates/core/src/dsp/resample.rs`, a port of `scipy.signal.resample_poly`), not a
third-party one. 44.1 kHz sources are the common case and the resampler is on the
critical path, so its cost is measured in [`measurements.md`](measurements.md).

Decode ahead of the renderer by a bounded amount (the lookahead the windower needs,
plus one window). Do not decode the whole file into RAM: an hour of stereo float is
1.38 GB.

### 2. Preprocess

Two operations, both inside the 5.12 s window, both ported from the Python chain.
This is the highest-fidelity-risk part of the whole project; see
[The preprocessing port](#the-preprocessing-port).

### 3. Window

Advance by `window − overlap`, clamp `overlap ≤ window/2 − 1`, zero-pad the final
partial window, and emit exactly the input sample count. Windows must be a multiple
of 30720 samples (0.64 s); the constraint comes from the mel framing inside the
VAE and is worth re-confirming at export time, because it decides whether latency
presets are possible at all.

### 4. Infer

One call per window on the render thread. Blocking, CPU-side wait on GPU, so it
gets its own thread and never runs inside an audio callback. Details in
[Inference](#inference).

### 5. Overlap-add and output

Raised-cosine crossfade over `overlap`, tail held back and blended into the next
window. Loudness match per window (BS.1770 momentary, replacing the PoC's RMS EMA,
which pumps on dynamic material), clamped for safety; see [Loudness](#loudness).

## Threads

| thread | owns | rules |
|---|---|---|
| decode | `symphonia` decoder, resampler | fills the source ring; stops when full |
| pump | windower, the `Model` (one sidecar call per window), cache writer | the only thread that calls the model |
| audio callback | `cpal` output, mixer, delay line | **alloc-free, lock-free, no logging, no GPU** |
| UI | egui, probes, file dialogs | reads atomics, sends commands |
| setup | `uv`, pip and the weights fetch, as child processes | cancellable, resumable |

Data flow between them:

- source ring: one bounded ring of interleaved frames, samples in `AtomicU32`, addressed
  by absolute frame number (`live.rs`). A producer writes, the audio thread reads, and
  neither takes a lock.
- output ring: the same ring type, pump → audio callback.
- control: `std::sync::mpsc` for commands (seek, open file, toggle, device errors), and
  `arc-swap` for what the UI reads (the render frontier, the resident segment, the
  render status). Neither is touched from the callback.
- meters: `AtomicU64`/`AtomicUsize` only, read by the UI at 30 Hz.

There is no `rtrb`, no `crossbeam-channel` and no `ureq` in the dependency tree: the
standard library's channels and atomics cover what the rings and the control plane need,
and downloads are child processes, not HTTP calls from Rust. See
[`dependencies.md`](dependencies.md).

The audio callback's only job is to pop samples from a ring, apply the A/B crossfade
and the master gain, and write them to the device. What it does when the ring is empty
depends on the source: a segment-backed file holds the playhead and writes silence
until the segment arrives, while the live path emits the dry side and counts the frames
it starved. It never waits for anything either way.

## Buffer math

Let `r` = RTF, seconds of compute per second of audio, **measured sustained** (not
from one window). Producer rate is `1/r` audio-seconds per wall-second, consumer is
`1`.

| measured sustained `r` | policy | prebuffer | time to first audio | risk |
|---|---|---|---|---|
| ≤ 0.4 | render-ahead | 1 window | ≈ `r × window` + load | none |
| 0.4 – 0.7 | render-ahead | 2 windows | ≈ 2–4 s | jitter only |
| 0.7 – 1.0 | render-ahead | 3 windows | ≈ 5–10 s | underrun on throttle; passthrough armed |
| > 1.0 | pre-render | `T(r−1)/r` | `T(r−1)` | not real-time; offer offline render |

Two consequences worth internalising:

1. **For a file, streaming is slower to first audio than rendering ahead.**
   The decoder is not rate-limited, so the renderer can produce a window as fast as
   the GPU allows. Waiting for wall-clock time to elapse is pure added latency. The
   only reason to bound the buffer is memory, and the cap is cheap: 30 s of stereo
   f32 at 48 kHz is 11.5 MB.
2. **A buffer cannot save a song when `r > 1`.** It can only save part of it, and
   the part it can save is exactly `T(r−1)/r`. So the answer to "how many seconds
   must we upscale to play a song continuously" is: none of them if the GPU is
   slower than real time; for `r < 1`, the buffer depth is set by jitter and by how
   much latency the user tolerates, not by the song length.

Cap and backpressure: the render thread sleeps when the output ring holds more than
`cap` seconds. `cap` = 30 s by default, 2 × window minimum. Backpressure is also what
keeps the A/B delay constant, so it is not just a memory optimisation.

## A/B alignment

The toggle is the product, so it has to be exact: same content, same instant, same
loudness, no click.

```
output ring ──▶ enhanced tap ─────────────────┐
   (position n)                                ├─▶ equal-power xfade (20 ms) ──▶ out
             └─▶ delay D ──▶ passthrough tap ─┘
```

- `D` equals the current ring depth **at the moment of the toggle**, in samples, and
  is held fixed while enhanced audio is selected. Backpressure keeps the ring at
  `cap`, so `D` is constant in steady state.
- Both taps read the same sample index `n`, so content is aligned by construction.
  There is no resampler and no timestamp comparison in this path.
- The switch is a 20 ms equal-power crossfade between two aligned signals. Neither
  branch restarts, so there is no gap and no re-seek.
- Loudness: the enhanced branch is matched to the input branch per window, and the
  residual difference is applied as a slow (~1 s) gain trim so the crossfade cannot
  produce a level step.
- If the ring cannot sustain `cap` (the GPU fell behind), the toggle is **disabled
  with a reason** ("buffering — A/B needs 3 s of lead") rather than desynced. A
  desynced A/B is worse than no A/B: it is a lie.

In cache mode (v0.3) both taps are file readers at the same position and `D` is zero.
Keep the same crossfade code so the two paths cannot drift apart.

## Live mode (v1.0)

One pipeline, two feeds. `Source::Live` takes a `Feed` — the decoder for a file, the
capture stream for the sink — and the mixer, ramp, level match and starvation policy are
the same code for both. ADR-014 records the decision; this section records the mechanics.

```
feed ──▶ dry ring (B + W) ──┐
                            ├──▶ wet producer (model) ──▶ wet ring (B + block) ──┐
                            │                                                   ├──▶ callback
                            └──────────── alignment delay W ─────────────────────┘
```

- `W` is the model window and `B` is the buffer `live_policy` derives from the measured
  RTF. The dry ring holds `B + W` because the dry side is delayed by one window: the
  wet side cannot exist earlier than that, and A/B must compare the same instant.
- One producer call renders a window and hands over one block of it, so the block is the
  pipeline's step: `block = W - overlap` with `overlap = W / 5`, the trained window's own
  1 s in 5.12 s. Keeping that share fixed across the presets is what keeps the compute
  per played second flat (D19).
- Resident audio is `(2B + W + block) × 2 channels × 4 bytes`. It is the rings only; the
  model's own input window and the in-flight output block are not counted. The wet ring
  carries one block over `B` so a whole-block write can land while the ring is otherwise
  full; what the playhead can be behind is still `B`. The sink adds its own capture ring in
  front of the dry ring: 400 ms, 153 600 bytes, steered to half full.
- Credit backpressure: the producer blocks while the wet ring is `B` ahead of the
  playhead. Nothing is dropped and nothing grows; a slow model slows the producer instead
  of the memory.
- The device starts only once the wet ring holds whole blocks that fill `B`, which is one
  block at every preset. A call renders a window before it can hand over anything, so the
  first call's own duration is the one dropout a careful start can prevent.
- Starvation: when the wet side cannot cover the playhead, the callback emits the dry
  side, counts the frames, and `has_enhanced` goes false with a stated reason until the
  buffer refills. This is deliberately different from the segmented path, which holds
  the playhead and writes silence — right for a missing cache segment, wrong for live
  audio, where silence is a gap.
- Latency is `W + B + device`, printed from `live_policy`, never implied by the preset.
  The measured ratio is printed beside it, per second of *played* audio, so it is the
  number to compare with 1.
- A stream has no length, no content hash and no segments, so live audio is in memory
  only: nothing about it joins the cache.
- Capture can succeed and deliver nothing: on macOS an unpermitted process opens the device and reads exact zeros, with no error. The sink treats digital silence as its own state and names the permission, because "it is playing" and "it is capturing" look identical otherwise.
- A file feed has one clock. The sink has two — the capture device and the output device
  drift by tens of ppm — so the sink corrects the rate instead of restarting: the capture
  ring is read through a Catmull-Rom interpolator whose ratio is a proportional controller
  on the ring's fill, `ppm = 2 × (fill − target)` clamped to ±1000 ppm, with the target at
  half of the 400 ms ring. The plant is first order, so the loop settles without overshoot:
  a 10 s time constant, and 50 ppm of drift held at 25 frames of error. When the correction
  saturates and the fill reaches a quarter-ring edge, the read drops or repeats a quarter
  ring (100 ms) under a 10 ms crossfade and counts a resync, because a deliberate resync
  beats a click or a latency that grows until the ring tears.
- A capture that delivers 5 s of digital silence is a state of its own, not quiet audio.
  macOS never refuses the input device for a missing Microphone permission; it opens it and
  returns zeros, so the permission is named rather than guessed at, and the sink refuses to
  start on that state.
- In 1.0 this path is opt-in (`play --live`, `flacify sink`). The default first play of
  an uncached file stays on the progressive path until 1.1.

## Determinism

The one-step sampler starts from `torch.randn(...)`, so **the PoC's output is a
different take on every run**. The Rust port must not inherit that:

- Generate the initial noise in Rust with a seeded RNG and pass it into the graph as
  a tensor. The graph takes `(cond, noise)`, not `(cond)`.
- Seed from `hash(file_id, window_index, pipeline_version)`. Re-rendering a file
  reproduces the same audio, so a cache hit and a fresh render are identical, and a
  bug report can be replayed.
- Expose "re-roll" later as an explicit action (a new seed) and show variants in the
  A/B report. Do not let randomness leak into normal playback.

This is also why window size is pinned per session: changing it changes the seed
mapping, so it is a render-parameter change, not a live toggle.

## Inference

The export boundary is the whole forward pass *after* preprocessing:

```
inference(cond: f32[2, 245760], noise: f32[latent]) -> f32[2, 245760]
```

`cond` is the preprocessed window; the one-step scheduler math, VAE encode/decode,
UNet, mel transform, vocoder and denormalisation are all folded into the graph. Noise
enters as an input so it stays seedable from Rust.

**The noise is not audio-shaped.** Upstream draws it *after* the VAE encode, so its
shape is the latent grid (`get_x_shape(cond)` where `cond` has already been replaced by
`vae.encode_to_z(...)`), not `[2, 245760]`. The export script prints the exact shape
and records it in `tools/out/export_report.json`; `Model::noise_len` exists so the
engine asks the graph instead of assuming the audio length.

| decision | choice | reason |
|---|---|---|
| graph format | ONNX, **opset 18** | the dynamo exporter needs 18+ and the TorchScript one cannot write a graph past protobuf's 2 GB; weights go to a `flashsr.onnx.data` sidecar (measured: 8.5 MB graph + 2.46 GB data) |
| shapes | static per export: `cond` and the output are `2×245760`, noise is the latent grid (`2×16×64×32` as exported) | CoreML prefers static; enables shape-specific kernels |
| dtype | fp32 export; fp16 tried for CoreML and measured, see below | bf16 is a CUDA win; CoreML wants fp16 |
| VAE encode | posterior **mean**, not `sample()` | `sample()` is a second unseeded RNG; ADR-006 wants one explicit noise input |
| window flexibility | **one graph, fixed 5.12 s** | 4 window sizes = 4 graphs ≈ 4× disk. See below |
| EP order (macOS) | CoreML → CPU | no CUDA exists on Apple Silicon |
| EP order (Win/Linux) | TensorRT → CUDA → DirectML → CPU | consumers rarely have a CUDA toolkit |

**Measured outcome (2026-11-12, M5 Max).** The numeric gate passes and the throughput
gate fails, on every ONNX Runtime configuration tried: CPU EP RTF 21.5, CoreML with fp32
NeuralNetwork 31.1, fp16 NeuralNetwork 21.7, fp16 MLProgram + ALL units 15.2, against a
gate of 1.5. PyTorch on Metal runs the same weights at RTF 0.51. The model is fast enough;
ONNX Runtime is what is slow, and the CoreML EP mostly fell back to CPU. The numbers and
the method are in [`measurements.md`](measurements.md); the open decision is backlog
E1-4 to E1-6.

**Prototype deployment (ADR-010).** Until a Rust-native backend reaches Metal, the app
runs the model in a Python sidecar it installs on first run and talks to over a binary
pipe (`tools/worker.py`): one window in, one window out, 2 ms of IPC, RTF 0.543,
bit-identical to in-process Metal. `Model` is the seam, so this is one implementation
rather than a fork of the design, and it never ships in a signed build.

**The window-size cost.** Supporting 0.64 / 1.28 / 2.56 / 5.12 s at full quality
means either dynamic axes (CoreML EP handles these poorly) or four copies of a
3.3 GB model. So v1 ships one window size, chosen by measurement, and the latency
presets arrive only if dynamic axes survive the spike. The PoC's flexibility was
free because the Python runtime was already there; here it costs gigabytes. Decide
this in v0.0, not in v0.4.

The sidecar made the presets cheap after all: `tools/worker.py` derives the latent
grid from `channels, frames`, so 1, 2, 4 and 8 granules are an input it already
accepts, and `flacify play --live --window N` / `flacify sink --window N` expose them
(`Preset` in `crates/core/src/policy.rs`). Measured cost per preset, and the listening
verdict that 2 granules are noisy, are in
[`measurements.md`](measurements.md#window-presets-s1-2026-11-18).

**Verification gate.** Export is not done when it runs; it is done when it matches
PyTorch. Take five tracks, render them with a fixed seed through both paths, and
require ≤ −40 dB relative RMS difference per window, plus an identical cutoff
decision per window. Below that the export is wrong, whatever it sounds like.

## Modules

One module per job, in the crate that owns it. Each file opens with a `//!` line saying
what it is; this table is the map to those lines.

### `crates/app` (the binary)

| module | job |
|---|---|
| `main.rs` | parse the CLI and dispatch to one `commands/` module |
| `cli.rs` | the clap surface: one subcommand per thing the app does |
| `commands/` | one `run` each: `setup.rs`, `probe.rs`, `enhance.rs`, `play.rs`, `ab.rs`, `batch.rs`, `cache.rs`, `devices.rs`, `diagnose.rs`, `sink.rs` |
| `decode.rs` | symphonia → planar f32, channel map, polyphase resample to 48 kHz |
| `render.rs` | one track through the model, start to finish |
| `session.rs` | one app run: installed paths, one resident model, a file becoming playback |
| `player.rs` | the mixer: two signals, one position, an 8 ms ramp, atomics only |
| `segments.rs` | segmented playback: the callback reads decoded FLAC segments, not a track |
| `source.rs` | the source tier: one sequential pass writes 48 kHz stereo FLAC segments |
| `cache.rs` | the render cache: keys, manifests, eviction, claims |
| `frontier.rs` | the render frontier: audio that exists, published one second at a time |
| `render_status.rs` | what the background render is doing, in a form the window reads |
| `live.rs` | bounded-memory playback: a feed in, the A/B mix out |
| `sink.rs` | the sink's feed: one capture device into the live rings, rate-corrected |
| `device.rs` | `cpal` output and capture: list, open by name, one-time resample |
| `ab.rs` | the blind A/B run: seeded plan, answers, level gate, report |
| `batch.rs` | a folder rendered one file at a time |
| `diagnose.rs` | one text file with the facts an issue needs |
| `setup.rs` | first-run provisioning: `uv`, CPython, wheels, weights, `yt-dlp`, resumable state |
| `process.rs` | running another program: one child, its output read as it arrives, killed on cancel |
| `youtube.rs` | a URL as a source: `yt-dlp` fetches the audio or lists a playlist, then the pipeline gets a file |
| `playlist.rs` | the queue: folder, `.m3u`/`.m3u8`/`.pls` and yt-dlp listings as entries, one cursor |
| `first_run.rs` | provisioning from inside the window |
| `ui.rs` | the egui window: one toggle, one transport, progress, the truth about enhancement |
| `spectrum.rs` | two spectra of the same moment, so the switch shows what it does |
| `splice.rs` | swap the model's low band for the input's |
| `loudness.rs` | BS.1770 loudness, so the two sides of the switch are the same level |
| `limit.rs` | keep a rendered track inside full scale |
| `flac.rs` | writing 24-bit FLAC |
| `wav.rs` | writing 32-bit float WAV |
| `bin/sink_probe.rs` | S2: play a tone into a loopback device and read it back, to measure drift |

### `crates/core` (pure, no I/O)

| module | job |
|---|---|
| `lib.rs` | the constants (`GRANULE`, `SAMPLE_RATE`) and the public surface |
| `block.rs` | the `Block`: planar f32 frames handed between stages |
| `pipeline.rs` | windowing, overlap-add and the `Tail` |
| `policy.rs` | `Preset`, `Probe` and what to do with the measured speed of the machine |
| `preprocess.rs` | band-limiting and zero-phase filtering before the model sees a sample |
| `upscaler.rs` | the `Upscaler` trait and its mocks (`Identity`, `Gain`, `Delay`, `Noise`, `Fail`) |
| `error.rs` | the crate's typed errors |
| `dsp/cheby.rs` | Chebyshev I lowpass design, ported from `scipy.signal.cheby1` |
| `dsp/cutoff.rs` | band-limit detection, a port of `UtilAudioSR.find_cutoff_freq` |
| `dsp/filtfilt.rs` | zero-phase filtering, ported from `scipy.signal.sosfiltfilt` |
| `dsp/resample.rs` | polyphase resampling, ported from `scipy.signal.resample_poly` |

### `crates/infer` (the model boundary)

| module | job |
|---|---|
| `lib.rs` | the seam between the pipeline and a model |
| `model.rs` | the `Model` boundary: one window in, one window out, `[channels, frames]` |
| `noise.rs` | reproducible noise for the sampler's starting point (ADR-006) |
| `engine.rs` | the `Upscaler` the pipeline drives: preprocess, add noise, call the model |
| `python.rs` | the sidecar: spawn `tools/worker.py` and serve one window per request |
| `session.rs` | one model worker, kept for the life of the app |
| `bin/fake_worker.rs` | a stand-in for `tools/worker.py`, so the client's tests need no Python |

## The preprocessing port

Three stages, all currently Python, all required before the model sees a sample.
This is the most likely place for the Rust output to diverge from the PoC.

1. **Cutoff detection.** STFT (n_fft 2048, hop 480, Hann, `center=False`), take
   channel 0, sum magnitudes over time, cumulative sum over bins, find where the
   cumulative energy crosses 98.3% of the total, map bin → Hz (`bin / 1024 × 24000`),
   clamp anything below 1000 Hz to 24000. Small enough to port in an afternoon;
   must match bin-for-bin on a golden corpus, because a different cutoff changes
   every subsequent sample.
2. **Zero-phase lowpass.** `scipy.signal.cheby1(N=8, rp=0.1, Wn=cutoff/24000)` in SOS
   form, applied with `sosfiltfilt` (forward then reverse). Zero-phase means
   non-causal, so it needs the whole window: fine, we already have 5.12 s. Note the
   edge transient: the PoC's own `align_length` truncation and the reflective
   padding behaviour matter for the first and last ~100 ms of each window.
3. **Band-limit round trip.** `resample_poly` down to `cutoff` and back up to 48 kHz.
   This is what actually removes the high band before the model sees it, and it is
   the stage most people miss when reimplementing. The port designs the polyphase FIR
   to match scipy rather than taking a library resampler: same Kaiser window, ten lobes
   per side, and the same output offset, pinned by the scipy golden tests in
   `crates/core/tests/fixtures.rs`.

Do **not** leave this inside the ONNX graph. `lowpass_input=True` in Python runs a
scipy filter with a cutoff computed per window; exporting it bakes one cutoff into
the graph and makes the model unaware of the actual source. Preprocess in Rust, feed
the model `lowpass_input=False`.

## Cache

v0.3. A render cache turns a fragile real-time problem into a file-management
problem, and it buys seek, instant replay, and a zero-GPU second listen.

- **Key**: `blake3(source bytes) + blake3(weights) + pipeline version + window +
  overlap + dtype + seed policy`. Any change invalidates.
- **Layout**: `cache/<key>/manifest.json` plus `seg-000000.flac` … in 7-step segments
  (1 384 320 frames, 28.84 s at the default window). A segment starts on a multiple of
  `window - overlap`, so its length is derived from the settings and is not a wall-clock
  30 s. Segments keep whatever a resumed render already produced, and a settings change
  invalidates only what it must. The manifest is the source of truth: a segment it does
  not list is not part of the entry, and its `state` says whether the table is the whole
  track yet.
- **Format**: FLAC, 24-bit. This is what FLAC is for: lossless storage of the
  generated signal. Output f32 rounded to 24-bit is transparent (−144 dB quantisation
  floor) and roughly half the size of raw float.
- **Source tier**: `cache/source/<source_hash>/source.json` plus `src-000000.flac` …,
  the decoded 48 kHz stereo source in segments of the same 7-step length. One sequential
  decode+resample pass writes it; `source.json` records the native rate and channels, the
  segment table, one RMS pair per second for the waveform, and the `frames_per_segment`
  a reader checks on open. A range read decodes only the segments it touches.
- **Playback from cache**: one segment reader answers for both sides (the source tier's
  original and the entry's enhanced audio) on the same segment grid, so the toggle is a
  crossfade with `D = 0` and the playhead is an index into a decoded segment. A worker
  thread decodes the segment at the playhead and the next one and keeps at most three
  resident, so a complete entry is never decoded whole. A segment that is not decoded yet
  falls back to passthrough; a seek past the prefetch holds until the worker catches up.
  The window draws a cache hit without a whole-track buffer: the waveform from the source
  tier's per-second peaks, the spectrum from the one resident segment at the playhead. The
  first play of an uncached track still holds the whole source, and is drawn differently on
  the waveform.
- **Eviction**: LRU by the manifest's `last_used`, against a 20 GB default cap
  (`flacify cache --evict [--cap-gb]`). It sweeps leftover staging and other-version
  directories first, then drops each entry's source tier, which a decode pass rebuilds, and
  only then whole entries. An entry something has open is skipped: an opener records
  `<pid> <claim> <seconds>` in `cache/.claims/entry-<key>` and refreshes it on every segment
  read, and eviction leaves a claim younger than an hour alone, so a process that died stops
  pinning its entry once that hour is up. The claim is advisory, not a kernel lock, so
  eviction re-checks it immediately before the rename: an opener that arrives in that gap
  finds the entry gone and misses cleanly instead of half-reading. A removal renames the
  directory to `<name>.evict-<pid>` and deletes that, so a kill leaves nothing a lookup can
  open. The accounting comes from the manifests and the bytes are measured, so a manifest
  that names more or less than its directory holds is reported rather than believed.
- **Writer**: one segment at a time, each written whole and renamed into place; the
  manifest is rewritten atomically after it, with both `fsync`ed, so a manifest never
  names bytes that are not on disk. An entry stays `partial` until a writer marks it
  complete, and a partial entry is a miss: a kill leaves playable segments plus a partial
  manifest, and the next writer adopts a segment the manifest does not list when it
  decodes to the length its index requires, and deletes the ones that do not. The loudness
  gain is written before segment 0 because a later process cannot recompute it; a resumed
  render recovers the crossfade tail the previous segment handed on by rendering that
  segment's last window, which one model call reproduces exactly.

## Fetched audio

A URL is resolved to a file before the pipeline sees anything, so everything above applies
unchanged: the source tier, the render key and the A/B switch cannot tell a download from a
local file.

- **Layout**: `<home>/downloads/<video id>/` holds the audio and a `names.tsv` with the
  title and artist yt-dlp printed. Outside `cache/` on purpose: the cache scan sweeps a
  directory it cannot read as an entry, and a download is not one. `flacify cache` reports
  the bytes separately and never evicts them.
- **One file per id**: the file named after the id wins over any other decodable file in
  the directory, because `flacify enhance <url>` used to write its render beside the source;
  it now writes to the working directory for a URL.
- **Container**: with ffmpeg, the fetch asks for FLAC, which is lossless and is what
  `symphonia` decodes. Without it, it asks for a container YouTube already serves
  (`NATIVE_FORMAT`: AAC in MP4), because yt-dlp only needs ffmpeg to remux or transcode —
  asking for FLAC on a machine that has none downloads the audio and then dies at the
  conversion step. `symphonia` here has no Opus and no ALAC codec, so yt-dlp's default
  WebM/Opus would not decode, and `youtube::DECODABLE` names only the containers whose
  codec is always one this build has.
- **Reuse**: a file keyed by id is trusted without asking yt-dlp again, but it is decoded
  before it is trusted, so a truncated file left by an interrupted run re-fetches rather
  than failing forever.
- **Playlists**: a URL that names a playlist (`/playlist`, or a `list=` parameter that is
  not a radio mix) is listed with `yt-dlp --flat-playlist`; each entry is fetched lazily and
  the next one is prefetched. `--radio` expands a video into its
  `list=RD<id>&start_radio=1` mix. A share link carrying `list=RD…` without `--radio` stays
  one video, which preserves the behaviour the URL measurement below records.

## Loudness

The PoC matches RMS per window with a 0.2–5× clamp and a 0.5 EMA. It works, but it
pumps on tracks with quiet verses. Replace with ITU-R BS.1770 momentary loudness
(400 ms window, K-weighting) over the window, gate on silence, and smooth with a
1 s time constant. Clamp for safety. The test is a dynamic track: the enhanced
version must not audibly breathe against the original when toggled.

## UI

One window, `egui`, custom-painted to look like a compact player rather than a
desktop form.

- **Spectrum**: 2048-point STFT, log frequency, 20 Hz–24 kHz, with the region above
  the detected cutoff shaded. Watching the top octave fill in is the demo.
- **Waveform**: per-second RMS bars, with the rendered frontier marked. Unrendered
  audio is dimmed so the user can see the renderer catching up.
- **Transport**: play/pause, seek, A/B, gain, next/previous (`N`/`B`). A queue strip shows
  the position (`3 / 12`), the next track, and a remote fetch in progress. A single file is
  a queue of one, which hides the strip.
- **URL box**: a text field in the window (empty state and player alike) turns a pasted
  link into a queue, or seeds a radio from its video when `radio` is ticked; a playlist URL
  is always a queue. The listing and the first fetch run on a thread, so the window that
  had nothing playing does not freeze on yt-dlp; a failed listing leaves the current track
  alone. This is the packaged app's only way to a remote source, and the transport keys
  stand down while the field has focus so `b` in `youtu.be` is a letter.
- **Status**: RTF, buffer depth, dropout count, backend. Small, monospace, always
  visible. This is the credibility surface.
- **A/B**: one key (`A`), one button, and optionally "hold to hear original" which is
  how people actually compare.

Rendering is on the UI thread at 60 Hz; nothing on the audio thread depends on it.

## Failure policy

One state machine, four terminal states. Every failure maps to one of them, and none
of them is "hang".

| state | trigger | behaviour |
|---|---|---|
| `Playing` | normal | enhanced, toggle enabled |
| `Passthrough` | no GPU, model missing, `r > 1.5`, underrun | original plays, UI explains why |
| `Rendering` | cache mode, ahead of frontier | plays rendered part, passes through the rest |
| `Blocked` | file decode error, device lost, licence not accepted | stops, says what to do |

Rules: never stop playback on an inference error (log, pass through, count it); never
block the audio thread (timeouts everywhere); never claim enhancement when the
output is passthrough (the UI has one authoritative source for that).

## Observability

- Structured logs (`tracing`) to a rolling file; no network by default.
- A "stats" panel and a `--stats` CLI flag: RTF p50/p95, dropout count, cache hit
  rate, window count, EP in use, peak memory.
- A `--diagnose` command that dumps hardware, drivers, EP availability, model hash,
  and one benchmark run — the thing you paste into an issue.

## Testing

The pipeline must be testable without a model, a GPU, or an audio device. That is a
design constraint, not a convenience.

| layer | how |
|---|---|
| `Upscaler` trait | `Identity`, `Gain(n)`, `Delay(n)`, `Noise`, `Fail(after n)` implementations |
| sample count | identity run reproduces the input sample-for-sample, bit-exact within 1 ULP |
| crossfade | raised-cosine pair sums to 1.0 ± 1e-6; no click at any overlap |
| A/B | null test: enhanced == passthrough when the mixer is at 0 or 1; a 10 ms impulse at the switch is below −80 dBFS |
| buffer | simulated producer rates 0.3–2.0 → assert no underrun below 0.95, passthrough above 1.0 |
| cutoff | golden corpus, Python vs Rust, bin-exact |
| preprocess | golden corpus, Python vs Rust, ≤ −60 dB |
| model | the fake worker (`crates/infer/src/bin/fake_worker.rs`) for the protocol, a real worker manually for numbers |
| audio device | a simulated device in CI, a real device by hand (`bin/sink_probe.rs` measures drift) |
| UI | state-machine tests over the plain data the window reads, manual for looks |

Fuzz the windower over lengths 0…3 windows and every overlap value; the off-by-one at
the final partial window is where this class of code always breaks. `proptest` is not a
dependency yet, so the case table is hand-written.

## Invariants

Write these as assertions and tests, not comments.

1. Output sample count equals input sample count, always, for any file length.
2. The emitted stream is continuous: no window boundary introduces a discontinuity
   larger than the crossfade's own first derivative.
3. The A/B taps are sample-locked; a toggle cannot change which content is playing.
4. The audio callback never allocates, locks, logs, or blocks.
5. No inference error can stop playback.
6. Re-rendering the same file with the same parameters produces identical bytes.
7. Latency reported in the UI equals measured latency within 50 ms.
