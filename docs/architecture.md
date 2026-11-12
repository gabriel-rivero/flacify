# Architecture

Status: draft, 2026-11-12. Written against the recommended option in each ADR
([`decisions.md`](decisions.md)).

## Overview

```
                 ┌──────────── control plane (egui, atomics, channels) ────────────┐
                 │  toggle · seek · gain · window size · probe state · meters      │
                 └───────▲──────────────────────────────▲─────────────────────────┘
                         │                              │
 source ──▶ decode ──▶ preprocess ──▶ window ──▶ infer ──▶ overlap-add ──▶ mixer ──▶ device
 (file,     (symphonia  (cutoff +  (5.12 s   (ONNX)    (cosine xfade,   (A/B xfade)   (cpal)
  capture)   → f32 48k)  lowpass)    frames)             loudness match)
                              │
                              └──▶ render cache (segmented FLAC, background)
```

Five stages, one queue, one clock. The two axes that vary are the source (file vs
live capture) and the buffer policy (render-ahead vs bounded). Everything else is
identical, which is the whole point of the rewrite.

## Crates

| crate | owns | depends on |
|---|---|---|
| `crates/core` | windows, overlap-add, xfade, delay line, buffer policy, cutoff detector, loudness, state machine, the `Upscaler` trait and its mocks | `realfft` — no I/O |
| `crates/infer` | ONNX session, execution-provider selection, model download/verify, the `Model` trait (one window in, one out) | `ort`, `reqwest`, `sha2` |
| `crates/app` (bin) | symphonia decode, `cpal` output, cache, probe, egui UI, CLI | the two above |
| `crates/eval` (bin, later) | metrics, spectrograms, blind A/B reports | `core` only |

Two traits, two boundaries, and the split is deliberate. `Upscaler` lives in `core`
because the pipeline's own tests need mock implementations and must run with no model;
`Model` lives in `infer` because it is the shape of the ONNX graph and nothing in
`core` may know about tensors.

`core` must compile and test with no GPU, no audio device, and no model. That is
what makes the project developable on a laptop, in CI, and by someone who has not
downloaded 3.3 GB.

Resist adding crates. `codec/`, `cache/`, `probe/`, `ui/` are modules inside `app`
until something else needs them.

## Stages

### 1. Decode

`symphonia` to `f32`, planar, **48 kHz** — the model's native rate. Resample once,
at the source, with `rubato` (sinc, high quality). 44.1 kHz sources are the common
case and the resampler is on the critical path, so measure it: it must cost well
under 1% of a core or it belongs on the render thread.

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
| render | windower, `ort` session, cache writer | the only thread that calls the model |
| audio callback | `cpal` output, mixer, delay line | **alloc-free, lock-free, no logging, no GPU** |
| UI | egui, probes, file dialogs | reads atomics, sends commands |
| download | model fetch, checksum | `ureq`, cancellable, resumable |

Data flow between them:

- source ring: SPSC (`rtrb`), decode → render.
- output ring: SPSC, render → audio callback.
- control: `crossbeam-channel` for commands (seek, open file, toggle) and events
  (underrun, render progress) — never in the callback.
- meters: `AtomicU64`/`AtomicU32` only, read by the UI at 30 Hz.

The audio callback's only job is to pop samples from a ring, apply the A/B crossfade
and the master gain, and write them to the device. If the ring is empty, it writes
silence and increments a dropout counter. It never waits for anything.

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

**The window-size cost.** Supporting 0.64 / 1.28 / 2.56 / 5.12 s at full quality
means either dynamic axes (CoreML EP handles these poorly) or four copies of a
3.3 GB model. So v1 ships one window size, chosen by measurement, and the latency
presets arrive only if dynamic axes survive the spike. The PoC's flexibility was
free because the Python runtime was already there; here it costs gigabytes. Decide
this in v0.0, not in v0.4.

**Verification gate.** Export is not done when it runs; it is done when it matches
PyTorch. Take five tracks, render them with a fixed seed through both paths, and
require ≤ −40 dB relative RMS difference per window, plus an identical cutoff
decision per window. Below that the export is wrong, whatever it sounds like.

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
   the stage most people miss when reimplementing. `rubato` is not
   `resample_poly`; the filters differ. Either build the polyphase FIR to match
   scipy, or accept a documented deviation and validate it with the eval harness.

Do **not** leave this inside the ONNX graph. `lowpass_input=True` in Python runs a
scipy filter with a cutoff computed per window; exporting it bakes one cutoff into
the graph and makes the model unaware of the actual source. Preprocess in Rust, feed
the model `lowpass_input=False`.

## Cache

v0.3. A render cache turns a fragile real-time problem into a file-management
problem, and it buys seek, instant replay, and a zero-GPU second listen.

- **Key**: `blake3(source bytes) + blake3(weights) + pipeline version + window +
  overlap + dtype + seed policy`. Any change invalidates.
- **Layout**: `cache/<key>/manifest.json` plus `seg-000001.flac` … in 30 s segments.
  Segments let a partial render be playable and a settings change invalidate only
  what it must.
- **Format**: FLAC, 24-bit. This is what FLAC is for: lossless storage of the
  generated signal. Output f32 rounded to 24-bit is transparent (−144 dB quantisation
  floor) and roughly half the size of raw float.
- **Playback from cache**: two readers (original 48 kHz and enhanced) at the same
  position. Toggle is a crossfade with `D = 0`. Unrendered regions fall back to
  passthrough and are drawn differently on the waveform.
- **Eviction**: LRU against a user-set cap (default 20 GB), never evict the file
  currently open, always keep the manifest.
- **Writer**: append to the current segment, flush on segment close, write the
  manifest last with `fsync`. A crash leaves playable segments and no manifest entry;
  the next run adopts them.

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
- **Transport**: play/pause, seek, A/B, gain. No playlist.
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
| ONNX | PyTorch vs `ort`, five tracks, ≤ −40 dB, fixed seed |
| audio device | `cpal` null/loopback device in CI, real device manually |
| UI | `egui_kittest` for the state machine, manual for looks |

Fuzz the windower with `proptest` over lengths 0…3 windows and every overlap value;
the off-by-one at the final partial window is where this class of code always breaks.

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
