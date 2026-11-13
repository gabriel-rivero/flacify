# Decisions

Status: draft, 2026-11-12. One page per decision: context, decision, consequences.
Reopen a decision by adding a superseding ADR, not by editing the old one.

---

## ADR-001 — Render ahead first, bound the buffer later

**Status:** accepted (reversible)
**Context:** The PoC streams in real time and measures RTF 0.37 on an RTX 4060 Ti.
The plan assumed the player must upscale in lockstep with playback, and asked how
many seconds of buffer a song needs.

For a local file the producer is not rate-limited: the decoder can hand over 5.12 s
instantly, so the renderer produces audio as fast as the GPU allows. Waiting for
wall-clock time to pass before rendering is added latency with no benefit. The
required buffer is set by jitter, not by song length, and for `RTF > 1` no buffer
size saves a whole track — it can only cover `T(r−1)/r` of it.

**Decision:** v0.1–v0.3 use render-ahead with a 30 s cap and backpressure. The
bounded real-time queue (the PoC's mode) arrives in v0.4 as a policy on the same
pipeline, for live capture and the system sink.

**Consequences:** Faster first audio than streaming, free seek, no dropout class of
bug in the MVP. Memory cost 11.5 MB. The "real-time" constraint is deferred but not
dropped; the code path is shared, so it stays honest.

---

## ADR-002 — ONNX Runtime as the inference backend

**Status:** **reopened** by its own spike (2026-11-12). The numeric half passed, the
throughput half failed on every configuration tried.
**Context:** The PoC needs Python, torch, librosa, scipy and 3.3 GB of weights. A
player that installs its own dependencies cannot ship that.

**Decision (as written before the spike):** export one static-shape ONNX graph and run
it through `ort`, with CoreML EP on macOS and DirectML/CUDA on Windows and Linux.
Alternative runtimes are measured in the same spike (candle + Metal, MLX) and this ADR
is rewritten with the numbers.

**Consequences:** No Python at runtime; ~40 MB dylib instead of ~7 GB. New risk: EP
coverage and per-node CPU fallback. The export becomes a build artifact that must be
verified numerically, not just executed.

**Outcome of the spike (M5 Max, 2026-11-12, numbers in `measurements.md`):**

| criterion | target | measured |
|---|---|---|
| ONNX vs PyTorch, per window | ≤ −40 dB | **−74.07 dB** on the export fixture, **−47.78 dB** worst of four tracks: pass |
| sustained RTF | ≤ 1.5 | 21.5 (CPU EP), 31.1 (CoreML fp32), 21.7 (CoreML fp16), 15.2 (CoreML fp16 MLProgram): **fail** |
| same weights under PyTorch/Metal | — | RTF **0.51**: the hardware is not the problem |

The graph is numerically sound and executable, which was the export question. What fails
is ONNX Runtime itself on this hardware: it is 30 to 60x slower than PyTorch, and the
CoreML EP largely falls back to CPU. The decision therefore stands open: either ONNX
Runtime is dropped for a runtime that can reach Metal (libtorch via `tch`, candle, MLX),
or the graph is split so the heavy convolutions are not interleaved with the ops CoreML
refuses. Backlog E1-4 to E1-6 carry the options; v0.0 does not pass until one of them
does.

---

## ADR-003 — FLAC is the cache format, not the promise

**Status:** accepted
**Context:** "Upscale MP3 to FLAC" reads as "convert lossy to lossless", which is
impossible and is the first thing a knowledgeable listener will attack.

**Decision:** the product claim is bandwidth extension; FLAC is the lossless container
for the generated 48 kHz signal, used for the render cache and explicit exports. The
player itself plays PCM.

**Consequences:** the pitch survives contact with an audiophile. The UI never says
"lossless" next to a lossy source. The cache gets a format that is seekable,
transparent and roughly half the size of raw float.

---

## ADR-004 — Rust runs the app, Python stays in `tools/`

**Status:** accepted
**Context:** Rust buys a single binary, fine-grained audio control and safe DSP. The
model ecosystem, the export path and the quality metrics are Python.

**Decision:** Rust owns decode, DSP, playback, cache and UI. Python owns ONNX export,
golden fixtures and eval reports. The boundary is files and JSON, never a subprocess
at runtime.

**Consequences:** no Python in the shipped binary; the eval harness is never
blocked on Rust progress; the export environment gets pinned in a container because
torch drifts.

---

## ADR-005 — `egui` for the UI

**Status:** proposed
**Context:** The look is a compact Winamp-style player with a spectrum analyser.
Options: `egui` (immediate mode, custom-painted, single binary), `iced` (retained,
nicer for complex apps), Tauri (web UI, best aesthetics, webview dependency).

**Decision:** `egui`. It renders a bitmap-style skin easily, ships as one binary, and
has no JS toolchain. Winamp `.wsz` skin support is a v2 easter egg, not a requirement.

**Consequences:** custom chrome and hand-painted widgets cost more than a web UI;
accessibility is weaker; the spectrum and waveform are straightforward with
`egui_plot` or a custom painter.

---

## ADR-006 — Noise is a graph input, seeded deterministically

**Status:** accepted
**Context:** the one-step sampler starts from `torch.randn`, so every render is a
different take. Caches, bug reports and blind A/B all need the same audio twice.

**Decision:** the exported graph takes the initial noise as a tensor. Rust generates
it from `blake3(file_id ‖ window_index ‖ pipeline_version)` and passes it in. "Re-roll"
is an explicit action with a new seed.

**Consequences:** reproducible renders, stable cache keys, replayable bug reports, and
a free future feature (multiple variants of the same window). The seed policy becomes
part of the cache key, so a change invalidates the cache.

---

## ADR-007 — A/B alignment by fixed-depth backpressure

**Status:** accepted
**Context:** the toggle is the product, so a desynced A/B is a lie. The enhanced
branch is ready `D` samples ahead of playback; the original must be delayed by the
same `D` for the switch to be sample-locked.

**Decision:** backpressure the renderer to a fixed queue depth (`cap`), so `D` is
constant, and crossfade between two taps that read the same sample index. If the queue
cannot sustain `cap`, disable A/B with a reason instead of drifting.

**Consequences:** instant, click-free, phase-aligned comparison; no timestamp
arithmetic, no resampler drift. The cost is that a slow GPU temporarily loses the
feature, which is the correct failure.

---

## ADR-008 — Do not bundle the weights

**Status:** accepted
**Context:** `FlashSR_Inference` has no LICENSE file and the weights dataset states no
terms. Redistribution is not granted by default.

**Decision:** the installer ships code only; the model is downloaded from the upstream
host at first run, checksum-verified, with attribution and a licence screen. An issue
is opened upstream asking for explicit terms before v1.0.

**Consequences:** the installer stays small and legally clean; first run needs a
2 GB download, so resume and a mirror are load-bearing. If the upstream answer is
non-commercial, that decision propagates to the whole product.

---

## ADR-010 — The prototype runs the model in a Python sidecar

**Status:** accepted for the prototype, with an exit criterion. Supersedes the ONNX
Runtime commitment in ADR-002 *for v0.0 only*.
**Context:** ADR-002's spike measured the exported graph at RTF 15.2 to 31.1 on every
ONNX Runtime configuration, against 0.51 for the same weights under PyTorch on Metal
(`measurements.md`). The model is fast enough; no runtime we can reach from Rust is.
Waiting for that to be fixed means not having a product to evaluate.

**Decision:** the Rust app stays the app. The model runs in a Python 3.12 process that
the app installs and owns: `uv` fetches the interpreter into the app data directory,
pip installs pinned wheels, the weights download from upstream, and the app spawns a
worker that serves one window per request over a binary pipe (`tools/worker.py`).
Rust keeps everything else: decode, preprocessing, windowing, the seed schedule, the
buffer policy and the A/B alignment.

**Consequences:**

- **First run downloads about 6 GB** (interpreter and wheels ~2.5 GB, weights 3.3 GB).
  That is a product-visible number and the setup screen has to say it before it starts.
- **"No Python at runtime" becomes a v1 requirement, not a prototype requirement.**
  The sidecar is never part of a signed build; it is scaffolding with a deadline.
- The `Model` trait does not change, so the rest of the Rust code cannot tell the
  difference, and swapping the backend later is a single implementation.
- Measured on M5 Max: RTF 0.543 through the pipe, 2 ms of IPC per window, output
  bit-identical to the in-process Metal reference.

**Exit criterion:** a Rust-native backend that reaches Metal at RTF ≤ 1.5 (or an ONNX
path that closes the 30x gap) replaces this before any user-facing release. Until then
the sidecar carries the prototype and no claim in `vision.md` depends on it.

---

## ADR-011 — The prototype plays from pre-rendered buffers

**Status:** **superseded by ADR-012** (2026-11-13). Kept for the reasoning: it was the
right trade to get a player working in a day, and the wrong one to keep.
**Context:** ADR-001 chose render-ahead with a bounded queue, which is still the right
shape for v0.4. But at RTF 0.54 a whole track renders faster than it plays, and a
streaming queue means shared mutable buffers handed to an audio callback under a lock
we are not allowed to take. The first playable build needs the toggle to be right, not
the scheduler.

**Decision:** `play` decodes, analyses and renders the whole track, then plays from two
immutable buffers of the same length. The A/B switch mixes between them at one shared
position with an 8 ms ramp. Controls are atomics; the callback only reads, never blocks,
never allocates.

**Consequences:** sample-locked A/B is a property of the data structure, not of a
synchronisation protocol, and it is testable without a device (six mixer tests). Memory
is 2 × 4 bytes × frames per channel (about 140 MB for a 3-minute track) and the user
waits for the render before pressing play — 95 s for 3 minutes. Both are acceptable for
a prototype and neither is acceptable at v0.2: the queue in ADR-001 replaces this, and
the mixer, the controls and the ramp survive the change unchanged.

---

## ADR-012 — Start on the original, promote to enhanced as the frontier arrives

**Status:** accepted, v0.2.
**Context:** ADR-011 plays from two completed buffers, which means the device opens only
after a whole render: 95 s for a three-minute track at RTF 0.54, measured at 19.5 s for a
5 s clip and 32 s for 30 s. The renderer is 1.84x realtime, so the wait buys nothing that
time would not have bought anyway. The cache hides it only from the second play onward.

**Decision:** playback starts as soon as the file is decoded and the device is open, on the
original signal. The renderer runs ahead into a frontier; the mixer ramps to the enhanced
signal once the frontier passes the playhead, and the A/B control reports which side is
live and whether generated audio exists at the current position. A cache hit skips the
renderer: both signals are complete at once.

**Outcome (measured 2026-11-13).** "Start on the original, promote when it catches up"
does not survive contact with the numbers. Through the pipeline the renderer runs at
**1.28x** realtime, not the worker's 1.84x, and the model load costs **8 s**. Starting
playback at t=0 therefore puts the frontier 8 s behind, and at +0.28 s per second of
playback it never catches up inside a track. Measured: 28 s of frontier against a 30 s
playhead after 30 s.

The decision stands, with one change: **playback waits for two seconds of frontier** and
then starts already enhanced. Measured: 12.6 s to first audio cold (8 s load + the first
window), 0.17 s on a cache hit, toggle live from the first note. The automatic promotion
stays for the case where a slow machine reaches the patience budget first and starts on
the original anyway. The remaining 8 s is the model load; a resident worker (started by
the app, not by each play) removes it, and that is packaging work, not playback work.

**Consequences:**

- **First sound moves from 95 s to 0.17 s (cached) or 12.6 s (cold)**, and the first note
  is already the generated signal. The product is a player from the first frame.
- **The toggle becomes position-aware.** It may only claim "generated" where generated
  audio exists. Seeking ahead of the frontier restarts the pipeline at that window and the
  control says so, which is honest and also the natural cache fill.
- **The producer needs a safe handoff.** The renderer writes while the audio callback
  reads, so the frontier is published as immutable chunks through an SPSC queue rather than
  a shared `Vec` — no unsafe, no lock in the callback.
- **ADR-001 is now actually implemented**, in its top-left cell: file source, producer
  unthrottled. The bounded-queue cell stays for live capture.
- **The alternative was rejected on measurement:** buffering one window and starting
  enhanced outright costs a full window of render (2.8 s) plus a prebuffer before any
  sound, and needs the queue before the first note. Starting on the original costs a decode
  and lets the frontier do the catching up.

---

## Open questions

1. **Runtime endgame.** Backlog E1-4 to E1-8: split the graph so the CoreML EP can take
   the convolutions, or move to libtorch/candle/MLX from Rust. ADR-010 buys the time. The
   Linux measurement (ORT CUDA EP at 0.545 vs torch's 0.453, and half the download) makes a
   split, platform-specific backend the likely end state: torch where Metal is the
   accelerator, ONNX Runtime where CUDA is.
2. **Where the toggle lives in the prototype.** With the sidecar at RTF 0.54 a 3-minute
   track renders in ~95 s, so the toggle can be file-based (render, then A/B between two
   decoded buffers) before the streaming queue exists (ADR-001's v0.4).
