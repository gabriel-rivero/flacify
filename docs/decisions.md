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

## Open questions for you

1. **Runtime.** Accept ADR-002 (ONNX) as the plan of record, or spike candle-Metal
   first because a pure-Rust binary matters more than EP breadth?
2. **MVP shape.** Accept ADR-001 (render-ahead, cache in v0.3), or keep real-time
   streaming as the v0.2 deliverable because that is the thing you want to see work?
3. **UI.** `egui` (one binary, hand-painted) or Tauri (better looking, webview
   dependency, JS toolchain)?
4. **Name.** Keep `flacify` for the app and accept that "flacify" describes the
   export, not the player? Or name the player something else and keep `flacify` as the
   CLI verb (`flacify enhance in.mp3`)?
