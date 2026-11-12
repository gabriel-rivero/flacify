# Measurements

Every number this repo quotes, with the machine, the command that produced it and the
date. If a number is not here, it is either borrowed (and says so) or it does not exist.

## Machines

| # | what | specs |
|---|---|---|
| M1 | development laptop (2026-11-12, every number below except P1's) | Apple M5 Max, 32-core GPU, 36 GB, macOS 27.2.1, Rust 1.99.1, Python 3.12.13, torch 2.15.0, onnxruntime 1.31.1 |
| P1 | the PoC's machine (PoC numbers only) | RTX 4060 Ti 8 GB, Ryzen 9 5900XT, 39 GB, Linux, torch 2.15.0+cu132 |

## Preprocessing port (Rust vs scipy)

The Rust reimplementation of FlashSR's two preprocessing stages is pinned against
`tools/fixtures/gen.py`, which mirrors upstream with numpy/scipy. Run
`cargo test -p flacify-core --test fixtures -- --nocapture` for the table.

| metric | value | source |
|---|---|---|
| worst max-error, either stage, all four fixtures | **−132.0 dB** (6.98e-8) | M1, 2026-11-12 |
| best max-error | −134.7 dB (5.68e-8) | M1, 2026-11-12 |
| f32 is the floor | the fixtures are f32, so ~−132 dB is rounding, not port error | — |

The two stages were first validated in Python against scipy's own implementations, then
ported:

| check | agreement |
|---|---|
| `cheby1` SOS matrix (`cheb1ap` → `bilinear_zpk` → `zpk2sos`, order 8, 0.1 dB) | 3e-16 |
| `sosfilt_zi` initial conditions | 0 |
| `sosfiltfilt` (zero-phase, 3× padding) | 1.3e-14 |
| `resample_poly` round trip | 1e-15 |

| fixture | detected cutoff | high-band energy above 21 kHz |
|---|---|---|
| `music_cut16k` | 15328 Hz | 0.0000 % |
| `music_cut8k` | 7664 Hz | 0.0000 % |
| `music_cut20k` | 19148 Hz | 0.0000 % |
| `music_fullband` | 22828 Hz | 4.4541 % |

The last column is why `Analysis::looks_full_band` uses `ratio ≥ 0.01` instead of
`cutoff ≥ 24000` (backlog E8-6): four decades of headroom between a real full-band
source and a band-limited one.

## Test suite

| metric | value | source |
|---|---|---|
| `cargo test --workspace` | 39 passed, 0 failed | M1, 2026-11-12 |
| breakdown | 19 core lib, 1 fixture, 11 pipeline, 8 infer | same |

## The ONNX export (E1)

`tools/export_onnx.py` ran for the first time on 2026-11-12 against the real weights.
It exports with torch's **dynamo** exporter, because the TorchScript exporter serialises
the whole `ModelProto` through protobuf and this graph is 2.46 GB of weights, past the
2 GB cap. Weights go to a sidecar `flashsr.onnx.data`.

| metric | value |
|---|---|
| export gate, fixture window (`music_cut16k`) | **−74.07 dB** vs eager PyTorch (gate −40) |
| the explicit-noise wrapper itself | −289.8 dB vs upstream `FlashSR.forward` |
| graph | opset 18, 4706 nodes, 8.5 MB graph + 2.46 GB external data |
| inputs | `cond` f32[2, 245760], `noise` f32[2, 16, 64, 32] |
| output | f32[2, 245760], finite |
| random ops left in the graph | none |
| ops a backend may refuse | `STFT` (the mel front-end) |
| VAE encode | posterior mean; `DiagonalGaussianDistribution.sample` patched |

Per-track comparison, `tools/compare_onnx.py`, one window from each fixture:

| track | cutoff | ONNX vs PyTorch |
|---|---|---|
| `music_cut16k` | 15328 Hz | −70.26 dB |
| `music_cut20k` | 19148 Hz | −91.10 dB |
| `music_cut8k` | 7640 Hz | **−47.78 dB** (worst) |
| `music_fullband` | 22851 Hz | −101.87 dB |

All four pass the −40 dB gate; the worst has 7.8 dB of margin. E1-3 asks for five real
tracks, and these are four synthetic fixtures, so `compare_onnx.py` reports the whole
gate as not passed (`tracks_required: 5`) even though every row passes. The gate is
advisory until E8-1's corpus exists.

## Throughput on this hardware (the runtime gate)

One 5.12 s stereo window, medians of repeated runs, same weights throughout.

| path | seconds / window | RTF | gate ≤ 1.5 |
|---|---|---|---|
| ONNX Runtime, CPU EP | 110.0 | **21.5** | fail |
| ONNX Runtime, CoreML EP, fp32, NeuralNetwork | 159.4 | **31.1** | fail |
| ONNX Runtime, CoreML EP, fp16, NeuralNetwork | 110.9 | **21.7** | fail |
| ONNX Runtime, CoreML EP, fp16, MLProgram, ALL units | 78.4 | **15.2** | fail |
| PyTorch eager, CPU | 31.3 | **6.1** | fail |
| PyTorch eager, MPS | 2.59 | **0.51** | **pass** |

Two facts matter here:

1. **The hardware is fast enough.** The same model, same weights, runs a window in
   2.59 s under PyTorch on Metal (RTF 0.51), which is 3x inside the gate.
2. **The ONNX Runtime path is 30 to 60x slower than PyTorch on the same machine**, and
   the CoreML execution provider never really engaged: with fp16 under the default
   NeuralNetwork format it produced load 2.2 s and exactly the CPU EP's 110.9 s, i.e. it
   fell back to CPU for the whole graph. MLProgram + ALL compute units is the best
   ONNX configuration measured and still 10x short.

So ADR-002's numeric gate passes and its throughput gate fails. The runtime choice, not
the model, is what failed.

## The Python sidecar (prototype runtime, ADR-010)

`tools/worker.py` keeps the model resident and serves one window per request over a
binary pipe; `tools/worker_bench.py` drives it exactly the way the Rust client will.

| metric | value |
|---|---|
| steady state, 12 consecutive windows of real fixture audio | 2.78 s per window, **RTF 0.543** |
| compute portion (the worker's own per-window log) | 2.78 s |
| IPC for a 2 MB request plus a 2 MB reply | **2 ms**, about 0.07 % of a window |
| interpreter start, import and weight load | 8.2 s |
| first audio (load + warmup + first window) | 12.7 s |
| worker RSS as `ps` reports it | 870 MB (MPS allocations are not fully visible to `ps`; weights are 3.3 GB on disk) |
| worker output vs in-process Metal reference, same window and noise | **bit-identical**, max abs diff 0.0 |

The last row is the acceptance test for the sidecar and it is the one that matters:
`tools/worker_check.py` must stay at zero difference, because the worker changes *where*
the model runs, never *what* it computes. RTF 0.543 is 2.8x inside the gate, and the
process boundary costs 2 ms.

Borrowed from P1, and **not** transferable to M1:

| number | value | source |
|---|---|---|
| FlashSR RTF, 5.12 s stereo, bf16 | 0.37 (1.889 s per window) | P1, `flashsr-stream` |
| peak VRAM | 3.69 GB | P1 |

## `flacify probe` arithmetic

The policy is arithmetic on the measured RTF, so these are exact given the input; the
RTF used below is the CLI's placeholder, not a measurement.

| input | RTF used | prebuffer | wait |
|---|---|---|---|
| 179.2 s track | 8.7 (assumed, the PoC's CPU number) | 158.6 s of audio | 1380 s |
| 5.12 s track | 8.7 (assumed) | 5.1 s of audio | 45 s |
| 5.12 s track | 0.3 (CLI default until the probe exists) | 5.1 s of audio | 1.5 s |

`--rtf` defaults to 0.3 and says so in `flacify probe --help`.
