# Measurements

Every number this repo quotes, with the machine, the command that produced it and the
date. If a number is not here, it is either borrowed (and says so) or it does not exist.

## Machines

| # | what | specs |
|---|---|---|
| M1 | development laptop (2026-11-12, every number below except P1's, M2's, W1's and W2's) | Apple M5 Max, 32-core GPU, 36 GB, macOS 27.2.1, Rust 1.99.1, Python 3.12.13, torch 2.15.0, onnxruntime 1.31.1 |
| P1 | the PoC's machine (PoC numbers only) | RTX 4060 Ti 8 GB, Ryzen 9 5900XT, 39 GB, Linux, torch 2.15.0+cu132 |
| M2 | the same RTX 4060 Ti box, running flacify (Linux numbers) | CachyOS, driver 620.70.06 (CUDA 13.5), Ryzen 9 5900XT, 39 GB, Rust 1.99.1, Python 3.12.16, torch 2.15.0+cu132, onnxruntime-gpu 1.31.1 |
| W1 | this Windows handheld (Windows numbers) | Ryzen Z1 Extreme, 8 cores / 16 threads, Radeon integrated graphics, 23.2 GB, Windows 11 Pro 26200, Rust 1.100.0 (MSVC), no CUDA |
| W2 | the RTX 4060 Ti box's Windows install (the 2026-11-15 numbers) | Windows 11 Home 10.0.26300, driver 621.01, Ryzen 9 5900XT, 16 cores / 32 threads, 39.9 GB, RTX 4060 Ti 8 GB, Rust 1.99.1 (MSVC), Python 3.12.14, torch 2.15.0+cu132 (CUDA 13.2), onnxruntime 1.32.0 |

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

The first number is the suite as it was at the start; the second is what
`cargo test --workspace` lists today, which `tools/check_docs.py --tests` compares
against the count in `README.md`.

| metric | value | source |
|---|---|---|
| `cargo test --workspace` at v0.0 | 39 passed, 0 failed | M1, 2026-11-12 |
| breakdown at v0.0 | 19 core lib, 1 fixture, 11 pipeline, 8 infer | same |
| `cargo test --workspace --all-targets` at v1.0 | 481 tests, 0 failed | M1, 2026-11-18 |
| breakdown at v1.0 | 62 core, 34 infer, 385 app (3 are `cfg(unix)`, so a Windows run counts fewer) | same |
| `cargo test --workspace --all-targets` today | 511 tests, 0 failed | M1, 2026-11-27 |
| breakdown today | 62 core, 34 infer, 415 app | same |

The total counts a test once per target it is compiled into: `process.rs` and `setup.rs`
are each mounted by `#[path]` in `tests/setup.rs` as well as built into the binary, so 14
of those tests run twice and the number overstates unique coverage.

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

## End to end, through the app (2026-11-12)

A 30.72 s stereo 48 kHz clip (the `music_cut16k` fixture tiled six times), rendered with
`flacify enhance` and played with `flacify play --smoke`:

| metric | value |
|---|---|
| render | 34.1 s wall for 30.7 s of audio, including ~8 s of worker start (steady RTF ≈ 0.85 with decode and preprocessing included) |
| output length | 1,474,560 frames, identical to the input |
| generated band vs input | +24.6 dB at 16–18 kHz, +61.6 dB at 18–22 kHz, +54.4 dB at 22–24 kHz |
| level | +0.57 dB (no loudness match yet: roadmap gap 2) |
| low band, level matched | −0.8 dB at 0.1–1 kHz, −0.3 dB at 1–4 kHz, +2.9 dB at 4–12 kHz, but −18.7 dB RMS waveform difference below 12 kHz (the model regenerates, it does not splice: gap 3) |
| playback | device 48 kHz × 2 ch, 57,344 frames (1.19 s) pulled in 1.2 s, switch enabled, no underrun |

`play --smoke` is the headless proof that the callback runs and the playhead moves; the
window itself is not verified visually here (the agent has no screen-recording
permission), only that the process runs clean.

## Real music: three MP3s from a user's library (2026-11-12)

Synthetic fixtures do not overshoot, do not have 44.1 kHz sources and do not have real
spectra. These three do:

| track | bitrate | source | detected cutoff | energy above 21 kHz |
|---|---|---|---|---|
| EP Hope Is the Thing With Feathers | 192 kbps | 44.1 kHz stereo, 175.6 s | 14062 Hz | 0.0256 % |
| PROI PROI | 126 kbps | 44.1 kHz stereo, 235.1 s | 13734 Hz | 0.0006 % |
| Weight of the World | 192 kbps | 44.1 kHz stereo, 344.2 s | 12632 Hz | 0.0009 % |

All three are band-limited by a wide margin (the full-band threshold is 1 %), so all
three get enhanced. The 44.1 → 48 kHz resample path is exercised for the first time and
the duration is preserved to the frame.

Render, first track: **136.9 s of wall clock for 175.6 s of audio**, including the 8 s
worker start. That is RTF 0.78, against 0.543 for the worker alone: the difference is
decode, 44.1 → 48 kHz resampling, the Chebyshev preprocess and the WAV write.

And the finding that matters: **the model overshoots full scale.** Peak was **1.36**
(2.7 dB over), with 10,014 samples past ±1.0 in a 2-minute track. Nothing was limiting
it, so playback would have clipped. `limit.rs` now applies a soft knee above 0.9:
after it, peak **0.999**, zero samples at or past full scale, 53,294 samples shaped
(0.6 % of the file) and RMS 0.2712 instead of 0.2714 — the level did not move.

### A clean install, end to end (2026-11-12)

`flacify setup` on a machine with no Python, no torch and no weights, from a release
build, then `play --smoke` on a 5.12 s clip:

| stage | result |
|---|---|
| `setup`, first run | all seven steps, about 5.9 GB downloaded |
| `setup`, second run | nothing to do, 0.7 s |
| `play --smoke`, cold (first run after install) | 38.1 s wall, 13.5 s CPU |
| `play --smoke`, warm | 16.5 s wall for 5.12 s of audio (load ~8 s, render ~4 s, playback 1.2 s) |
| output | 48 kHz × 2 ch, 57,344 frames pulled in 1.2 s, switch enabled |

Two bugs only a clean install could find:

1. **The inference package cannot be pip-installed.** Its `setup.py` uses
   `find_packages`, which skips directories without an `__init__.py`, and
   `TorchJaekwon/Model` is one of them, so the worker died on
   `No module named 'TorchJaekwon.Model'`. The setup step now downloads the source archive
   and puts the tree on `sys.path`, which is what the dev checkout always used.
2. **The wheels list was missing `matplotlib` and `einops`.** Upstream imports matplotlib
   at module load (`UtilTorch.py`) and einops inside the UNet, so the worker died on the
   first import either way. Both are in the list now, and the wheels step records the list
   as a manifest so changing it re-runs the step on an existing install instead of being
   silently skipped.

| | before the limiter | after |
|---|---|---|
| peak | 1.3622 | 0.999 |
| samples ≥ full scale | 10,014 | 0 |
| RMS | 0.2714 | 0.2712 |
| samples the knee touched | — | 53,294 (0.6 %) |

Borrowed from P1, and **not** transferable to M1:

| number | value | source |
|---|---|---|
| FlashSR RTF, 5.12 s stereo, bf16 | 0.37 (1.889 s per window) | P1, `flashsr-stream` |
| peak VRAM | 3.69 GB | P1 |

## Linux: the download, and whether ONNX Runtime can shrink it (2026-11-12)

The macOS probes rejected the ONNX path on throughput, so the full torch install was
the only viable one there. Linux has a different accelerator, so the same question was
asked again on M2, the RTX 4060 Ti box.

`flacify setup` on this machine downloads **about 6.6 GB**: 3.20 GB of compressed
wheels (5.9 GB installed), the 3.32 GB of `.pth` weights, and ~0.15 GB of
Python/uv/sources. That is more than the 5.9 GB the setup screen prints and more than
macOS, because PyPI's torch pulls the whole CUDA 13 stack (cublas 405 MB, cuDNN
519 MB, triton 286 MB, nccl 216 MB, and the rest).

The exported graph (`tools/export_onnx.py`) is unchanged from the macOS one: opset 18,
4706 nodes, 8.6 MB graph + 2.47 GB external data, export gate **−56.4 dB** on one
fixture window (gate −40). The runtimes on that graph, same weights, same GPU:

| path | load | s/window | RTF | gate ≤ 1.5 |
|---|---|---|---|---|
| PyTorch 2.15 + CUDA, resident worker (`worker_bench.py`) | 9.6 s | 2.32 | **0.453** | pass |
| ONNX Runtime 1.31.1, CUDA EP, fp32 graph | 1.8 s | 2.79 | **0.545** | pass |
| ONNX Runtime 1.31.1, CUDA EP, fp16 graph | 1.2 s | 1.78 | **0.348** | pass (numerics fail) |

The sidecar is bit-identical to in-process CUDA here too: `worker_check.py --device
cuda` reports **−290.6 dB, max abs diff 0.0**. `play --smoke` opens a 44.1 kHz × 2 ch
device, resamples to 48 kHz and plays with the switch enabled; a 12.0 s clip takes
21.9 s wall including the ~10 s worker start, and a 345 s track renders and plays in
221 s (RTF 0.64 including decode and preprocessing).

The numeric gate on the four fixtures, PyTorch TF32 off, fixed seed:

| graph | worst track | worst dB | gate −40 |
|---|---|---|---|
| fp32 | `music_cut8k` | **−41.70** | pass, 1.7 dB margin |
| fp16 | `music_cut8k` | **−10.20** | **fail** |

Whole-graph fp16 conversion is not a download lever: it halves the weights to
1.24 GB but changes the signal past the gate. A mixed-precision graph (fp16 only where
safe) might work; it is not a one-line conversion.

**What switching runtimes would save.** ONNX Runtime CUDA removes the 3.2 GB torch
wheel stack and replaces the 3.32 GB of `.pth` weights with the 2.47 GB exported
graph:

| deployment | wheels | weights | runtime | first-run total |
|---|---|---|---|---|
| torch sidecar (today) | 3.20 GB | 3.32 GB | 0.15 GB | **~6.6 GB** |
| ONNX Runtime GPU, pip CUDA extras | 1.08 GB | 2.47 GB | 0.05 GB | **~3.6 GB** |
| ONNX Runtime GPU, system CUDA/cuDNN | 0.26 GB | 2.47 GB | 0.05 GB | **~2.8 GB** |
| Rust `ort` (no Python) | 0.04 GB | 2.47 GB | — | **~2.5 GB** |

The pip extras are `nvidia-cuda-runtime`, `nvidia-cuda-nvrtc`, `nvidia-cufft`,
`nvidia-curand` and `nvidia-cudnn-cu13`, 0.82 GB together; a machine with CUDA 13 and
cuDNN 9 already installed does not need them. The weights are now the floor, which is
why fp16 was worth testing.

The trade is 20 % slower than torch on the same GPU (0.545 vs 0.453), still 2.75×
inside the gate. On macOS the same runtime is 15–31 RTF and the trade does not exist.
That is the platform split: Linux can roughly halve the download, macOS cannot.
Windows is measured below: its CPU EP is 33 RTF and DirectML cannot load the graph.

## Windows: build and test suite (2026-11-12)

Windows had only ever been compiled by CI. This is the first native run on a real
device, and the machine has no CUDA (Radeon integrated graphics), so `auto` resolves
to `cpu`. The build and test rows are model-free; the `play --smoke` row is the
end-to-end path.

| metric | value | source |
|---|---|---|
| cold `cargo build --release` (every dependency, MSVC) | 75 s | W1, 2026-11-12 |
| warm `cargo build --release` (one file changed) | 17.7 s | W1, 2026-11-12 |
| `cargo test --workspace`, warm | **91 passed, 0 failed**, 1.5 s | W1, 2026-11-12 |
| breakdown | 19 core lib, 1 fixture, 11 pipeline, 13 app lib, 9 decode, 12 setup, 14 infer lib, 3 fake worker, 9 worker IPC | W1, 2026-11-12 |
| `flacify setup`, first run | all seven steps, **4.2 GiB on disk** (weights 3.09, venv 0.93, runtime 0.16) | W1, 2026-11-12 |
| `flacify setup`, second run | 0.1 s, no step executes | W1, 2026-11-12 |
| `flacify play --smoke`, 5.12 s clip | 152.1 s wall (load + render + 1.2 s of playback), **48 kHz × 2 ch**, 58,656 frames pulled in 1.2 s, switch enabled, `model cpu` | W1, 2026-11-12 |
| toolchain | rustc 1.100.0, stable-x86_64-pc-windows-msvc | W1, 2026-11-12 |

The suite is 91 tests at this commit; the README's 87 predates the Windows-port
fixes and the one added below.

Windows is a renderer on this machine, not a player: 152 s of wall clock for one
5.12 s window, load included, is far outside the 1.5 RTF gate and that is the CPU
path the README already describes. It became a player on W2 once torch came from the
CUDA index; the [2026-11-15 section](#windows-cuda-and-the-4060-ti-as-a-player-2026-11-15)
has that run. DirectML through ONNX Runtime was the candidate for a
Windows accelerator; it cannot load this graph, and the reason is below. Setup is
smaller than on the other platforms
(4.2 GiB vs the 5.9 GB the screen prints) because PyPI's Windows torch is CPU-only,
so the wheels are ~0.9 GiB instead of the 2.5 GB the plan reserves for CUDA.

**Smart App Control blocks every local build.** This Windows 11 install had Smart App
Control enforced (`VerifiedAndReputablePolicyState = 1`), and the first build died in
`khronos_api`'s build script with `An Application Control policy has blocked this file.
(os error 4551)`; CodeIntegrity logged policy `{0283ac0f-…}`. SAC has no exclusion
list, and it would also block the `uv.exe`, managed CPython and wheels that
`flacify setup` installs, so the build and test numbers above were taken after
Settings → App & browser control → Smart App Control → Off and a reboot. That toggle
is one-way: re-enabling it needs a Windows reset.

A clean first run also found a provisioning bug that macOS never hit, because macOS
had `uv` on `PATH` and skipped the download. `install_uv` hands `curl -o` a path under
`runtime/`, and `tar -C runtime/bin`, before either directory exists, so on a machine
with no `uv` the `uv` step fails with `curl: (23) client returned ERROR on write`.
The step now creates those directories first, and a test covers the download branch
with no `uv` on `PATH`.

## Windows: the download, and whether ONNX Runtime can shrink it (2026-11-12)

Windows starts from a different number than macOS and Linux: PyPI's Windows torch is
CPU-only, so the wheel stack is already small and the `.pth` weights are most of the
download.

| component (first run, W1) | download | on disk |
|---|---|---|
| three `.pth` weights | 3.32 GB | 3.09 GiB |
| 66 distributions in the venv | **294 MB** | 0.93 GiB |
| CPython 3.12 (installed by uv) | 63 MB | 63 MB |
| FlashSR source archive | 53 MB | 53 MB |
| uv.exe and the kept uv.zip | 46 MB | 46 MB |
| **total** | **~3.8 GB** | **4.17 GiB** |

The wheels, largest first: torch 127 MB, llvmlite 42, scipy 37, onnxruntime 15, numpy
13, matplotlib 9, scikit-learn 8, onnx 8, pillow 7, sympy 6; the other 56 distributions
are 21 MB together. So Windows is already ~1.9 GB below the 5.9 GB the setup screen
prints, because that figure reserves 2.5 GB for the CUDA wheels a CPU-only Windows
install never fetches. Dropping the keep of `uv.zip` and trimming the FlashSR tree are
noise next to the weights; the librosa chain (llvmlite, numba, scikit-learn,
matplotlib, ~63 MB) cannot go without patching upstream, which imports matplotlib and
scikit-learn at module load.

The exported graph is the same one macOS and Linux produced — dynamo, opset 18, 4706
nodes — at **8.6 MB graph + 2.46 GB external data**, 2,472,256,261 bytes in total.
Exported on W1 against the `music_cut16k` fixture, the gate is **−66.7 dB** against
eager torch (gate −40); the four-track comparison was not re-run on Windows because
each track costs a torch window plus an ONNX window on CPU, and the conversion is
architecture-independent (−41.70 dB worst on M2, fp32).

| deployment | wheels | weights | python + uv | first-run total |
|---|---|---|---|---|
| torch sidecar (today) | 294 MB | 3.32 GB | 162 MB | **~3.8 GB** |
| ONNX Runtime, Python sidecar | ~35 MB | 2.46 GB | 109 MB | **~2.6 GB** |
| Rust `ort` (no Python) | — | 2.46 GB | — | **~2.5 GB** |

That is **~1.2 GB less (31 %)** for the Python path and **~1.3 GB less (34 %)** for the
Rust one. Whole-graph fp16 halves the weights again to 1.24 GB, but it fails the
numeric gate on Linux (−10.20 dB on the worst track), and that conversion does not
depend on the OS, so it was not retried here.

The catch is throughput, and on Windows it is the same catch as on macOS:

| path, 5.12 s stereo window | load | s/window | RTF |
|---|---|---|---|
| PyTorch 2.15, CPU, resident worker | 12.5 s | 39.40 | **7.70** |
| ONNX Runtime 1.31.1, CPU EP | 5.0 s | 169.14 | **33.0** |

The first row is W1's CPU-only Windows result. On W2, once torch came from PyTorch's
CUDA index, the same model ran on the 4060 Ti at RTF 0.66; see
[Windows: CUDA](#windows-cuda-and-the-4060-ti-as-a-player-2026-11-15) below. Everything
in this ONNX section was measured on W1 and is unchanged.

ONNX Runtime's CPU EP is 4.3x slower than torch's CPU path on this machine, against
3.5x on the M5 Max, so on a CPU-only Windows box the smaller download buys a much worse
renderer. The trade is only worth it alongside a GPU execution provider, and on Windows
that is DirectML — tried here, and it cannot load this graph.
`onnxruntime-directml` installs cleanly at 25.1 MB (137 MB on disk), but its newest
release is ORT 1.24.4, seven minor versions behind the 1.31.1 the CPU package ships, and
`DmlExecutionProvider` dies in session init on the exported graph:

    RUNTIME_EXCEPTION: Exception during initialization:
    .../DmlExecutionProvider/src/MLOperatorAuthorImpl.cpp(2818) ... 80070057
    The parameter is incorrect.

That is `onnxruntime-directml==1.24.4` in a scratch venv plus
`python tools/bench_onnx.py --provider directml`; `tools/bench_onnx.py` needs no change
to reach it, because the failure is before any timing starts.

The blocker is 1D transposed convolution. The vocoder is 94 `ConvTranspose` nodes and the
EP claims them: a one-node model whose only op is a 1D `ConvTranspose` fails identically,
while its 2D twin, a plain 1D `Conv`, a grouped 1D `Conv` and a 4706-node `Conv` chain all
load and run on the DML EP on this machine. Disabling graph optimisation, and disabling
the memory pattern together with switching to sequential execution, change nothing. So
DirectML needs those 94 nodes rewritten as `Resize` plus `Conv` — which the graph already
does six times — or a newer EP that does not exist yet, and no DirectML number exists as a
result. It is not a configuration switch either: `tools/worker.py` imports torch, so
shipping the ONNX path means a second worker or the Rust `ort` work, which is why the
Linux section lists it as future work.

## The render cache (2026-11-13)

`flacify play` and `flacify enhance` write the finished stream to `<data>/cache/<key>/`
and read it back on the next run. Same machine, same tracks, second run only.

| run | wall clock | output |
|---|---|---|
| 30.72 s clip, first (render + store) | 32.0 s | `clip_cache_a.wav` |
| 30.72 s clip, second (cache hit) | **0.1 s** | `clip_cache_b.wav`, **byte-identical** (same SHA-256) |
| 5.12 s clip, `play --smoke`, first | 19.5 s | plays, stores |
| 5.12 s clip, `play --smoke`, second | **1.48 s** | plays from cache, no model, no worker |

The cache entry for the 5.12 s clip is 1.9 MB, which is exactly 245,760 × 2 × 4 bytes: the
cache stores the float stream, not a lossy copy.

## Alignment and the low-band splice (2026-11-13)

Whether frame `k` of the generated signal is frame `k` of the original, measured in three
places:

| stage | test | result |
|---|---|---|
| pipeline (windowing, crossfade) | impulse at 0, 1 000, 197 760, 400 000, 500 000 frames, identity upscaler | comes out at its own index, length exact |
| preprocessing round trip | single-sample impulse through `Preprocessor::process` at three positions | lands on itself; the 62-sample shift first seen was an artefact of using a 64-sample rectangular burst, whose ringing moves the apparent peak |
| the model | band-limited burst at frames 60 000 and 120 000 in one window | **12 samples late both times** (peak); the energy centroid says 24, the expected disagreement for an asymmetric kernel |

The player compensates the model's 12 samples (`Model::latency_samples`), so the two sides
are aligned to 0.25 ms.

**The splice, measured.** `out − low(out) + low(input)`, same zero-phase Chebyshev on both
sides, cutoff 15,351 Hz, on a 30.7 s track:

| variant | correlation with the input's low band (100–6 000 Hz) |
|---|---|
| unspliced | 0.39 |
| spliced | 0.32 |

It does not work. The model's low band is not a delayed copy of the input's: it is
regenerated with its own phase, so subtracting it and adding the input's leaves a mixture of
both rather than the input. The `--splice` flag stays off.

## Loudness matching (2026-11-13)

The generated side used to be **1.49 LU louder** than the original on a steady track, so
the switch changed level as well as brightness. The renderer now measures both sides
(BS.1770, 400 ms blocks, absolute and relative gating), applies one gain derived from the
first window, then limits the peaks.

| measurement | before | after |
|---|---|---|
| integrated loudness step | +1.49 LU | **−0.10 LU** |
| gain applied to the generated side | 0 dB | −1.49 dB |
| RMS step | +0.57 dB | −0.92 dB |

The RMS figure moving the other way is the point, not a fault: at equal loudness, content
that adds energy above 15 kHz is RMS-quieter, because RMS weights that energy far more
than hearing does. Verified twice — once by the app's own meter, once by an independent
scipy implementation of the same specification.

The gain comes from the first 5.12 s and is held for the whole track: a per-window gain
would pump, and a track whose opening is unrepresentative gets a slightly wrong gain
rather than audible modulation. Clamped to ±6 dB, because beyond that the measurement is
lying.

## Progressive playback (2026-11-13)

The renderer fills a frontier behind playback. Two modes, same 30.7 s clip, same machine:

| path | first audio | toggle live | generated when playback starts |
|---|---|---|---|
| cache hit | **0.17 s** | 0.3 s | all 30.7 s, both sides complete |
| cold (first play of a track) | **12.6 s** | 12.7 s | 8 s ready, then growing |

Where the 12.6 s goes: about 8 s is the worker starting and loading 3.3 GB of weights, and
about 4 s is the first window (5.12 s of audio at RTF 0.543 plus the Rust preprocessing).
That is the whole cost of a cold start; nothing waits for the track.

**The renderer runs at 1.28x realtime, not 1.84x.** The worker alone is 0.543 RTF (1.84x),
but through the pipeline the rate is 30.7 s of audio in about 24 s of render after the
load. That matters for the design: starting playback at t=0 makes the frontier fall
behind by the load time and never catch up (it gains 0.28 s per second), so playback
waits for two seconds of frontier instead. Once it starts, the frontier stays ahead.

## `flacify probe` arithmetic

The policy is arithmetic on the measured RTF, so these are exact given the input; the
RTF used below is the CLI's placeholder, not a measurement.

| input | RTF used | prebuffer | wait |
|---|---|---|---|
| 179.2 s track | 8.7 (assumed, the PoC's CPU number) | 158.6 s of audio | 1380 s |
| 5.12 s track | 8.7 (assumed) | 5.1 s of audio | 45 s |
| 5.12 s track | 0.3 (CLI default until the probe exists) | 5.1 s of audio | 1.5 s |

`--rtf` defaults to 0.3 and says so in `flacify probe --help`.

## Windows: CUDA, and the 4060 Ti as a player (2026-11-15)

W1 is CPU-only, so Windows had only ever been a renderer there. W2 is the same 4060 Ti
box M2 runs Linux on, booted into Windows, and the first Windows run that uses the GPU.
The change is one environment variable: `FLACIFY_TORCH_INDEX` pointed at PyTorch's
`cu132` index, one of the two that publish the installed torch (2.15.0, the other being
`cu134`; checked 2026-11-15) and the one that matches M2's Linux install. Nothing in the
code is platform-conditional; the flag is the same one the docs already documented.

| metric | value | source |
|---|---|---|
| CUDA torch, download and install | 2.0 GiB download, 13 min 24 s to download and prepare, 6.3 s to install, `torch 2.15.0+cpu` → `2.15.0+cu132` | W2, 2026-11-15, `uv pip install --upgrade torch --index-url .../cu132` |
| device after the reinstall | `cuda` available, `NVIDIA GeForce RTX 4060 Ti`, capability `(8, 9)`, CUDA runtime 13.2 | W2, 2026-11-15, `torch.cuda.get_device_capability` |
| model load | **21.88 s** (`device=cuda torch=2.15.0+cu132 load_secs=21.88`) | W2, 2026-11-15, `RUST_LOG=info flacify enhance` |
| whole install on disk | **6.90 GiB**: models 3.09, venv 3.45, runtime 0.11, cache 0.25 | W2, 2026-11-15, `Get-ChildItem -Recurse \| Measure-Object` |
| `torch/lib` alone | 2.79 GiB of the venv's 3.45 | W2, 2026-11-15, same |
| 60.06 s clip, cold | **61.3 s wall** for the whole `enhance`, RTF **0.66** once the load is subtracted (≈39.4 s of render) | W2, 2026-11-15, `Measure-Command flacify enhance` |
| 5.0 s clip, cold (one window) | 31.5 s wall: 21.88 s load plus 9.6 s for the first window, which pays CUDA's kernel setup — the 60 s row is the throughput number | W2, 2026-11-15, `flacify enhance` |
| same box, CPU-only torch, 5.12 s clip | 124.5 s wall, `model on cpu` | W2, 2026-11-15, before the CUDA reinstall |
| `play --smoke`, 45 s clip, uncached | first audio **25.27 s**, `generated 8.0 s ready of 45.0 s`, `live` and `promoted` at **25.4 s**, `switch generated`, device 48000 Hz × 2 ch | W2, 2026-11-15 |

RTF 0.66 is the number that matters: below 1.0, so the frontier gains on the playhead and
the A/B switch unlocks by itself. Arithmetic on the measured rate puts the 619.13 s MP3
this run started from at ≈6.8 min of render after the load. On the CPU path that file had
not produced its first generated second before the playhead passed it, which is why the
window showed no generated curve at all.

### The high band, with the same weights (CPU inference)

A 5.0 s clip cut from the same 12 kHz-cut MP3, both sides normalised to the same RMS so
the bands compare, one FFT with a Hann window per side. Measured while the box was still
on CPU torch, before the reinstall; the weights and the pipeline are the same, and the
point of the row is the spectral shape, not the speed.

| band | source | generated | delta |
|---|---|---|---|
| 100–1000 Hz | 61.0 dB | 61.0 dB | −0.0 |
| 1000–8000 Hz | 43.1 dB | 43.4 dB | +0.3 |
| 8000–14000 Hz | 21.7 dB | 27.9 dB | +6.2 |
| 14000–18000 Hz | 7.4 dB | 18.7 dB | **+11.4** |
| 18000–22000 Hz | −38.2 dB | 14.7 dB | **+52.9** |

### Two setup bugs this run found

Both are fixed in `crates/app/src/setup.rs`; the numbers are here because they are the
evidence for the fix.

1. **The index was a no-op on an existing install.** `uv pip install torch` with no
   version spec keeps what is there. Run against the already-installed CPU wheel with
   the `cu132` index, uv answered **"Would make no changes"** (exit 1) — the venv stayed
   on `2.15.0+cpu` while `setup` printed `wheels: done`. The index now comes with
   `--upgrade`; without the index the command is unchanged.
2. **Setting the index made a good install look missing.** `is_ready` compared the
   wheels receipt against the *current* environment, so launching `flacify play` without
   `FLACIFY_TORCH_INDEX` exported again printed *"the model is not installed yet"* on a
   working venv. Readiness now ignores the receipt's one environment-dependent line
   (`torch-index …`); `ensure` still compares the whole manifest, so changing the index
   re-runs the step.

## The FLAC cache container (2026-11-17)

The v0.3 cache is float WAV because `flacenc`'s output did not decode. This is the
measurement that explains it, and the size the segmented cache will actually pay.

**Decoding, on a 3.0 s 48 kHz 24-bit stereo file** (two tones, 441 Hz left and 1.5 kHz
right, so a channel mix-up is visible):

| reader | `flacenc` 0.5.1 output | after the `STREAMINFO` fix |
|---|---|---|
| symphonia 0.5.5 | **cannot probe**: `IoError: end of stream` | 144000/144000 frames, L and R `max_abs = 0.0` |
| claxon 0.4.3 | exact, both channels | exact, both channels |
| libFLAC (`flac -t`) | warning: "sample or frame number does not increase correctly" | ok |

`STREAMINFO` says `min_blocksize = 640`, `max_blocksize = 4096`, because the last frame is
short, while every frame header carries the fixed-blocking-strategy bit. libFLAC writes
min = max = the block size for the same input, and the file then reads everywhere. Calling
`stream_info_mut().set_block_sizes(block, block)` after `encode_with_fixed_block_size` and
before `Stream::write` produces the libFLAC layout through the public API.

**Size and speed, 28.84 s (1 384 320 frames) of 48 kHz stereo**, pink noise, so the
worst case for a lossless codec (M5 Max, 2026-11-17):

| container | bytes | share of float WAV | encode wall time |
|---|---|---|---|
| float WAV (today's cache) | 11 074 560 | 100% | n/a |
| 24-bit WAV | 8 306 022 | 75% | n/a |
| `flacenc` 24-bit FLAC | 6 161 396 | 56% | 0.03 s (~960× realtime, multithreaded) |
| libFLAC 24-bit FLAC | 3 439 904 | 31% | 0.05 s |

Music compresses better than pink noise, so the real cache ratio sits between the two FLAC
rows. Encoding costs under 0.5% of the model's render time either way.

**The writer that shipped (2026-11-17).** `flacify enhance -o x.flac` now uses
`crates/app/src/flac.rs`. Measured on 1 384 320 frames (28.84 s) of white noise through
that writer, release profile, M5 Max:

| quantity | value |
|---|---|
| FLAC bytes | 4 157 946 (ratio 0.50 of 24-bit raw, 0.38 of float WAV) |
| encode wall time | 58 ms, single-threaded, no `par` feature |
| share of the model time for the same audio (RTF 0.54, ≈53 s) | 0.11% |

## Segment playback (2026-11-17)

A complete cache entry plays from three resident segments, each holding both the source
and the rendered side. Measured by building two segment sources over 172 800 000 frames
(one hour at 48 kHz) with the shipped 1 384 320-frame segment and reading them the way the
mixer does, M5 Max, release profile:

| quantity | value |
|---|---|
| resident after a seek | 44 298 240 bytes |
| resident in steady playback | 66 447 360 bytes |
| the same hour as whole-track buffers (before) | 1 380 000 000 bytes per side |

The first play of an uncached track still holds the decoded track and the frontier, so this
figure applies to an entry that is complete.

## Window presets (S1), 2026-11-18

The sidecar takes `channels, frames` per request and derives the latent grid for that
window (`tools/worker.py:109-113`), so 1, 2, 4 and 8 granules are an input it accepts
today. What was unknown is what each window costs and whether the short ones still sound
right. The cost is measured here; the listening verdict is the human's, and the files are
in `tools/out/s1/` with a `README.txt`.

M1 (Apple M5 Max, 32-core GPU, 36 GB, macOS 27.2.1, Python 3.12, torch 2.15.0, MPS),
2026-11-18, the resident worker the app spawns, one worker across every preset and both
tracks. Five timed calls per row after one warmup call at that window size, on the first
30 s of "Weight of the World" (NieR, 192 kbps 44.1 kHz MP3) and "PROI PROI" (Star Rail,
126 kbps 44.1 kHz MP3), decoded and resampled to 48 kHz stereo. The table is the last of
three runs of the same command; the spread between them is a bullet below.
`tools/bench_windows.py` produced it and `tools/out/s1/report.json`.

| track | preset | frames | compute s/call p50 | compute p95 | RTF p50 | RTF p95 | output bytes/call | listening verdict |
|---|---|---|---|---|---|---|---|---|
| NieR | 1 granule | 30 720 | 0.41 | 0.42 | 0.64 | 0.66 | 245 760 | pending |
| NieR | 2 granules | 61 440 | 1.06 | 1.10 | 0.83 | 0.86 | 491 520 | pending |
| NieR | 4 granules | 122 880 | 1.94 | 2.00 | 0.76 | 0.78 | 983 040 | pending |
| NieR | 8 granules | 245 760 | 3.74 | 3.83 | 0.73 | 0.75 | 1 966 080 | pending |
| PROI PROI | 1 granule | 30 720 | 0.51 | 0.63 | 0.80 | 0.98 | 245 760 | pending |
| PROI PROI | 2 granules | 61 440 | 0.99 | 1.04 | 0.77 | 0.81 | 491 520 | pending |
| PROI PROI | 4 granules | 122 880 | 1.72 | 1.79 | 0.67 | 0.70 | 983 040 | pending |
| PROI PROI | 8 granules | 245 760 | 3.57 | 3.65 | 0.70 | 0.71 | 1 966 080 | pending |

What the rows say, before anyone listens:

- **A preset buys latency, not throughput.** RTF p50 sits between 0.62 and 0.80 in 22 of
  the 24 rows measured over three runs; the other two are 0.83 and 0.95, both at short
  windows. What does move with the preset is the cost of one call: 3.2 to 3.7 s at 8
  granules against 0.41 to 0.61 s at 1, so the eightfold shorter window buys a five- to
  ninefold cheaper call without changing the per-second rate.
- **The run-to-run spread is the size of the effect being measured.** Three runs of the
  same command on an otherwise idle machine: the NieR 8-granule p50 was 3.24, 3.48 and
  3.74 s, the NieR 1-granule p50 0.41, 0.44 and 0.61 s, and RTF p50 across all 24 rows
  ranged 0.62 to 0.95. Treat ±15 % as the noise floor before reading one preset as faster
  or slower than another.
- **IPC is nil.** Round trip is within 20 ms of the worker's own compute time in every
  row, under 4 % of a 1-granule call.
- **The first call at a new window size costs 0.5 to 3.7 s** and is not in the table. It
  tracks the window length (0.58 s at 1 granule, 3.67 s at 8), which fits a VAE encoder
  pass over the window to derive its latent shape rather than one-off kernel compilation;
  rows on the second track are cheaper because the resident worker has the shape cached.
- **Worker load was 6.5 to 22.5 s across six starts** on this machine, same command,
  nothing else running. The rows above start after the load, and a latency claim that
  folded a 22 s start in would be wrong.
- **Peak worker RSS as `ps` reports it was 217 to 1344 MB** across the runs, which is not
  the model's resident cost (3.3 GB of weights; MPS allocations are largely invisible to
  `ps`, as above).

**A preset cannot run through the client that exists today.** `ResidentModel::noise_len`
(`crates/infer/src/session.rs:167-173`) answers `channels * frames` for any window that is
not the trained 245 760, but the worker sizes its noise from the VAE's latent grid, one
frame per 3 840 samples. Verified against the running worker: a 30 720-frame request with
the 61 440 samples that rule produces is rejected with

    ValueError: noise_len 61440 does not match the graph's latent grid (2, 16, 8, 32)

`tools/bench_windows.py` derives the count (`channels * 16 * 32 * frames / 3840`) and
checks it against HELLO's count for the trained window before its first call. A preset
therefore needs that derivation, or a per-window HELLO, in the client as well as in the
pipeline (D6, D11).

### Listening verdict, and what it found (2026-11-18)

The human listened to the ten files once, without a detailed comparison: **2 granules
had audible noise on the NieR track; 1, 4 and 8 granules were acceptable.** That is not
a verdict per preset, and it is not yet enough to drop one, but it is a real complaint
and it has a measurable shape:

| track | preset | mean spectral flatness 14–22 kHz | 2g vs 8g at 19–25 s |
|---|---|---|---|
| NieR | 1 granule | 0.058 | — |
| NieR | 2 granules | 0.050 | 2g 0.10–0.16, 8g 0.001–0.011 |
| NieR | 4 granules | 0.065 | — |
| NieR | 8 granules | 0.029 | — |

Spectral flatness runs from 0 (tonal) to 1 (noise) in the band the model generates, per
second. The 2-granule complaint lands at 19–25 s, where its added band measures 0.10–0.16
against 0.001–0.011 at 8 granules: the model is filling the top octave with noise rather
than with harmonics or air, in a passage where the dry level is unchanged (−15 to −21
dBFS across all four presets). The same measurement says 1 granule is no cleaner
elsewhere (flatness peaks at 0.256 around 20 s), so the effect is "short windows get
noisy in some passages", not "2 granules is uniquely broken".

Two things this is not. It is not a seam: sample-to-sample difference at each preset's
overlap-add step is 0.81–1.15× the difference elsewhere, i.e. indistinguishable, because
`tools/bench_windows.py` applies the app's crossfade. And it is not a level change: the
per-second RMS of every preset matches the dry file to the decibel.

The open decision is a product one: offer all four presets with the tradeoff stated, or
drop the short ones. A focused 18–26 s A/B of `nier-2g.wav` against `nier-8g.wav` is the
test that settles it, and the human owns it.

## Live file playback (U1), 2026-11-18

`flacify play --live` streams one window at a time: the rings hold `2B + W + block` frames
and nothing else, so an hour costs the same as a minute. M1 (the M5 Max), 2026-11-18, the
resident worker the app spawns, the 60-minute fixture
(`tools/fixtures/out/hour.mp3`, 192 kbps 48 kHz MP3, so the feed is already at 48 kHz;
the 44.1 kHz path is pinned by a test against the whole-file resampler). A 60 s excerpt
for the four-preset rows, 15 s of watching each; the acceptance row is ten minutes of the
hour. The measured ratio is the pump's own timing per second of *played* audio.

| preset | window | buffer | latency stated | measured RTF | dropouts in 15 s |
|---|---|---|---|---|---|
| 8 granules | 245 760 frames (5.12 s) | 245 760 (5.12 s) | 10.26 s | 0.79 | 0 |
| 4 granules | 122 880 (2.56 s) | 122 880 (2.56 s) | 5.14 s | 0.84 | 0 |
| 2 granules | 61 440 (1.28 s) | 61 440 (1.28 s) | 2.58 s | 0.76 | 0 |
| 1 granule | 30 720 (0.64 s) | 30 720 (0.64 s) | 1.30 s | 0.82 | 0 |

Throughput is flat across the presets (0.76–0.84), which is D19's claim measured through
the app rather than through the worker: a preset buys latency, not speed. `--rtf`
defaults to 0.8, the end-to-end figure these rows land on, and sizes the buffer until S3
feeds it the app's own measurement.

The ten-minute acceptance, `play <hour.mp3> --live --smoke --watch 600`:

| metric | value |
|---|---|
| audio played | 600.00 s in 600.0 s of watching |
| measured RTF | **0.75** over the last 147 blocks |
| dropouts | **0** (0 starved frames) |
| resident rings | 7 471 104 bytes, allocated once at start |
| process RSS | 36.7 MB → 52.3 MB at startup, then **52.3 MB flat** for nine minutes |
| switch | original; no automatic promotion on the live path |
| machine | load average 3.7 mean, 10.7 peak, 18 cores, other applications running |

The first ten-minute run, before the pump was changed to ask for the wet ring's room
after the render instead of before it, played 600.1 s but starved **59 %** of its frames
(17.0 M of 28.8 M): the compute and the playhead's drain took turns. That run is why the
dry ring also carries one block over the window now, and both parts are pinned by tests.

A second ten-minute run taken while the machine was busier (load average mean 8.6, peak
41.4) is the failure case, and it behaves as D2 says: RTF **0.79**, **7 dropouts** over
6.6 s of the 600 (1.1 %), the dry side played through each one, no gap and no silent
stall, and the same flat RSS. A spike the producer cannot cover costs a dropout, not a
stall, and the next block recovers it on its own.

## The 60-minute acceptance (A1), 2026-11-18

M1 (Apple M5 Max, 36 GB, macOS 27.2.1), release build, a 3600.0 s file built by
`tools/make_hour.py` from two library MP3s (82 MB, 48 kHz stereo, cutoff 13.1 kHz).

**Render**, through `flacify batch` on a folder holding that one file: 3401.6 s of wall
clock for 3600 s of audio (ratio 0.945), including the source tier, the FLAC segments and
the 8 s model load. One file, one worker.

**Replay**, `play hour.mp3 --smoke --watch 60` from the completed entry:

| what | value |
|---|---|
| first audio | 0.33 s after the command (2 s prebuffer included) |
| played | 60.03 s in 60 s of watching |
| dropouts | **0** frames (0.000 s) |
| model process | none started: a cache hit does not load the model |
| user CPU | 0.12 s |

**Seek to 55:00**, `play hour.mp3 --smoke --watch 20 --seek 3300`:

| what | value |
|---|---|
| landed | 0.07 s after the seek (playhead past 3300.0 s, audio flowing) |
| starved | 2048 frames (0.043 s), the segment read and decode |
| rest of the run | no further dropouts |

**The replay used to take 78 s to first audio.** A cache hit re-ran `analyse_streaming`
over the whole source to rediscover a cutoff the manifest already stored, which cost 60 s
of CPU for an hour-long track. Reading it from the entry took first audio from 78.18 s to
0.33 s and user CPU from 62.24 s to 0.12 s. The startup starvation those 78 s produced
was also being counted as dropouts; a miss before the first sample is now reported once,
as time to first audio, not twice.

## Clock drift between two devices (S2), 2026-11-18

M1 (Apple M5 Max), `cargo run --release --bin sink_probe` inside `Flacify.app` launched
through LaunchServices (a shell-launched process gets digital silence from CoreAudio; see
R23 in `risks.md`). Capture from "BlackHole 2ch", playback to "External Headphones",
silent output, 600 s, `--silent`.

| what | value |
|---|---|
| capture clock vs system clock | −5.1 ppm |
| playback clock vs system clock | +48.7 ppm |
| **relative drift** | **−53.8 ppm** |
| over 600 s | 32 ms |
| over an hour | 194 ms |
| discontinuities in the captured stream | 0 |

The measured drift sits almost exactly where the controller was proved: the simulated
+50 ppm case held 52 ppm of correction with **0 resyncs**, and the ring is 200 ms against
a clamp of 1000 ppm. A crystal pair 20× worse than this one would still resync rather than
click, which is the fallback `resync_for` implements.

Two caveats, stated rather than implied. This is one device pair on one machine; a USB
interface with its own clock is the case to check if a user reports it. And the capture
stream was silent, so this measures the two clocks rather than audio through them — which
is all the drift question needs, since the sample counts are what drift moves.

## The sink, end to end (A3), 2026-11-18

M1, `flacify sink --from "BlackHole 2ch" --to "External Headphones" --smoke --watch 600`,
with a 620 s WAV fed into BlackHole by `sink_probe --play` (ffmpeg's AudioToolbox output
would not start on this machine: `AudioQueueStart -66637`).

| what | value |
|---|---|
| played | 600.04 s in 600.0 s of watching, i.e. real time |
| dropouts | **0** |
| starved frames | **0** |
| resyncs | 1 |
| measured RTF | 0.85 over 147 blocks |
| ring capacity | 245 760 frames (one window), steering target 200 ms, latency 10.46 s |

**The first run of this failed, and found a bug the simulations could not.** The capture
ring was 19 200 frames (400 ms) because it was sized to the steering target, while one
model call takes 3.7 s. A capture callback cannot be blocked, so the ring overflowed on
every window: 11 044 640 frames dropped and 369.77 s played out of 600. The ring is now one
model window with the target unchanged, and the second run above is the result.

Two caveats, so this is not read as a clean acceptance. The output device fell back to the
MacBook Pro Speakers — the headphones had disconnected — so the route tested was not the
one this run names. And the capture state reads `silent` at report time, most likely
because the feeder's 620 s ran out a few seconds before the sink's report; that is the
harness, not the sink, but it is unconfirmed. A clean A3 needs one more run with both
devices present and a feeder that outlasts the watch.

## The blind report (A5), 2026-11-18

`flacify ab "<NieR track>" --trials 20 --seed 1` on the release binary. It **refused to
score**, which is the gate working:

```
Error: trial 2 at 27.551 s is out of level by +0.53 dB, over the 0.5 dB gate;
the score would measure loudness
```

The report was still written, marked not evidence. What this says is worth keeping: the
loudness match is fixed per track from the first window, and on real dynamic music one
6 s excerpt landed 0.53 dB out. Either the match needs to be per-excerpt or the gate needs
a defensible number; 1.1 owns that. 1.0 ships the command and the refusal, and **no blind
report is attached to this release**, because the only run of it failed its own gate.

## A URL as a source, end to end (2026-11-27)

`flacify play` and `flacify enhance` take a URL now, with `yt-dlp` doing the fetch.
yt-dlp 2026.10.26, ffmpeg 9.0.2, both from Homebrew; M1, 2026-11-27. The video is
`https://www.youtube.com/watch?v=hyWDKjljFt0`, a 188.4 s upload, with the playlist
parameters the share link carries (`&list=RDMM…&start_radio=1`) left on, which is what
`--no-playlist` and the `v=` parse have to survive.

| what | measured |
|---|---|
| fetched file | 39 162 240 bytes, FLAC 24-bit 48 kHz stereo, 9 042 408 frames (188.38 s) |
| download + first audio, play --smoke | 19.75 s cold, 0.33 s once the file exists |
| detected cutoff | 13 757 Hz, so the top 10.2 kHz is inferred |
| playback | 289 792 frames in 6 s of watching, 0 dropouts, switch `generated`, promoted at 19.9 s |
| enhance, whole track | 141.1 s of compute for 188.4 s of audio (RTF 0.75) |
| high band, source vs rendered, 30–50 s | 16–22 kHz **+5.1 dB** relative to 1–5 kHz; output length equals input to the sample |
| replay with yt-dlp removed from `PATH` | 0.32 s to first audio, same title, file mtime unchanged |

The replay line is the one that matters for the design: the title comes from a
`names.tsv` written beside the audio, so a machine that no longer has yt-dlp can still
replay a file it fetched, and nothing re-downloads.

The high-band figure is a crude FFT mean over a 20 s window, not the loudness-matched
per-window comparison the acceptance runs use; it says the model added content, not how
well it matched. A proper listening verdict on this track is not recorded here.

## Playlists (unverified)

The session queue (ADR-015) has no end-to-end number here yet. `yt-dlp --flat-playlist`
listing time and the first-track download were **not** timed against a real playlist when
the queue landed, so this file makes no claim about how long a radio mix takes to start.
A run on the M1, with the playlist source and the date, is the next row.

## A URL with no yt-dlp and no ffmpeg installed (2026-11-28)

The provisioning in ADR-016, measured on the same Windows box as the W-numbers: Ryzen 9
5900XT, 39 GB, Windows 11, yt-dlp **2026.11.23** installed by the app, `ffmpeg` **absent
from `PATH`** for every run.

| what | measured |
|---|---|
| `flacify setup`, yt-dlp step only | one `curl` of 23 627 865 bytes into `<home>\runtime\bin\yt-dlp.exe`; step reports done |
| fetch with no ffmpeg, `-f` branch | `aqz-KE-bpKQ.m4a`, 9.8 MB, no postprocessing step, no error |
| playback of that download, `--smoke` | played; 634.6 s track, source tier built from AAC, no frame-count mismatch |
| fresh home, no setup run | the old error, naming yt-dlp and how to install it — the message is worth keeping |

The second row is the point: yt-dlp needs ffmpeg only to remux or transcode, so asking it
for `140` (AAC in MP4) writes the file as served. `symphonia` in this build decodes it —
`flacify probe` reports 44 100 Hz, 2 ch, 634.62 s.

**Not verified:** the FLAC branch, which needs an ffmpeg on `PATH`; the yt-dlp asset
names for macOS and Linux (only the Windows name was executed — the others are read from
the release listing, not run).

### An AAC-backed segment that reads short — FIXED, root cause was `flacenc` (2026-11-28)

`https://www.youtube.com/watch?v=X_YHXdCcmVc` used to fail with `src-000001.flac holds
1380224 frames, source.json says 1384320`. The cause was not the decoder, the source tier
or the resampler: **`flacenc` 0.5.1 drops a whole frame on some inputs, and which block
size trips it depends on the data.**

Measured on this Windows box (Ryzen 9 5900XT, yt-dlp 2026.11.23, no ffmpeg), encoding
each of the nine 1 384 320-frame segments of the track at seven block sizes:

| block size | result |
|---|---|
| 4096 (the crate default) | segment 1 lost 4096 frames, wrote 33 624 078 bytes (≈ `4096 × 4096 × 2`) of unreadable stream |
| 3456 | segment 6 lost 3456 frames |
| 3072 | segment 7 lost 3072 frames |
| 2304 | **all 9 segments exactly right** |
| 2048, 1152, 1024 | all 9 segments exactly right |

The failures are always *exactly one block* short, so the encoder is dropping the frame
that contains a particular sample rather than truncating a file.

How it was narrowed, for whoever hits the next one: the writer and the decoder were
cleared first (`flac::write` → `decode::decode` at 1 384 320 frames returned the exact
count for every synthetic signal tried — sine, noise, silence, both rails), and a
different YouTube track (634 s) produced 23 uniformly sized segments. The remaining
variable was the encoder's own block size, which the table above settles.

Fix, both pinned by tests in `crates/app/tests/flac.rs`:

- `flac::BLOCK` is **2304**, the largest size that verified on every segment.
- `flac::write` decodes what it wrote and refuses to publish a stream whose frame count
  disagrees, so the next input that trips this fails a write instead of a track.

Verified after the change: all nine segments decode to exactly 1384320 frames (the last
to 260494), the tier totals 11 335 054 frames, and `flacify play` on the file reaches the
model check with no frame mismatch instead of stopping in the source tier.

Not verified: whether any block size above 2304 is safe for other content. The check in
`write` covers that — a bad encode now fails loudly — but the numbers above are one track.

