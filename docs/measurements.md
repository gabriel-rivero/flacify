# Measurements

Every number this repo quotes, with the machine, the command that produced it and the
date. If a number is not here, it is either borrowed (and says so) or it does not exist.

## Machines

| # | what | specs |
|---|---|---|
| M1 | development laptop (2026-11-12, every number below except P1's and M2's) | Apple M5 Max, 32-core GPU, 36 GB, macOS 27.2.1, Rust 1.99.1, Python 3.12.13, torch 2.15.0, onnxruntime 1.31.1 |
| P1 | the PoC's machine (PoC numbers only) | RTX 4060 Ti 8 GB, Ryzen 9 5900XT, 39 GB, Linux, torch 2.15.0+cu132 |
| M2 | the same RTX 4060 Ti box, running flacify (Linux numbers) | CachyOS, driver 620.70.06 (CUDA 13.5), Ryzen 9 5900XT, 39 GB, Rust 1.99.1, Python 3.12.16, torch 2.15.0+cu132, onnxruntime-gpu 1.31.1 |
| W1 | this Windows handheld (Windows numbers) | Ryzen Z1 Extreme, 8 cores / 16 threads, Radeon integrated graphics, 23.2 GB, Windows 11 Pro 26200, Rust 1.100.0 (MSVC), no CUDA |

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
path the README already describes. DirectML through ONNX Runtime was the candidate for a
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
