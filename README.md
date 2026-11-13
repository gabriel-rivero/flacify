# flacify

A desktop player that reconstructs the high frequencies an MP3 threw away, and lets
you hear the difference on a single keystroke.

Status: **prototype, playable on macOS and Linux.** `flacify setup` installs the Python sidecar and
the model, `flacify enhance` renders a file, and `flacify play` opens a window with an
instant A/B switch. The Rust side owns decode, analysis, the windowing pipeline, the seed
schedule and the mixer; the model runs in a Python process the app installs and owns
(ADR-010), because ONNX Runtime measured 30 to 60x slower than PyTorch on the same Apple
Silicon (ADR-002). The working reference implementation is the Python PoC,
[`gabriel-rivero/flashsr-stream`](https://github.com/gabriel-rivero/flashsr-stream).

Runs end to end on macOS (verified: Metal, 48 kHz out, A/B enabled) and on Linux
(verified: CUDA on an RTX 4060 Ti, a 44.1 kHz device resampled to 48 kHz, A/B
enabled). Windows is still unrun on a real device; CI builds and tests it. The
accelerator is `cuda` if present, `mps` on Apple Silicon, otherwise `cpu`, which is a
renderer rather than a player (RTF 6–10). On Linux the same graph through ONNX
Runtime's CUDA EP reaches RTF 0.545 and would cut the first-run download roughly in
half, but the sidecar still runs torch; the numbers are in
[`docs/measurements.md`](docs/measurements.md#linux-the-download-and-whether-onnx-runtime-can-shrink-it-2026-11-12).
See the platform gap in [`docs/roadmap.md`](docs/roadmap.md).

## What works today

```sh
cargo build --release
./target/release/flacify setup                 # once, about 6 GB
./target/release/flacify probe song.mp3        # cutoff, verdict, wait
./target/release/flacify enhance song.mp3      # -> song.mp3.flacify.wav
./target/release/flacify play song.mp3         # window: space toggles, p plays
./target/release/flacify play song.mp3 --smoke # headless audio-path check
```

Measured on an M5 Max: a 30.7 s clip renders in 34.1 s (including the 8 s worker start)
and plays back at 48 kHz with the switch enabled. The generated band gains +24.6 dB at
16–18 kHz and +61.6 dB at 18–22 kHz over the input; the level moves +0.57 dB, which is
one of the open gaps in [`docs/roadmap.md`](docs/roadmap.md#gaps-found-building-the-prototype-2026-11-12).

## The one-paragraph pitch

Lossy codecs delete everything above ~16 kHz. FlashSR, a one-step distilled
diffusion model, puts plausible content back: on an RTX 4060 Ti it processes 5.12 s
of stereo in 1.9 s (RTF 0.37), fast enough to stay ahead of playback. flacify wraps
that in a player with an honest A/B switch: the original and the reconstructed
signal stay sample-locked, so toggling is instant and gap-free. It runs locally, on
your machine, on your files.

## What this is not

It does not recover your original master. The high band is generated, not restored,
and FLAC is only the lossless container for that generated signal. Any claim about
"lossless conversion" or "restoring what was lost" is wrong and we do not make it.
See [`docs/vision.md`](docs/vision.md).

## Documents

| doc | what it answers |
|---|---|
| [`docs/vision.md`](docs/vision.md) | who this is for, what we claim, what we refuse to claim |
| [`docs/roadmap.md`](docs/roadmap.md) | releases v0.0 → v1.0, exit criteria, the MVP cut |
| [`docs/architecture.md`](docs/architecture.md) | components, threads, buffer math, A/B alignment, cache |
| [`docs/dependencies.md`](docs/dependencies.md) | crates, model runtime, artifacts, licences |
| [`docs/risks.md`](docs/risks.md) | risk register, kill criteria |
| [`docs/backlog.md`](docs/backlog.md) | epics and issue drafts, in build order |
| [`docs/decisions.md`](docs/decisions.md) | ADR-001 … ADR-010 with status and outcomes |

## Start here

```sh
cargo test --workspace   # 87 tests, no GPU, no audio device, no model, no Python
```

`core` and `infer` are testable without a model, which is what makes the DSP and the
mixer developable on any laptop. The Python sidecar is the one seam that needs the
6 GB install, and even it is tested against a fake worker binary in CI.

`docs/decisions.md` records the accepted ADRs and, for ADR-002, the measurement that
reopened it.

## Reference material

- [`flashsr-stream`](https://github.com/gabriel-rivero/flashsr-stream) — the Python
  implementation and its measured numbers (the PoC this repo rewrites)
- [`docs/measurements.md`](docs/measurements.md) — every number in this repo, with the
  machine and date behind it
