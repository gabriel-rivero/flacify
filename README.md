# flacify

A desktop player that reconstructs the high frequencies an MP3 threw away, and lets
you hear the difference on a single keystroke.

Status: **prototype.** The Rust workspace builds and its FlashSR preprocessing port
matches scipy to −132 dB. The ONNX export passes its numeric gate (−74 dB vs PyTorch,
worst track −48 dB) but ONNX Runtime runs it 30 to 60x slower than PyTorch on the same
Apple Silicon, so the prototype runs the model in a Python sidecar the app installs on
first run: RTF 0.543 through the pipe, output bit-identical to in-process Metal
(ADR-010). There is no audio output and no UI yet. The working reference implementation
is the Python PoC,
[`gabriel-rivero/flashsr-stream`](https://github.com/gabriel-rivero/flashsr-stream).
This repo is the Rust rewrite.

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
| [`docs/decisions.md`](docs/decisions.md) | ADR-001 … ADR-008 with status |

## Start here

```sh
cargo test --workspace                 # 39 tests, no GPU, no audio device, no model
cargo run -p flacify -- probe song.mp3 # cutoff, what would be inferred, and the wait
```

`docs/decisions.md` records the accepted ADRs. Everything downstream is written
against them.

## Reference material

- [`flashsr-stream`](https://github.com/gabriel-rivero/flashsr-stream) — the Python
  implementation and its measured numbers (the PoC this repo rewrites)
- [`docs/measurements.md`](docs/measurements.md) — every number in this repo, with the
  machine and date behind it
