# Dependencies

Status: draft, 2026-11-12. Versions are deliberately unpinned; pin them in
`Cargo.toml` at v0.1 and let `cargo update` move them.

The rule: **the shipped binary must not contain Python.** If a dependency needs a
Python runtime, it belongs in `tools/`, not in `crates/`.

## Rust crates

| crate | why | risk |
|---|---|---|
| `symphonia` | demux + decode MP3, AAC, FLAC, ALAC, Vorbis, Opus, WAV, MP4, OGG | MPL-2.0: file-level copyleft, fine to link, don't fork silently |
| `cpal` | one audio output API over CoreAudio / WASAPI / ALSA | device matrix is the real cost, not the crate |
| `rubato` | high-quality resampling 44.1 → 48 kHz | different filter than scipy's `resample_poly`, see the preprocessing port |
| `flacenc` | pure-Rust FLAC encoder for the cache | young; verify 24-bit + streaming append |
| `claxon` | fast FLAC decode for cache replay | redundant with symphonia; add only if 30 s seek latency is too slow |
| `ort` | ONNX Runtime bindings; the only place GPU code is called | dylib distribution, EP quirks, `coreml` coverage |
| `ndarray` | tensor shapes across the `ort` boundary | keep it out of `core` |
| `realfft` | FFT for the cutoff detector, spectrogram, loudness | — |
| `biquad` *(not used)* | was the plan for the cheby1 port | the port is ~120 lines of SOS filtering with the scipy matrix pinned in a test; a crate would hide the bug surface |
| `egui` + `eframe` | UI, custom-painted, single binary | skinning is manual; no Winamp `.wsz` for free |
| `egui_kittest` | UI state-machine tests | young |
| `rtrb` | SPSC ring for audio, wait-free, no allocation | pick one ring crate and stay with it |
| `crossbeam-channel` | control plane | never touch it from the audio callback |
| `blake3` | cache keys and model checksums | — |
| `ureq` | blocking HTTP for model download (resume, ranges) | no async runtime needed |
| `serde`, `toml`, `serde_json` | `profile.toml`, cache manifest, settings | — |
| `directories` | platform config/cache paths | — |
| `clap` | CLI (`play`, `enhance`, `probe`, `diagnose`) | — |
| `tracing` + `tracing-subscriber` + `tracing-appender` | structured logs to a rolling file | — |
| `thiserror`, `anyhow` | typed errors internally, context at the edges | — |
| `souvlaki` | OS media keys / Now Playing (optional) | nice-to-have, adds platform surface |

Dev: `proptest`, `insta`, `criterion`, `cargo-nextest`, `cargo-deny` (licences and
advisories), `cargo-machete` (unused deps).

## Deliberately absent

| not using | because |
|---|---|
| `rodio` | it owns the player and the device; we need our own queue and clock |
| `tokio` | nothing here needs async; the download is one blocking HTTP call |
| `tauri` | a webview and a JS toolchain for one window, and it breaks "the binary is the app" |
| `candle` / `burn` (for now) | reimplementing the graph and the vocoder is more work than exporting it; revisit if `ort`'s macOS story disappoints |
| any `libmpg123` / `ffmpeg` sys dependency | the first-run story stays trivial if decode is pure Rust |
| Python at runtime | see the rule above |

## Model runtime

The decision that matters. Same graph, four ways to run it.

| option | macOS | Windows | Linux | ship cost | verdict |
|---|---|---|---|---|---|
| `ort` + CoreML EP | native, fp16, compiled-model cache | n/a | n/a | ~40 MB dylib | **recommended for macOS** |
| `ort` + CUDA EP | n/a | needs CUDA + cuDNN versions | needs CUDA toolkit | 40 MB + 1–3 GB runtime | render-ahead only; don't require it |
| `ort` + DirectML EP | n/a | works on AMD/Intel/NVIDIA | n/a | 40 MB | **recommended for Windows** |
| `ort` CPU EP | slow | slow | slow | 40 MB | fallback, offline only |
| `candle` + Metal | good if it works | — | — | none (pure Rust) | spike alternative for Apple |
| `mlx` | fastest on Apple in theory | — | — | extra runtime | immature Rust bindings; watch it |
| Python sidecar | works today | works | works | 6–8 GB, interpreter, venv, version skew | v0.1 scaffolding only, never ship |

Two things to settle in v0.0 with numbers, not opinions:

1. **CoreML EP vs candle-Metal vs MLX on the M5 Max.** Measure sustained RTF for each
   and pick the fastest one that survives 60 s without throttling.
2. **Whether `ort`'s CoreML EP covers every op in the graph.** Node-level fallback to
   CPU is allowed but can quietly halve throughput; log the partition and count CPU
   nodes.

## Model artifacts

| artifact | size (fp32) | notes |
|---|---|---|
| `student_ldm.pth` | ~2.6 GB | the distilled UNet/DiT |
| `vae.pth` | ~0.3 GB | encoder/decoder to mel + normalisation |
| `sr_vocoder.pth` | ~0.4 GB | mel → waveform |
| **exported ONNX, fp32** | ~3.3 GB | reference for verification |
| **exported ONNX, fp16** | ~1.7 GB | what ships; CoreML computes fp16 anyway |

Total first-run download target: under 2 GB, resumable, checksummed, with a mirror.
`cargo-dist`-produced installers stay under 100 MB because the model is not in them.

Export is a `tools/export_onnx.py` job (Python, dev-only), committed as a script plus
the hash of its output, not as a binary in git. `flacify diagnose` prints the model
hash so a bug report can name the exact artifact.

## Python (development only, `tools/`)

| tool | use |
|---|---|
| `torch` | export, fixed-seed reference renders, numeric comparison |
| FlashSR_Inference | the reference implementation, pinned to a commit |
| `numpy`, `scipy`, `librosa` | golden files for the preprocessing port, spectrograms, metrics |
| `onnx`, `onnxruntime` | export validation before the Rust side sees it |
| `pytest` | the comparison suite that the Rust tests read their fixtures from |

The Python suite's job is to produce **fixtures and numbers**, not to run the app. It
lives in CI on a machine with no GPU (it only needs to run the CPU path for
comparisons).

## Packaging

| task | tool |
|---|---|
| releases, checksums, changelog | `cargo-dist` |
| macOS `.app` + `.dmg` | `cargo-bundle` or `cargo-dist`, then `codesign` + `notarytool` |
| Windows `.msi` | `cargo-wix` (unsigned at v1, signed later) |
| Linux | AppImage + `.deb` |
| licence attribution | `cargo-about` (required, see below) |
| SBOM | `cargo-sbom` or `cargo-cyclonedx` |

macOS notarisation needs an Apple Developer Program membership (USD 99/yr) and a
Developer ID certificate. Plan for it at v0.5; without it, Gatekeeper blocks the
first launch on other people's Macs and the project looks broken.

## Licensing

| component | licence | action |
|---|---|---|
| `flacify` code | MIT (matching the PoC) | add `LICENSE` |
| ONNX Runtime | MIT | attribution |
| `symphonia` | MPL-2.0 | attribution; disclose any file we modify |
| everything else in the table | MIT / Apache-2.0 | `cargo-deny` allow-list |
| FlashSR_Inference | **no LICENSE file at the repo root** | see below |
| FlashSR weights (HF dataset) | **no stated licence** | see below |
| vocoder (BigVGAN-derived, NVIDIA) | MIT upstream | verify the vendored copy |

The last two rows are a shipping blocker, not a footnote. With no licence file, the
default under copyright is "all rights reserved", which is not a licence to
redistribute. Three mitigations, in order of preference:

1. **Do not redistribute the weights.** Download from the upstream HF dataset at
   first run, attribute, and keep the model out of the installer. The user obtains it
   from the author, not from us. This is also what keeps the installer small.
2. Open an issue with the author asking for an explicit licence (MIT/Apache-2.0 or
   CC-BY-NC for weights). Ask before v1.0, since a "no commercial use" answer changes
   the product's future.
3. If neither works, keep flacify a personal/OSS research tool with a clear notice,
   and do not ship a signed consumer build.

`cargo-deny` covers our crates; the model is a manual gate in the release checklist.
