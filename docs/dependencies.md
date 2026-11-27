# Dependencies

Status: current, 2026-11-18. Versions are deliberately unpinned; pin them in
`Cargo.toml` at v0.1 and let `cargo update` move them.

The rule: **the shipped binary must not contain Python.** If a dependency needs a
Python runtime, it belongs in `tools/`, not in `crates/`.

**Prototype exception (ADR-010).** v0.0 installs a Python sidecar on first run: `uv`
fetches CPython 3.12, pip installs the pinned wheels, and the weights download from
upstream. About 6 GB, once — 6.9 GiB measured with the CUDA wheels, 4.2 GiB with PyPI's
CPU-only torch. This is time-boxed scaffolding for evaluating the product,
not a shipping design, and the rule above is unchanged for v1.

PyPI's `torch` is CPU-only on Windows and Linux. A CUDA build comes from PyTorch's own
index and costs about 2.5 GB more, so it is opt-in per machine rather than assumed:

```sh
flacify setup --cuda
```

That is the same as exporting `FLACIFY_TORCH_INDEX` yourself:

```sh
FLACIFY_TORCH_INDEX=https://download.pytorch.org/whl/cu132 flacify setup
```

The index must carry the same torch version the pins resolve to; `cu132` is the one
that publishes 2.15.0. Setting it upgrades an existing CPU install on the next
`setup` (2.0 GiB, about 13 minutes measured), and the chosen index is recorded in the
wheels receipt, so changing it re-runs that step. On Windows the PowerShell form is
`$env:FLACIFY_TORCH_INDEX = "https://download.pytorch.org/whl/cu132"`. Leaving it
unset keeps the CPU wheel and does not re-download anything. The in-window setup does
not offer the flag yet; use `flacify setup --cuda` on the CUDA machine.

## Rust crates

Sixteen direct dependencies, and `cargo deny check bans licenses sources advisories`
in CI is what keeps the list honest. The tables below are the ones in `Cargo.lock`
today; a crate that is not there is not used, whatever an earlier plan said.

### In the tree

| crate | where | why | risk |
|---|---|---|---|
| `symphonia` | app | demux + decode MP3, AAC, FLAC, ALAC, Vorbis, Opus, WAV, MP4, OGG | MPL-2.0: file-level copyleft, fine to link, don't fork silently |
| `cpal` | app | one audio output and input API over CoreAudio / WASAPI / ALSA | device matrix is the real cost, not the crate |
| `flacenc` | app | pure-Rust FLAC encoder for the cache and `enhance --out x.flac` | young; 24-bit round-trip verified 2026-11-17, streaming append still to check |
| `hound` | app | WAV reading in `sink_probe` and writing in `enhance`/`wav.rs` | — |
| `lofty` | app | reads the title/artist tag the window shows | — |
| `egui` + `eframe` | app | UI, custom-painted, single binary | skinning is manual; no Winamp `.wsz` for free |
| `arc-swap` | app | hands the render frontier, resident segment and status to the window without a lock | — |
| `blake3` | app | cache keys and model checksums | — |
| `realfft` | core, app | FFT for the cutoff detector, spectrogram, loudness | — |
| `clap` | app | CLI (`play`, `enhance`, `probe`, `sink`, …) | — |
| `serde`, `serde_json` | app, core (dev) | cache manifest, settings, fixture goldens | — |
| `tracing` + `tracing-subscriber` | app | structured logs to a rolling file | no `tracing-appender`: the rolling file is hand-rolled |
| `thiserror`, `anyhow` | core + infer, app | typed errors internally, context at the edges | — |

The resampler is not a crate: `crates/core/src/dsp/resample.rs` is a port of
`scipy.signal.resample_poly`, pinned by the scipy golden tests. The rings and the
control plane are the standard library (`AtomicU32` samples in a bounded ring,
`std::sync::mpsc`, `arc-swap`), so there is no `rtrb` and no `crossbeam-channel`.

### Considered, not used

| crate | why it is not here |
|---|---|
| `rubato` | the band-limit round trip has to match `scipy.signal.resample_poly` bin for bin, and a general resampler does not. The port is ~285 lines of implementation in `dsp/resample.rs` (452 with its tests) and already pinned against scipy goldens |
| `biquad` | the Chebyshev port is SOS filtering with the scipy matrix pinned in a test; a crate would hide the bug surface |
| `ort`, `ndarray` | ADR-002's ONNX Runtime measured 15–31 RTF against PyTorch's 0.51 on the same weights, so the backend is the Python sidecar (ADR-010) and no Rust tensor crate is in the tree |
| `reqwest`, `ureq`, `sha2` | nothing downloads from Rust: `setup` runs `uv`, `pip` and `curl` as child processes, and the weights come through them |
| `claxon` | redundant with symphonia; add only if 30 s seek latency becomes a problem |
| `egui_kittest` | UI state-machine tests target the plain data the window reads; the crate was not needed |
| `toml`, `directories` | settings are JSON with hand-written platform paths (`platform_data_dir` in `setup.rs`) |
| `souvlaki` | OS media keys / Now Playing: nice-to-have, adds platform surface |

Dev-only crates that a reader might expect are also absent: dev dependencies are
`serde` and `serde_json` in `core` and nothing else. There is no `proptest`, `insta`
or `criterion`; the DSP goldens are JSON fixtures generated by `tools/fixtures/gen.py`
and the timing numbers come from the `tools/bench_*.py` scripts in
[`measurements.md`](measurements.md). `cargo-nextest` and `cargo-machete` are
developer conveniences, not build requirements.

## Deliberately absent

| not using | because |
|---|---|
| `rodio` | it owns the player and the device; we need our own queue and clock |
| `tokio` | nothing here needs an async runtime; every long job is a child process. It reaches `Cargo.lock` only as `wasm-bindgen-futures`' dependency under `cpal`/`eframe`/`winit`, which builds for wasm only, not for the shipped targets |
| `tauri` | a webview and a JS toolchain for one window, and it breaks "the binary is the app" |
| `candle` / `burn` (for now) | reimplementing the graph and the vocoder is more work than exporting it; revisit if the runtime endgame in [`decisions.md`](decisions.md) moves |
| any `libmpg123` / `ffmpeg` sys dependency | the first-run story stays trivial if decode is pure Rust |
| a Rust inference runtime | the graph is run by `tools/worker.py` under the app's Python sidecar (ADR-010), so no ONNX, CUDA or Metal binding is linked |

## External programs

The app runs other programs; it links none of their libraries. `flacify setup` installs
the sidecar with `uv`, `pip`, `curl` and `tar`, and `flacify play|enhance <url>` fetches
audio with `yt-dlp`. All of them are located on `PATH` (honouring `PATHEXT` on Windows)
and each is overridable for tests: `FLACIFY_UV`, `FLACIFY_YTDLP`.

| program | needed for | if it is missing |
|---|---|---|
| `yt-dlp` | any URL as a source | the command says how to install it and stops; local files are unaffected |
| `ffmpeg` | the audio conversion `yt-dlp -x` does | yt-dlp reports it; we ask for FLAC because it is what `symphonia` decodes |
| `uv`, `curl`, `tar` | `flacify setup` | the install fails with the step name |

Choosing `PATH` over a vendored binary is deliberate: `yt-dlp` changes weekly to keep up
with the sites it reads, so pinning it here would pin the product's usefulness with it.
`flacify diagnose` prints where `yt-dlp` was found.

The decode side stays pure Rust and links no `libmpg123`/`ffmpeg`; a fetched file is an
ordinary input the moment it lands.

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

**Status 2026-11-18:** mitigation 2 is unanswered — no issue has been filed and no reply
exists. 1.0 ships under mitigation 1: weights downloaded at first run, never bundled,
attributed in the release notes, unsigned, with a notice that the weights are the author's
and are not covered by this project's MIT licence. Filing the issue is still the right
move; it changes 1.1's options, not 1.0's.

`cargo-deny` covers our crates; the model is a manual gate in the release checklist.
