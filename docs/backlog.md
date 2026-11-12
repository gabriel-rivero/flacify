# Backlog

Status: draft, 2026-11-12. Issue drafts are written to be pasted into `gh issue
create` as-is. Labels: `epic`, `spike`, `p0`, `dsp`, `audio`, `ui`, `platform`,
`quality`, `legal`, `docs`.

Definition of done for every issue: the code, a test that fails without it, and the
doc line that would have been wrong. No issue closes on "works on my machine".

## E0 — Project setup

| # | title | acceptance |
|---|---|---|
| E0-1 | Scaffold workspace (`core`, `infer`, `app`) | `cargo test` green on a machine with no GPU, no audio device, no model |
| E0-2 | CI: fmt, clippy `-D warnings`, nextest, cargo-deny | red on a lint violation and on a GPL dependency |
| E0-3 | Add `LICENSE` (MIT), `CONTRIBUTING.md`, issue/PR templates | licence detected by GitHub |
| E0-4 | `docs/measurements.md` in this repo, filled as numbers arrive | every number in the README has a row here with the machine and date |

## E1 — Runtime spike (v0.0, kills or confirms the project)

| # | title | acceptance |
|---|---|---|
| E1-1 | `tools/export_onnx.py`: FlashSR → ONNX, fixed shapes `1×2×245760`, fp32 | file loads in `onnxruntime` Python and produces audio |
| E1-2 | Fold the one-step scheduler into the graph; noise becomes an input | same noise tensor → identical output to PyTorch, ≤ −40 dB |
| E1-3 | Numeric comparison harness (PyTorch vs ONNX, 5 tracks, fixed seed) | fails loudly below −40 dB; JSON report per track |
| E1-4 | `ort` session in Rust, all EPs enumerated, partition logged | prints which EP runs which node, with a CPU-node count |
| E1-5 | Sustained RTF benchmark on the M5 Max (10-window median + 60 s run) | `rtf_p50`, `rtf_p95`, `rtf_sustained`, peak memory in one JSON |
| E1-6 | 30 s of audio through the Rust path to `cpal` | audible output, no dropout, RTF logged |
| E1-7 | Close ADR-002 with the measured winner (CoreML / candle-Metal / MLX) | numbers in the ADR, not adjectives |

## E2 — Core DSP (v0.1)

| # | title | acceptance |
|---|---|---|
| E2-1 | `Upscaler` trait + `Identity`, `Gain`, `Delay`, `Noise`, `FailAfter` mocks | identity run reproduces input within 1 ULP |
| E2-2 | Windower: advance by `window − overlap`, zero-pad tail, exact sample count | property test over lengths 0…3 windows × all overlaps |
| E2-3 | Raised-cosine overlap-add, tail carry | crossfade pair sums to 1.0 ± 1e-6; no discontinuity > 1e-6 |
| E2-4 | Cutoff detector port (STFT 2048/480, 98.3% cumulative, ≥1000 Hz clamp) | bin-identical to Python on a 20-file corpus |
| E2-5 | Preprocess port: cheby1 order 8 rp 0.1 (SOS, forward-backward) + polyphase round trip | ≤ −60 dB vs Python; edge behaviour documented |
| E2-6 | Loudness matcher (BS.1770 momentary, 1 s smoothing, silence gate) | dynamic track: no breathing; ≤ 0.5 dB step on toggle |
| E2-7 | Buffer policy + fixed-depth backpressure | simulated RTF 0.3–2.0 → no underrun below 0.95 |
| E2-8 | Deterministic noise seeding from `(file hash, window index, version)` | two renders byte-identical |

## E3 — Playback (v0.1)

| # | title | acceptance |
|---|---|---|
| E3-1 | Symphonia decode → planar f32, all formats in the matrix | MP3, AAC/M4A, FLAC, WAV, OGG/Opus all play |
| E3-2 | Rubato 44.1 → 48 kHz at the source | < 1% of a core; 44.1 and 48 sources sound identical in A/B |
| E3-3 | `cpal` output with SPSC ring, alloc-free callback | no alloc/lock/log in the callback (assert with a custom allocator in tests) |
| E3-4 | A/B mixer: delayed passthrough, 20 ms equal-power crossfade | null test passes; toggle at any point is click-free |
| E3-5 | CLI `flacify play <file>` with keyboard toggle | plays a whole album side without a dropout |
| E3-6 | Underrun handling: passthrough, counter, no gap | forced stall produces passthrough, not silence or a click |

## E4 — UI (v0.2, the MVP)

| # | title | acceptance |
|---|---|---|
| E4-1 | `eframe` window, compact player chrome, always-on-top option | launches in < 1 s warm, < 3 s cold |
| E4-2 | Spectrum view 20 Hz–24 kHz with the reconstructed band shaded | 30 fps, < 5% of a core |
| E4-3 | Waveform with rendered frontier + unrendered dimming | frontier visible, matches the cache |
| E4-4 | Transport, gain, A/B button + `A` key, hold-to-compare | toggle < 30 ms, no click |
| E4-5 | First-run hardware probe → `profile.toml`, skipped on later runs | probe ≤ 40 s, re-probes on device change |
| E4-6 | Drag and drop a file onto the window | plays, replaces current file cleanly |
| E4-7 | Status surface: RTF, buffer, dropouts, EP, mode | matches `--stats` output |
| E4-8 | Failure states: `Passthrough`, `Rendering`, `Blocked` with a reason | every failure path reachable in a test |

## E5 — Render cache (v0.3)

| # | title | acceptance |
|---|---|---|
| E5-1 | Content-addressed cache dir + manifest + blake3 keys | two files with identical bytes share an entry |
| E5-2 | Segmented FLAC writer (30 s), crash-safe append | kill -9 mid-segment leaves playable segments |
| E5-3 | Cache reader + playback from cache, `D = 0` A/B | 60 min track, seek anywhere, zero dropouts |
| E5-4 | Partial render: passthrough ahead of the frontier | playback starts immediately, frontier advances |
| E5-5 | LRU eviction with a user cap, never evict the open file | cap respected; manifest survives |
| E5-6 | Re-render only invalidated segments on parameter change | changing overlap re-renders 0 segments when it should |

## E6 — Real-time mode (v0.4)

| # | title | acceptance |
|---|---|---|
| E6-1 | Bounded queue policy with RTF-derived depth | no underrun at RTF 0.5, passthrough at 1.1 |
| E6-2 | Latency presets, with the quality cost stated in the UI | each preset measured and labelled |
| E6-3 | Live capture source (`cpal` input) | 10 minutes of capture with no drift |
| E6-4 | Device hot-swap and sleep/wake recovery | unplug a headset mid-play, recover in < 2 s |

## E7 — Packaging (v0.5)

| # | title | acceptance |
|---|---|---|
| E7-1 | Model download: resume, checksum, mirror, cancel | survives a killed connection at 90% |
| E7-2 | Manual model-file escape hatch | user can point at a local `.onnx` |
| E7-3 | Licence screen + attribution screen (`cargo-about`) | shown once, stored |
| E7-4 | macOS `.app` + `.dmg`, signed and notarised | launches on a clean Mac with no dialog |
| E7-5 | Windows `.msi` (unsigned), Linux AppImage + `.deb` | installs and plays |
| E7-6 | `flacify diagnose` bundle | one file to paste into an issue |

## E8 — Quality harness

| # | title | acceptance |
|---|---|---|
| E8-1 | Five-track corpus with native-lossless references | committed as hashes + download script, not audio |
| E8-2 | Metrics: LSD, spectrogram diff, null-test residual, spectral flatness in the added band | report per track, regression threshold in CI |
| E8-3 | Artifact detector: transient smear, sibilance harshness, added-band noise | flags at least one deliberately bad render |
| E8-4 | Blind A/B harness (randomised, level-matched) | produces a report you can attach to a release |
| E8-5 | Bypass rule for full-band input | a FLAC source is passed through untouched |
| E8-6 | Classify "already full-band" from the spectrum, not the cutoff | `high_band_ratio ≥ 0.01` above 21 kHz flags full-band; a 16 kHz-limited source measures 1e-6, so the threshold has four decades of headroom and `cutoff ≥ 24000` stops being the rule (it almost never fires on real music) |

## E9 — Docs and demo

| # | title | acceptance |
|---|---|---|
| E9-1 | README with measured numbers and a 30 s demo clip | a stranger can install and hear it |
| E9-2 | `docs/measurements.md` per machine | every claim traceable |
| E9-3 | "What this does not do" section | reviewed by someone who does not like the idea |

## State of the scaffold (2026-11-12)

What actually exists, with its evidence. Everything here is uncommitted and the repo has
no git history yet.

| issue | state | evidence |
|---|---|---|
| E0-1 | **done** | `cargo test --workspace` green with no GPU, no device, no model: 19 lib + 1 fixture + 11 pipeline + 8 infer tests |
| E0-2 | **drafted, unverified** | `.github/workflows/ci.yml` exists and has never run on GitHub |
| E0-3 | partial | `LICENSE` (MIT) only; no CONTRIBUTING or templates |
| E0-4 | **done** | `docs/measurements.md`; the model numbers are explicitly marked unmeasured, not guessed |
| E2-1 | **done** | `crates/core/src/upscaler.rs`; identity run reproduces the input within one ULP |
| E2-2 | **done** | exact-length test over lengths 1…5000 × overlaps 0/1/7/512 × chunk sizes 1…4096 |
| E2-3 | **done** | `curve_rise[i] + curve_fall[i] == 1` tested; overlap-add covered by the identity test |
| E2-4 | partial | ported and matches scipy's detected cutoff on the four fixtures, not yet a 20-file corpus |
| E2-5 | **done** | both stages within −132 dB of scipy on all four fixtures; the SOS matrix matches to 3e-16 |
| E2-8 | **done** | two renders from the same seed are bit-identical; noise from Irwin–Hall, no libm |
| E1-1, E1-2, E1-3 | **written, never run** | `tools/export_onnx.py`, `tools/compare_onnx.py`, `tools/flashsr_onnx.py`; no torch on this machine, so the -40 dB gate has no number yet |

The three pipeline bugs the tests caught, recorded so they are not reintroduced: the
windower drained a whole window per render instead of `step`; `finish` lost samples when
a track ended mid-overlap (the reference implementation still does this, see
`docs/risks.md`); and zero-phase filtering squares the DC gain, which is why
`rms_out/rms_in ≈ 0.979` and not 1.0.

## Start here

1. **E1-1 → E1-5.** Nothing else matters until the model runs in Rust at a known
   speed. Book a week, accept the answer.
2. **E0-1 → E0-2.** Scaffold while the export runs.
3. **E2-1 → E2-3.** The pipeline with mocks, tested on the laptop with the GPU idle.
4. **E2-4 → E2-5.** The port, against golden fixtures.
5. **E3-3 → E3-4.** Device out, then the toggle.
6. **E4-4.** The button. That is the product.

E8 can start at any point after E2-5; it is the cheapest insurance in the list.
