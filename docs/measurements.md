# Measurements

Every number this repo quotes, with the machine, the command that produced it and the
date. If a number is not here, it is either borrowed (and says so) or it does not exist.

## Machines

| # | what | specs |
|---|---|---|
| M1 | development laptop (2026-11-12, all Rust numbers) | Apple M5 Max, 32-core GPU, 36 GB, macOS 27.2.1, Rust 1.99.1 |
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

## The model on this hardware

**Not measured.** There is no ONNX export yet (`tools/export_onnx.py` has never run;
this machine had no torch until 2026-11-12), so FlashSR's RTF, peak memory and
CoreML execution-provider coverage on M1 are all **unknown**. ADR-002's kill criteria
(sustained RTF ≤ 1.5, ≤ −40 dB ONNX-vs-PyTorch) therefore have no number.

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
