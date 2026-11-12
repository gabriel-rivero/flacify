# Roadmap

Status: draft, 2026-11-12. Estimates assume one developer at roughly 20 h/week, so
treat them as ±50% and as ordering, not commitments.

## The shape of it

The PoC does three things at once: batch conversion, real-time playback, and a
system-wide sink. All three share one pipeline, so the rewrite should too. What
changes between them is only **where the audio comes from** and **how far ahead the
producer is allowed to run**. Those are two independent axes, and separating them is
what keeps the code from turning into three copies of D''s crossfade loop.

| | file source | live source (capture) |
|---|---|---|
| **render-ahead** (producer unthrottled) | v0.1 — instant seek, zero dropouts | n/a |
| **bounded queue** (producer throttled) | v0.3 — "real-time mode", low memory | v0.4 — system sink |

The MVP lives in the top-left cell. That is not a retreat from the real-time idea;
it is the same pipeline with a different policy, and it is the only cell where you
can guarantee no dropouts on hardware you have not tested.

## Releases

| release | theme | exit criteria | weeks |
|---|---|---|---|
| **v0.0** | kill the risk | ONNX model runs in Rust on the M5 Max, RTF measured, audio out through `cpal` | 1 |
| **v0.1** | walking skeleton | `flacify play file.mp3` decodes, enhances, plays; `A` toggles; no UI | 1–2 |
| **v0.2** | **MVP** | window, spectrum, waveform, instant sample-locked A/B, hardware probe | 2 |
| **v0.3** | render cache | 60-minute track, zero dropouts, seek anywhere, second play costs no GPU | 2 |
| **v0.4** | real-time mode | bounded-buffer streaming from file, underrun → passthrough, latency presets | 2 |
| **v0.5** | first-run | installer, model download with resume + checksum, GPU gate, crash-safe | 2 |
| **v1.0** | utility | system-wide sink, batch folder, docs, signed builds | 3 |

## v0.0 — kill the risk

The whole plan rests on one unproven assumption: that FlashSR exports to ONNX and
runs fast enough on Apple Silicon. Nothing else matters until that is answered.

Work:

1. Export `student_ldm` + VAE decoder + `sr_vocoder` to ONNX from the Python repo.
   Static shapes (1×2×245760), opset 17.
2. Run it through `ort` with the CoreML execution provider, and again on CPU, and
   compare against the PyTorch reference output.
3. Measure RTF: single window, 10-window median, and 60 s sustained (thermal).

**Kill criteria.** Stop and re-plan if any of these hold:

- Numerically the Rust output differs from PyTorch by more than −40 dB relative RMS
  on any of the five test tracks (means the export is wrong).
- Sustained RTF > 1.5 on the M5 Max (means real-time is off the table and this
  becomes an offline tool).
- The vocoder needs ops that neither CoreML nor the CPU provider implements.

Fallback if ONNX fails: ship a Python sidecar for v0.1 to keep the player work
moving, and run the export spike in parallel with a two-week timebox. Do not ship
the sidecar.

## v0.1 — walking skeleton

Goal: audio comes out, and the seam-free windowing survives the port.

- `flacify play <file>`: symphonia decode → 48 kHz float → windower → inference →
  crossfade → `cpal`.
- One thread for inference, one for output, a fixed-depth queue between them.
- `A` on stdin toggles between enhanced and delayed passthrough.
- Identity upscaler (`--mock`) so the pipeline can be tested without a GPU.
- No cache, no seek, no UI.

Exit: a whole track plays without a seam or dropout, on a machine with a GPU and on
one without (`--mock`).

## v0.2 — MVP

This is the release the user asked for. One window, one toggle.

- `egui` window: transport, spectrum analyser, waveform with the rendered frontier,
  A/B button, RTF and dropout readouts.
- Sample-locked A/B: equal-power 20 ms crossfade, loudness-matched, verified by a
  null test in CI.
- Hardware probe on first run, persisted to `profile.toml`; second run skips it.
- Drag and drop a file onto the window.
- Errors become a visible state, never a silent stop.

Deliberately absent: playlists (your call, and correct), library view, EQ, skins,
export UI, system sink, streaming input.

## v0.3 — render cache

This is where the player stops being a demo.

- Segmented FLAC cache, content-addressed by `(file hash, model hash, params)`.
- Background renderer fills it at whatever speed the GPU allows; the player reads
  from it and passes through unrendered regions.
- Seek anywhere; re-render only the segments the settings invalidate.
- A 60-minute track: full render completes in the background, playback never waits,
  a second play starts instantly with the GPU idle.

Exit: 60-minute track, zero dropouts, seek to 55:00 works, second play has 0 s of
GPU time.

## v0.4 — real-time mode

What the PoC does today, done with a policy object instead of a separate code path.

- Bounded queue with a target depth derived from the measured RTF, not a guess.
- Latency presets 0.64 / 1.28 / 2.56 / 5.12 s window, with the quality tradeoff
  stated in the UI.
- Underrun policy: emit passthrough for the missed window, log it, show a counter.
  Never a gap.
- Optional live capture source (input device), which is the same code the system
  sink needs.

## v0.5 — first-run

- Model download: 1.7–3.3 GB, resumable, checksum-verified, resumable across
  restarts, with a mirror and a manual-file escape hatch.
- Hardware gate: clear message and passthrough mode on 8 GB Macs or old GPUs.
- Installer: `.dmg`/`.app` on macOS (notarised), `.msi` on Windows (unsigned v1),
  AppImage/.deb on Linux.
- Opt-in crash reports to a local log file only. No network.

## v1.0 — utility

- System-wide sink: virtual output device, "enhance everything" mode.
- Batch: drop a folder, render to cache, progress, resume, skip list.
- Signed and notarised builds for all three platforms.
- A blind A/B report you can export and share, which is the marketing asset.

## Later, explicitly not scheduled

Playlists and library management; Winamp `.wsz` skin support; EQ and DSP chain;
VST/DAW plugin; alternative models behind the `Upscaler` trait (AERO, AudioSR,
AP-BWE); a local HTTP + MCP server so agents and other tools can call the enhancer;
multi-channel and 44.1 kHz-native models; a public RTF-by-GPU compatibility table fed
by opt-in reports.

## The MVP cut, in one table

| in | out |
|---|---|
| one file at a time | playlists, queue, library |
| one model, one window | model picker, presets |
| A/B toggle | separate volume-matched master |
| spectrum + waveform | skins, themes, visualisers |
| render-ahead buffer | real-time low-latency mode |
| GPU required for enhancement | CPU-only playback (passthrough works) |
| local only | sync, cloud, accounts |
