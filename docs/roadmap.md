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

Revised 2026-11-13 after it played on real music for a week. Two things moved:
**the UI is the product now**, and **waiting for a whole render before the first note is
not acceptable**. Packaging stays, one release later.

| release | theme | exit criteria | weeks |
|---|---|---|---|
| **v0.0** | kill the risk | model runs, RTF measured, audio out | done |
| **v0.1** | walking skeleton | `play` decodes, enhances, plays; A/B is sample-locked | done |
| **v0.2** | **the player** | first sound in ≤ 2 s, enhanced within ~10 s, a window worth looking at | 2 |
| **v0.3** | **a real app** | `.app` bundle, first-run setup inside the window, signing | 2 |
| **v0.4** | streaming and cache polish | any-length track, seek anywhere, eviction, no full-track buffers | 2 |
| **v1.0** | utility | system-wide sink, batch folder, Windows/Linux builds | 3 |

The old plan put the cache at v0.3 and real-time at v0.4. Both are partly here already
(the cache shipped, seek shipped), and both were the wrong priority next: a player that
takes 95 s to make a sound is not a player, however good its cache is.

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

## v0.2 — the player

Goal: it feels like a player, not a renderer with a window on it.

**Playback starts before the render finishes.** Today `play` decodes, renders the whole
track (95 s for three minutes) and only then opens the device. The renderer is 1.84x
realtime, so there is no reason to wait for it:

1. Decode and start the device immediately. Sound in ~0.5 s, original signal.
2. Render in the background into a growing frontier, and let the mixer ramp to the
   enhanced signal once the frontier passes the playhead. At RTF 0.54 the frontier gains
   about 1.8 s per second, so the toggle is honest within a few seconds.
3. The A/B control shows the frontier: it says "generating" until enhanced audio exists
   at the playhead, then it is live. It never claims a region it does not have.
4. A cache hit skips steps 2 and 3 entirely: both signals are ready at once.
5. Seeking backwards into rendered audio is instant. Seeking ahead of the frontier
   restarts the pipeline at that window and the control says "generating from here".

Exit: first sound ≤ 2 s cold (≤ 1 s from cache), toggle live ≤ 10 s on this machine,
no dropouts in a five-minute session, and a counter in the window to prove it.

**The window.** The current one is a button, a slider and two grey lines.

- Track title/artist from tags, with the file name as the fallback.
- A real transport: play/pause, seek bar with elapsed/total, and a visible playhead.
- The A/B control is the centrepiece: a large, obvious switch that names both sides
  honestly ("generated top band" / "original") and shows which is live.
- A spectrum of both sides overlaid, with the cutoff and the generated band marked. This
  is the one picture that explains the product in a second.
- A status line that says what is happening: rendering (with the frontier), cached, the
  measured RTF, and the peak limiting it applied.
- Drag a file onto the window; errors are states, never silent stops.

**Honesty work that belongs with the toggle.** Loudness-match the two sides (today the
enhanced side is +0.57 dB) and try upstream's low-band splice, so the A/B differs in
brightness and not in level or in something the model regenerated badly. Both are
measured before they ship.

Deliberately absent: playlists, library view, EQ, skins, export UI, system sink.

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

## Gaps found building the prototype (2026-11-12)

The first playable prototype works: `flacify setup`, `enhance` and `play` exist, the
toggle is sample-locked, and a 30.7 s clip renders in 34.1 s and plays through the
output device. These are the gaps it exposed, in the order they hurt. Each one names the
release where it should land.

**v0.2, before anyone else uses it**

1. **No FLAC out.** `enhance` writes 32-bit float WAV; `flacenc` is not wired up, so the
   cache container from ADR-003 does not exist yet. WAV is also 3× larger.
2. **No loudness match.** Measured +0.57 dB on a steady track, so the A/B switch steps in
   level as well as in brightness. That makes the comparison slightly dishonest and it is
   exactly what E2-6 was written for. (Peak protection is now in place: a soft knee above
   0.9, because the model overshot full scale by 2.7 dB on real music. That fixes
   distortion, not the level step.)
3. **No low-band preservation.** The model regenerates the whole waveform: level-matched,
   the output differs from the input by −18.7 dB RMS below 12 kHz and gains +2.9 dB in
   4–12 kHz. Upstream's `mel_replace_ops` splices the input back below the cutoff; try it
   and measure whether listeners prefer it.
4. ~~No seek.~~ **Done (2026-11-13):** the position bar is draggable and the arrow keys
   jump five seconds. Seek is instant because the track is decoded into memory; seeking a
   *streaming* track arrives with the queue in v0.4.
5. ~~Whole-track render before play.~~ **Scheduled for v0.2 (2026-11-13):** the player
   starts on the original and promotes to enhanced as the frontier passes the playhead
   (ADR-012). The mixer, the device path and the cache already carry it; what is missing is
   the producer side.
6. **No visual feedback of its own.** No spectrum and no waveform (ADR-005 called for a
   spectrum). The window is text, a button and a bar.
7. **No full-scale policy beyond the limiter.** The original is played as decoded, so a
   master that already peaks above 1.0 clips at the device exactly as it would in any
   player; the enhanced side is limited to 0.999 while the original is not, which can make
   the toggle differ by a few tenths of a dB on hot material.

**v0.3, or when the prototype becomes something to hand over**

8. **No cache.** Every play re-renders. A second play of the same file costs the same as
    the first, and the seed policy (ADR-006) is only useful once the result is stored.
9. **Setup is verified, but its progress is coarse.** A clean install now runs end to end
   (~5.9 GB, and it caught two real bugs: a non-installable upstream package and two missing
   wheels). What is still missing: per-step progress rather than `step: ... / step: done`,
   resume *inside* a step (a dropped connection restarts that step's download), and a
   check that the installed sidecar actually imports before declaring success. Windows is
   now installed end to end (4.2 GiB on a CPU-only machine) and setup is idempotent; that
   run found the `uv` pre-download bug in gap 11.
10. **The worker is not supervised.** If it dies mid-render the app errors out; there is no
   restart, and a wedged model with no output blocks the render thread forever.
11. **Platform coverage is one machine plus Linux and Windows runs.** The portable-core work is
    done: uv's Windows `.zip` is handled, the audio callback covers every integer and
    float sample format cpal exposes, a refused `BufferSize::Fixed` falls back to the
    device default, the sidecar picks `cuda > mps > cpu`, and a Windows CI job builds and
    tests the workspace. Linux has now run end to end on a real device (CUDA, a 44.1 kHz
    device resampled to 48 kHz, `play --smoke`, A/B enabled), and ONNX Runtime's CUDA EP
    was measured on the same graph: RTF 0.545 fp32, versus 0.453 for torch. Switching to
    it would cut the Linux first-run download from ~6.6 GB to ~2.5–3.6 GB, but it is 20 %
    slower and fp16 weights fail the numeric gate. The sidecar therefore stays on torch
    for the prototype. Windows has now run end to end on a real device as well (CPU
    inference, 48 kHz device, A/B enabled, `play --smoke`), which makes it a renderer
    there: 152 s of wall clock for a 5.12 s clip, and ONNX Runtime's CPU EP is 4.3x
    slower again. DirectML is not the answer there: its newest release (ORT 1.24.4)
    refuses to initialise a graph containing a 1D `ConvTranspose`, which is what the
    vocoder is built from. What is still missing: a Windows accelerator, the rest of the
    device matrix (WASAPI/PipeWire, 96 kHz, 5.1), and a decision on the accelerator.
12. **Metadata is the filename.** No title, artist or duration from tags, no drag and drop.

**Explicitly parked**

13. Playlists, library view, EQ, export UI, system-wide sink, live capture: unchanged from
    the MVP cut below, and none of them are blocked by the gaps above.
14. `flacify probe --rtf` still defaults to 0.3. The app now knows how to measure this for
    real (the render prints its own RTF); folding that into a saved hardware profile is
    v0.2 work.

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
