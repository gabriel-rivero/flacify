# Roadmap

Status: draft, 2026-11-16. Estimates assume one developer at roughly 20 h/week, so
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

Exit: first sound ≤ 2 s from cache (met: 0.17 s), ≤ 15 s cold (met: 12.6 s), toggle live
with the first note (met), no dropouts in a five-minute session (not yet exercised).

**Landed so far (2026-11-15):** progressive playback with a prebuffer, automatic promotion
once, a draggable seek bar, the frontier progress bar and the status line; then the two
spectra, title and artist from tags, a device stream-error counter, and drag-and-drop onto
the window (2026-11-15). The model worker is now process-scoped: it starts lazily on the
first render that needs it — a cache hit or a full-band file never starts it — is reused
for every later file in the window, and is killed when the window closes. No daemon, no
idle 3 GB. The first uncached file in a run still pays the load; the second does not
(ADR-013). The waveform landed on 2026-11-16; see the gaps at the end of this file.

**The window.** Everything in this list has landed.

- Track title/artist from tags, with the file name as the fallback.
- A real transport: play/pause, seek bar with elapsed/total, and a visible playhead.
- The A/B control is the centrepiece: a large, obvious switch that names both sides
  honestly ("generated top band" / "original") and shows which is live.
- A spectrum of both sides overlaid, with the cutoff and the generated band marked. This
  is the one picture that explains the product in a second.
- A waveform of the whole track, with the render frontier marked and the unrendered part
  dimmed. **Landed 2026-11-16:** per-second RMS bars in the accent colour where the
  renderer has reached and grey beyond it, a frontier line, a playhead, and click or drag
  anywhere to seek.
- A status line that says what is happening: rendering (with the frontier), cached, the
  measured RTF, and the peak limiting it applied.
- Drag a file onto the window; errors are states, never silent stops. **Landed
  2026-11-15:** a drop replaces the file in place, a failed load keeps the current track
  and names the file it could not open, and the worker is reused across drops.

**Honesty work that belongs with the toggle.** Both landed and both were measured.
Loudness matching brought the two sides within 0.1 dB on a steady track, so the switch now
changes brightness and not level. The low-band splice did not work: with the input's low
band subtracted from the generated side and its own added back, the result correlates
*worse* with the input's low band than the unspliced render (0.32 vs 0.39), so `--splice`
stays off. The numbers are in
[`docs/measurements.md`](measurements.md#alignment-and-the-low-band-splice-2026-11-13).

Deliberately absent: playlists, library view, EQ, skins, export UI, system sink.

## v0.3 — a real app

Goal: a downloaded app that installs and starts without a terminal.

**Landed 2026-11-16 — first-run setup in the window.** `flacify play` checks `is_ready()`;
when the model is missing it opens the window on a setup screen instead of stopping with a
CLI instruction. The screen states the size (the sum of `setup::plan()`), starts
`setup::ensure` on a thread only after the user presses Download, shows the current step and
its progress, offers Retry after a failure, and hands the same window to the player once the
file opens. Cancel stops the running child through a token and keeps the finished steps;
Resume re-enters the state machine and skips them. `--smoke` still requires `flacify setup`,
because a headless run cannot click.

Still to do:

- Installer: `.msi` on Windows, `.app` + notarisation on macOS, AppImage/.deb on Linux.
- Code signing. A `v*` tag already attaches a raw Windows `flacify.exe` from
  `.github/workflows/release.yml`.
- Per-step byte progress (the bar still jumps 0→100 per step).
- A hardware gate: a clear message and passthrough mode on 8 GB Macs or old GPUs.
- Choosing the CUDA torch wheel without `FLACIFY_TORCH_INDEX` (a per-platform default or a
  documented `setup` flag).
- Worker supervision: a dead child is replaced and the window retried once; a model that
  wedges with no output and no error still blocks its caller (gap 10).
- Opt-in diagnostics to a local log file only. No network.

Exit: a clean machine installs from the window, plays a file, and a second run uses the
install without contacting the network.

## v0.4 — streaming and cache polish

This is where the player stops being a demo.

- Segmented FLAC cache, content-addressed by `(file hash, model hash, params)`.
- Background renderer fills it at whatever speed the GPU allows; the player reads
  from it and passes through unrendered regions.
- Seek anywhere; re-render only the segments the settings invalidate.
- A 60-minute track: full render completes in the background, playback never waits,
  a second play starts instantly with the GPU idle.

Exit: 60-minute track, zero dropouts, seek to 55:00 works, second play has 0 s of
GPU time.

## v1.0 — utility

- Real-time mode: what the PoC does today, done with a policy object instead of a separate
  code path — a bounded queue whose depth comes from the measured RTF, latency presets
  0.64 / 1.28 / 2.56 / 5.12 s with the quality tradeoff in the UI, and an underrun policy
  that emits passthrough and never a gap.
- Optional live capture source (input device), which is the same code the system sink needs.
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
2. ~~No loudness match.~~ **Done (2026-11-13):** the generated side was 1.49 LU louder;
   BS.1770 matching brings the step to −0.10 LU, measured twice (the app's meter and an
   independent scipy implementation). One gain per track, taken from the first window, so
   nothing pumps; a track whose opening is unrepresentative gets a slightly wrong gain
   instead. Peak protection (soft knee above 0.9) landed earlier.
3. **Low-band preservation: attempted, does not work, parked.** The splice exists
   (`--splice`, zero-phase filter on both sides, unit-tested on synthetic signals) and it
   still does not restore the input's low band on real music: spliced and unspliced outputs
   correlate with the input's low band at 0.32 and 0.39 (100–6,000 Hz, freshly rendered,
   after latency compensation). The measurements that motivated it, and the ones that
   retired it, are in `measurements.md`.

   The alignment question it raised is answered, and the earlier "wandering lag" was noise:
   - the pipeline is exact (an impulse comes out at its own index, tested);
   - the preprocessing round trip is exact too (the 62-sample shift I first measured was an
     artefact of a rectangular burst; a single-sample impulse lands on itself);
   - the model is late by a fixed **12 samples** (0.25 ms), measured twice, which the player
     now compensates because the model declares its own latency.

   So the sides are sample-aligned to a quarter of a millisecond. The remaining reason the
   splice fails is that the model's low band is not a delayed copy of the input's — it is a
   regenerated signal with its own phase (correlation ~0.4), and subtracting one and adding
   the other leaves a mix of both. A working version means doing it where upstream does, in
   the mel domain, or a linear-phase crossover much steeper than the current Chebyshev. Not
   worth either until the product needs it.
4. ~~No seek.~~ **Done (2026-11-13):** the position bar is draggable and the arrow keys
   jump five seconds. Seek is instant because the track is decoded into memory; seeking a
   *streaming* track arrives with the queue in v0.4.
5. ~~Whole-track render before play.~~ **Done (2026-11-13):** the player starts on the
   original and promotes itself once the frontier passes the playhead (ADR-012); 0.17 s
   from cache and 12.6 s cold on the M5 Max, then the switch is live. The resident worker
   landed on 2026-11-15 (ADR-013): it is started by the first render that needs it, reused
   for later files, and dropped with the window. The second file therefore skips the load;
   how much that saves is **not measured yet** — 12.6 s includes the 8 s load, but the
   first window has not been timed alone on this machine.
6. ~~No waveform.~~ **Done (2026-11-16):** the whole track is drawn as per-second RMS
   bars, bright where the renderer has reached and grey past the frontier, with a
   frontier line and a playhead; click or drag anywhere in it to seek. It shows where the
   generated audio exists, not what it sounds like — the gaps are at the end of this file.
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
   now installed end to end (4.2 GiB on a CPU-only machine, 6.9 GiB on W2 with the CUDA
   wheels) and setup is idempotent; that run found the `uv` pre-download bug in gap 11.
10. ~~The worker is not supervised.~~ **Partly done (2026-11-16):** a worker that dies
    between windows is dropped and replaced, and that window is retried once on the fresh
    child, so one death costs a load rather than the track (tested with the fake sidecar).
    A model that wedges with no output and no error still blocks its caller; catching that
    needs a timeout on the exchange, not just a restart.
11. **Platform coverage is one machine plus Linux and Windows runs.** The portable-core work is
    done: uv's Windows `.zip` is handled, the audio callback covers every integer and
    float sample format cpal exposes, a refused `BufferSize::Fixed` falls back to the
    device default, the sidecar picks `cuda > mps > cpu`, and a Windows CI job builds and
    tests the workspace. Linux has now run end to end on a real device (CUDA, a 44.1 kHz
    device resampled to 48 kHz, `play --smoke`, A/B enabled), and ONNX Runtime's CUDA EP
    was measured on the same graph: RTF 0.545 fp32, versus 0.453 for torch. Switching to
    it would cut the Linux first-run download from ~6.6 GB to ~2.5–3.6 GB, but it is 20 %
    slower and fp16 weights fail the numeric gate. The sidecar therefore stays on torch
    for the prototype. Windows has now run end to end twice: on W1 with CPU inference
    (48 kHz device, A/B enabled, RTF 7.7, ONNX Runtime's CPU EP 4.3x slower again), and on
    W2 — the 4060 Ti box — with torch from PyTorch's `cu132` index, where it is no longer
    a renderer but a player: RTF 0.66, and an uncached 45 s `play --smoke` reported the
    switch offered and promoted at 25.4 s, 22 s of which is the model load. That is opt-in
    through `FLACIFY_TORCH_INDEX`, not a default, and the run found two setup bugs that are
    now fixed (the index did not upgrade an existing CPU install, and setting it made a
    good install look missing). DirectML is still not the answer: its newest release (ORT
    1.24.4) refuses to initialise a graph containing a 1D `ConvTranspose`, which is what
    the vocoder is built from. What is still missing: choosing the CUDA wheel without an
    environment variable (a per-platform default, or a documented `setup` flag), the rest
    of the device matrix (WASAPI/PipeWire, 96 kHz, 5.1), and a decision on the accelerator.
12. **Metadata is partial.** Title and artist come from tags, with the filename as the
    fallback; duration comes from the decode. There is no artwork and no album or year.
    Drag and drop landed on 2026-11-15, and it replaces the current file in place.

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

## Gaps found building the waveform (2026-11-16)

The waveform landed as per-second RMS bars with the render frontier marked, a playhead,
and click-or-drag seek. It answers "does the generated audio exist here" and nothing more.
What it does not do, in the order it would hurt:

1. **It draws the original only.** The generated side has no shape on screen; the spectrum
   is still the only place the enhanced band appears. For the A/B story an overlay — the
   original as an outline, the generated as a fill, or two RMS curves — would show *what*
   changed, not just where the renderer is.
2. **The frontier moves in one-second steps.** It is `Frontier::ready_frames()`, published
   a second at a time, so the line jumps. That is the playback granularity, not the
   model's 5.12 s window, and the bar that contains the frontier is already bright.
3. **Bars cap at 512.** A track much longer than ~8.5 minutes at 48 kHz trades per-second
   detail for legible bars (the cap is one bar per pixel), so the architecture's
   "per-second" is only true up to that length. Zoom or a min/max envelope would fix both
   the cap and the loss of detail.
4. **It is computed at device rate from the resampled buffer.** A 44.1 kHz output device
   resamples before the mix, so the picture follows the device buffer, not the 48 kHz
   source. The shape is right; the sample clock is the device's.
5. **"Rendered" is one bit.** A cached track is all bright, a full-band source is all
   accent with no frontier, and a partial cache (once the segmented cache lands) would
   need its own state to say "rendered but not cached" or vice versa.
6. **The window grew to 560×560 and is still not resizable.** That fits the added
   waveform on this font; a larger text scale or a long status line can clip, and there
   is no scroll.
7. **Seeking is whole-track, not precise.** Barrier: the bar width is the finest gesture
   (~0.2–13 s depending on length), the arrow keys still jump five seconds, and there is
   no zoom or fine scrub.
8. **No waveform for live or streaming sources.** The bars come from a buffer decoded into
   memory; the v0.4 queue and the capture source have no equivalent yet.

## Bug found testing a cached render (2026-11-16)

A cache hit of a 619 s track played its **right channel from the wrong position**: the
left was correct for the first 4.12 s (one render step), then both channels were a mesh of
different blocks. The cause was not the model or the mixer. `enhance` and
`render_into_frontier` appended each pipeline window's `[left, right]` chunk whole, so
`Render.planar` was *block*-major, while `cache.store`, `wav::write` and
`interleave_stereo` all read a track as *channel*-major. Cold playback hid it because the
frontier interleaves one block at a time; only the cache and the `enhance` WAV exposed it.
The bug predates the worker and waveform work (`3963196` and earlier).

**Fixed:** a `ChannelMajor` collector joins the per-block chunks the way the pipeline's own
tests do (`flatten` in `crates/core/tests/pipeline.rs`), with a test that fails on the old
behaviour. **`CACHE_VERSION` is 6**; every v5 entry misses and is re-rendered once.

**Left over:** the v5 entries are not deleted (eviction is E5-5), so a track that was cached
before this fix keeps its scrambled copy on disk until eviction lands. The next play of the
FFVII track has to render again, and the second-file time to first sound is still unmeasured.

## Gaps in the in-window setup (2026-11-16)

The setup screen covers the thing that made a downloaded exe useless without a terminal:
the size is stated, the download starts on a button press, the step and its fraction are
shown, a failure can be retried, and the window becomes the player. What it still does
not do, in the order it would hurt:

1. ~~No cancellation.~~ **Done (2026-11-16):** a `Cancel` token runs through the state
   machine and the real runner polls it while the child is alive, so Cancel (or closing the
   window) kills the running `uv`/`curl`/`pip` step and keeps the finished ones. One gap
   remains: a child that leaves a grandchild holding the pipes can delay the join.
2. **The progress bar jumps, it does not move.** `Progress` reports a step at 0 and at 1,
   so the bar is empty or full for each step. Byte-level progress already exists inside the
   runner; it is not threaded out.
3. **Retry is coarse.** It re-enters `ensure` from the top, skipping steps already marked
   done. It cannot retry only the failed step, and there is no backoff between attempts.
4. **A cached or full-band file still triggers setup.** `Session::new` checks `is_ready`,
   and the window offers setup before opening any file, even one that would never start the
   model. The app could defer the requirement to the first render that needs it.
5. **No accelerator choice on the screen.** The plan installs PyPI's CPU torch; a CUDA
   machine still has to set `FLACIFY_TORCH_INDEX` before the first run. The screen should
   detect and offer the CUDA wheel (gap 11).
6. **Verification is size-only.** The plan's bytes label the download; the weights step
   checks sizes, not a checksum. The CLI has the same limitation (E7-1).
7. **The window is fixed at 560×560 and does not scroll.** A long failure message can push
   the Retry and Quit buttons out of the viewport.
8. **`play` is the only entry point.** `flacify enhance` and `probe` still say "run
   `flacify setup`", and `--smoke` bails for the same reason: a headless run cannot click.
9. **Only the state machine is tested.** The screen's transitions are unit-tested; the
   download path is not exercised end to end without a network and 6.9 GB.
