# Roadmap

Status: current for v1.0, 2026-11-18. Estimates assume one developer at roughly 20 h/week, so
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
| **v0.3** | **a real app** | `.app` bundle, first-run setup inside the window (signing deferred) | 2 |
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

**Landed 2026-11-16 — a real app.** First-run setup in the window (size, Download, Cancel,
Resume, per-step byte progress, a CUDA checkbox), full-band and cached files playing without
the install, a worker that restarts after a death or a wedge, `flacify setup --cuda`,
`flacify diagnose`, and packaging: the release workflow builds a per-user Windows `flacify-setup.exe`, an
unsigned macOS `.dmg` with `Flacify.app`, and a Linux `.deb`.

Signed by nobody: the `flacify-setup.exe`, `.dmg` and `.deb` are unsigned, and macOS
Gatekeeper needs right-click → Open (or `xattr -dr com.apple.quarantine`). Signing waits
on a certificate and is deferred, with the rest of the v0.3 leftovers, to
[Not on the roadmap](#not-on-the-roadmap).

Exit: a clean machine installs from the window, plays a file, and a second run uses the
install without contacting the network — met, with unsigned artifacts.

## v0.4 — streaming and cache polish

This is where the player stops being a demo. The plan, the decisions and the evidence are
in [`v0.4-plan.md`](v0.4-plan.md).

- Segmented FLAC cache, content-addressed by `(file hash, model hash, params)`.
- Background renderer fills it at whatever speed the GPU allows; the player reads
  from it and passes through unrendered regions.
- Seek anywhere; re-render only the segments the settings invalidate.
- A 60-minute track: full render completes in the background, playback never waits,
  a second play starts instantly with the GPU idle.

Exit: 60-minute track, zero dropouts, seek to 55:00 works, second play has 0 s of
GPU time.

**Released 2026-11-17 as 0.4.0.** The FLAC blocker is solved: `flacenc` 0.5.1 writes
`STREAMINFO.min_blocksize` as the short last frame instead of the block size, so symphonia
cannot probe its output; restoring the field after encoding through the public API makes
the file exact in symphonia and claxon and clean under `flac -t`, no fork. Segments are 7
render steps (28.84 s), because a segment start must sit on the window lattice.

What 0.4.0 does:

- A render is written as 24-bit FLAC segments behind a manifest, each readable as soon as it
  is finished, so a killed render resumes. `CACHE_VERSION` 9.
- A source tier (`cache/source/<hash>/`) holds the decoded 48 kHz stereo source in the same
  segments; the cache key is the source file's bytes, not a decode.
- The renderer works a segment at a time and holds about 25 MB, against 1.38 GB per hour.
- A track with a complete entry plays from three resident segments: 66 MB for an hour,
  seek is a segment read, and the A/B switch runs the same arithmetic as a whole-track render.
- `flacify enhance -o out.flac` writes 24-bit FLAC.

What 0.4.0 does not do, and the next release needs to:

- **The first play of an uncached track still holds the whole track and a never-released
  frontier**, about 1.4 GB each per hour. Only the complete-entry path is bounded.
- **No eviction, and no accounting of disk use.** Every render is kept, and `flacify diagnose`
  counts the source tier as one entry.
- **The waveform and the spectrum are blank** on the segment path, because the buffers they
  read no longer exist.
- **No 60-minute acceptance run.** The memory figure is measured on a synthetic entry; the
  end-to-end run with a real hour of audio, zero-dropout counting and a 55:00 seek has not
  happened.

Those four are in [Not on the roadmap](#not-on-the-roadmap) with the trigger that revives
each one.

## v1.0 — utility

The first release where audio arrives while it is still happening, and the first one
worth pointing at a library rather than a file. [`v1.0-plan.md`](v1.0-plan.md) is the
working plan: decisions, risks and waves.

- **`play --live [--window 1|2|4|8]`**: one live pipeline with two feeds (a file and a
  capture stream), bounded wet/dry rings, credit backpressure, and passthrough with a
  dropout counter on starvation. Latency presets are 0.64 / 1.28 / 2.56 / 5.12 s, each
  labelled with the latency it costs. Opt-in: the default play of an uncached file keeps
  the progressive path (D13).
- **`flacify sink --from <loopback> --to <device>`**: macOS only, routing through
  BlackHole or whatever loopback the user installed. No driver ships, nothing is bundled,
  Windows is documented as unsupported (D4).
- **`flacify batch <dir>`**: sorted walk, skip a complete entry, isolate failures, keep
  going, resume after a kill, exit non-zero, skip list next to the cache.
- **`flacify ab <file>`**: blind trials with randomised order and a fixed replay seed,
  promotion disabled for the run, a report that states the measured level match.
- **Waveform and spectrum on the segment path** (landed 2026-11-18): a cache hit draws the
  whole track from the source tier's one-second peaks and the spectrum from the resident
  segment, so neither reads a buffer the segmented path does not have.

1.0 also closes what 0.4 left open: cache eviction and accounting (V4-7), the waveform and
spectrum on the segment path (V4-9), and the 60-minute acceptance run (V4-10). It does
not include bounded memory on the default first play or playhead-priority rendering;
both wait for 1.1, and the release notes say so.

The build work is single-sourcing the version, adding a macOS CI job, and packaging
parameters — not adding platforms, which 0.3 and 0.4 already shipped.

## Gaps found building the prototype (2026-11-12)

The first playable prototype works: `flacify setup`, `enhance` and `play` exist, the
toggle is sample-locked, and a 30.7 s clip renders in 34.1 s and plays through the
output device. These are the gaps it exposed, in the order they hurt. Each one names the
release where it should land.

**v0.2, before anyone else uses it**

1. ~~No FLAC out.~~ **Partly done (2026-11-17):** `enhance` writes 24-bit FLAC when the
   output name ends in `.flac`, and 32-bit float WAV otherwise, which is 3× larger than
   FLAC. The cache writes FLAC segments behind a manifest, keyed by the blake3 of the
   source file's bytes, so a lookup no longer decodes the track first (V4-3 in
   [`v0.4-plan.md`](v0.4-plan.md)).
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
10. ~~The worker is not supervised.~~ **Done (2026-11-16):** a worker that dies between
    windows is dropped and replaced, and that window is retried once on the fresh child.
    Each exchange also runs under a 120 s watchdog, so a model that stops answering is
    killed and handled by the same retry instead of blocking forever (tested with the fake
    sidecar's `--hang` and `--fail-after`).
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
2. ~~The progress bar jumps, it does not move.~~ **Done (2026-11-16):** a monitor walks the
   step's target directory every 400 ms and reports `used / planned` bytes, so the bar moves
   through `uv`, the venv and the weights. It is an estimate: the plan's byte counts are
   approximate, so a step can read 100% before it finishes (clamped), and the walk is I/O
   the step would not otherwise pay.
3. **Retry is coarse.** It re-enters `ensure` from the top, skipping steps already marked
   done. It cannot retry only the failed step, and there is no backoff between attempts.
4. ~~A cached or full-band file still triggers setup.~~ **Done (2026-11-16):** `Session`
   records whether the model is installed, and only the branches that render require it.
   Full-band files play with no model; cache hits are found by audio and settings when the
   model id cannot be computed, and older entries are annotated on the next ready play.
   `--smoke` reports the requirement instead of opening a window.
5. ~~No accelerator choice on the screen.~~ **Done (2026-11-16):** the setup screen offers
   the CUDA wheel, and `flacify setup --cuda` covers the CLI. Nothing auto-detects an NVIDIA
   GPU yet, so the box is manual.
6. **Verification is size-only.** The plan's bytes label the download; the weights step
   checks sizes, not a checksum. The CLI has the same limitation (E7-1).
7. **The window is fixed at 560×560 and does not scroll.** A long failure message can push
   the Retry and Quit buttons out of the viewport.
8. **`play` is the only entry point.** `flacify enhance` and `probe` still say "run
   `flacify setup`", and `--smoke` bails for the same reason: a headless run cannot click.
9. **Only the state machine is tested.** The screen's transitions are unit-tested; the
   download path is not exercised end to end without a network and 6.9 GB.

## Not on the roadmap

Everything below is real and none of it is scheduled. The first table waits on a trigger;
the second has no trigger yet.

### Deferred, with the trigger that revives it

| item | why it waits | what brings it back |
|---|---|---|
| Playhead-priority rendering and seek-during-render (V4-6 remainder) | The renderer walks segments in order, so a seek into an unrendered part of a track renders everything before it. `docs/v1.0-plan.md` D14 keeps it out of 1.0. | the first report of a long seek during a cold render, or 1.1 |
| Bounded memory on the first play (V4-8c) | The progressive path holds the decoded track and a frontier that never releases a published second: about 1.4 GB each per hour. Only a complete entry is bounded. 1.0's bounded rings serve `--live`, not the default path (D13). | 1.1, after `--live` has evidence |
| macOS signing and notarisation (E7-4) | Needs a paid Apple Developer certificate. The unsigned `.dmg` installs with right-click → Open, or `xattr -dr com.apple.quarantine`. Deferred 2026-11-17; it will be paid for eventually. | the certificate, or a release handed to anyone who will not read the README |
| Windows and Linux code signing | Same shape: a certificate and a price, and SmartScreen/AV warnings do not block the install. | the certificate |
| RAM/VRAM detection for the hardware gate | The CPU warning and passthrough cover the case that hurts (v0.3 exit). | a report of the app wedging a low-memory machine |
| Auto-detecting an NVIDIA GPU | The CUDA checkbox and `flacify setup --cuda` cover it, and a wrong guess downloads 2.7 GiB more. | a second CUDA user, or a wrong guess becoming likely |
| Resume inside a setup step | Per-step progress landed and a dropped connection restarts only that step, not the install. | a flaky connection long enough to lose a whole step |
| End-to-end setup test | Needs a network and 6.9 GB of weights in CI. | a hosted runner with a warm cache, or a nightly job |
| Checksum verification of the weights (E7-1) | Size checks caught the real download bugs so far. | a download that passes its size and is still wrong |
| Linux AppImage (E7-5) | The `.deb` installs and plays. | a non-Debian Linux user |
| `CONTRIBUTING.md`, issue and PR templates (E0-3) | One developer, no outside contributors. | a first outside contributor |
| `flacify probe --rtf` still defaults to 0.3 | The app measures RTF for real on every render; the saved hardware profile has no consumer yet. | the RTF-by-GPU table, below |

### Later, no trigger

Playlists and library management; Winamp `.wsz` skin support; EQ and DSP chain;
VST/DAW plugin; alternative models behind the `Upscaler` trait (AERO, AudioSR,
AP-BWE); a local HTTP + MCP server so agents and other tools can call the enhancer;
multi-channel and 44.1 kHz-native models; a public RTF-by-GPU compatibility table fed
by opt-in reports.
