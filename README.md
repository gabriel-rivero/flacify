# flacify

A desktop player that reconstructs the high frequencies an MP3 threw away, and lets
you hear the difference on a single keystroke.

Status: **1.1.4**, playable on macOS, Linux and Windows. `flacify setup` installs the
Python sidecar and the model, `flacify enhance` renders a file, `flacify play` opens a
window with an instant A/B switch, and `flacify play --live` renders a file while it plays
from bounded memory. `flacify sink` does the same for system audio on macOS. The Rust side
owns decode, analysis, the windowing pipeline, the seed schedule and the mixer; the model
runs in a Python process the app installs and owns (ADR-010), because ONNX Runtime measured
30 to 60x slower than PyTorch on the same Apple Silicon (ADR-002). The working reference
implementation is the Python PoC,
[`gabriel-rivero/flashsr-stream`](https://github.com/gabriel-rivero/flashsr-stream).

`play` no longer waits for a whole render: it starts the original as soon as the device
opens and fills a frontier behind it, 0.17 s from cache and 12.6 s cold on the M5 Max
(model load plus the first window), then hands the A/B switch over once the frontier is
ahead. The model worker is process-scoped now: it starts lazily on the first uncached file,
is reused when another file is dropped on the window, and is killed when the window closes,
so an idle app loads nothing. A full-band file or an already-cached render plays with no
model installed at all; only a file that has to render opens the setup screen. The window is
complete for v0.2: the spectrum and the waveform both show the render frontier, and the
waveform is the seek surface. What remains is the five-minute dropout soak and the
second-file timing. The numbers are in
[`docs/measurements.md`](docs/measurements.md#progressive-playback-2026-11-13).

Runs end to end on macOS (verified: Metal, 48 kHz out, A/B enabled), on Linux
(verified: CUDA on an RTX 4060 Ti, a 44.1 kHz device resampled to 48 kHz, A/B enabled) and
on Windows (verified: CUDA on the same RTX 4060 Ti under Windows 11, 48 kHz out, A/B
enabled, RTF 0.66). That makes Windows a player on that box, not just a renderer: a 45 s
track that had never been rendered reported the switch offered and promoted at 25.4 s, the
first 22 s of which is the model load. A CPU-only Windows machine is still a renderer — the
handheld in the measurements ran the render at RTF 7.7 — and the accelerator is `cuda` if
present, `mps` on Apple Silicon, otherwise `cpu`, which is a renderer rather than a player
(RTF 6–10). The CUDA wheel is opt-in (`FLACIFY_TORCH_INDEX`, see
[`docs/dependencies.md`](docs/dependencies.md)); PyPI's torch is CPU-only on Windows and
Linux. On Linux the same graph through ONNX Runtime's CUDA EP reaches RTF 0.545 and would
cut the first-run download roughly in half, but the sidecar still runs torch; the numbers
are in
[`docs/measurements.md`](docs/measurements.md#linux-the-download-and-whether-onnx-runtime-can-shrink-it-2026-11-12).
See the platform gap in [`docs/roadmap.md`](docs/roadmap.md).

## What works today

```sh
cargo build --release
./target/release/flacify setup                 # once, 4-7 GB; the window does it on first play too
./target/release/flacify setup --cuda          # the CUDA torch wheel on an NVIDIA machine
./target/release/flacify probe song.mp3        # cutoff, verdict, wait
./target/release/flacify enhance song.mp3      # -> song.mp3.flacify.wav
./target/release/flacify                       # open the window; drop a file, folder, playlist or paste a URL
./target/release/flacify play song.mp3         # window: space toggles, p plays, drag to seek, drop more
./target/release/flacify play ~/Music          # every audio file under a folder, in path order
./target/release/flacify play mix.m3u          # an .m3u/.m3u8/pls playlist
./target/release/flacify play https://youtu.be/ID # the same, for a URL: yt-dlp fetches the audio first
./target/release/flacify play --radio https://youtu.be/ID # the song's start_radio mix, as a queue
./target/release/flacify play https://www.youtube.com/playlist?list=ID # a YouTube playlist
./target/release/flacify enhance https://youtu.be/ID  # render a URL's audio; --out names the file
./target/release/flacify play song.mp3 --live  # bounded memory: one window at a time, latency stated
./target/release/flacify play song.mp3 --live --window 4
./target/release/flacify play song.mp3 --smoke # headless audio-path check
./target/release/flacify ab song.mp3           # blind A/B trials -> song.mp3.flacify-ab.md
./target/release/flacify batch ~/Music         # render a folder; skips cached files, writes batch-last.md
./target/release/flacify batch . --dry-run     # what would render, and why each file is skipped
./target/release/flacify sink --from "BlackHole 2ch" --to "External Headphones"  # capture -> enhance -> play
./target/release/flacify devices               # capture and playback devices, with rates and channels
./target/release/flacify diagnose              # -> flacify-diagnose.txt
./target/release/flacify cache                 # what the cache holds, by kind, against its 20 GB cap
./target/release/flacify cache --evict         # evict least recently used entries until it fits
```

`play` takes a queue: a folder, an `.m3u`/`.m3u8`/`.pls`, a YouTube playlist, or
`--radio <video>` for that song's recommendations. Tracks advance at the end, `N`/`B` move
through them, and the next remote track is fetched while the current one plays. A single
file is a queue of one. `flacify` with no argument opens the same window empty so a queue
can be dropped onto it, and both the empty window and the player carry a URL box: paste a
link and it becomes the queue, or — with `radio` ticked — seeds a mix from its video, which
is how the packaged app reaches a YouTube source with no command line. `flacify setup`
installs `yt-dlp` for that into the app's own directory, so a URL works with nothing else
on the machine; it asks yt-dlp for a FLAC transcode when `ffmpeg` is present and otherwise
keeps the container YouTube already serves (AAC in MP4), which needs no conversion. Neither
tool has to be on your `PATH`, and nothing is added to it.

`flacify sink` captures one device, enhances it, and plays it on another: it is the same
bounded path as `play --live`, with the capture stream as the feed. macOS has no loopback of
its own, so install BlackHole and route your player's output to it; Windows has no cpal
loopback in 1.0 and is unsupported there. Capture also needs Microphone permission for the
app that launched the sink: without it macOS returns digital silence rather than an error, so
the sink treats five seconds of silence as that state, says so, and refuses to start. The two
devices run on different crystals, so the sink trims its read rate by up to 1000 ppm and
resyncs with a 10 ms crossfade when that is not enough; the status line prints both.

`flacify ab` renders or loads the generated side once, plays 20 six-second excerpts in a
seeded order, and asks which of the two was the generated one. The seed is in the report,
so the run can be replayed. The report states the measured level match, and the command
exits non-zero if any excerpt's two sides differ by more than 0.5 dB: an unmatched pair
measures loudness, not the model. The score is a listening test, so the operator's answers
are the evidence, not the program's.

Measured on an M5 Max: a 30.7 s clip renders in 34.1 s (including the 8 s worker start)
and plays back at 48 kHz with the switch enabled. The generated band gains +24.6 dB at
16–18 kHz and +61.6 dB at 18–22 kHz over the input. That measurement predates loudness
matching, which now holds the two sides within 0.1 dB so the toggle changes brightness,
not level (see
[`docs/measurements.md`](docs/measurements.md#loudness-matching-2026-11-13)).

## Releases

Pushing a `v*` tag runs [`.github/workflows/release.yml`](.github/workflows/release.yml),
which builds and attaches:

- **Windows:** `flacify-setup.exe` (per-user install, Start Menu shortcut) and the raw `flacify.exe`.
- **macOS:** `flacify.dmg` containing `Flacify.app`.
- **Debian and Ubuntu:** `flacify_1.1.4_amd64.deb` (`sudo apt install ./flacify_1.1.4_amd64.deb`).
- **Arch and CachyOS:** `flacify-bin-1.1.4-1-x86_64.pkg.tar.zst`
  (`sudo pacman -U flacify-bin-1.1.4-1-x86_64.pkg.tar.zst`). The binary needs only alsa,
  libc and libgcc to link; GL, X11 and Wayland are opened when the window starts.

Nothing is signed or notarised yet: on macOS, right-click → Open (or
`xattr -dr com.apple.quarantine /Applications/Flacify.app`) the first time. The first
`flacify play` opens a setup screen and downloads the Python sidecar, wheels and weights
(about 6.9 GB with the CUDA wheels); `flacify setup` does the same from a terminal. A
full-band file or an already-cached render plays without the model.

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
| [`docs/architecture.md`](docs/architecture.md) | crates, modules, stages, threads, buffer math, A/B alignment, cache |
| [`docs/dependencies.md`](docs/dependencies.md) | crates in the tree and considered, model runtime, artifacts, licences |
| [`docs/risks.md`](docs/risks.md) | risk register, kill criteria |
| [`docs/backlog.md`](docs/backlog.md) | epics and issue drafts, in build order |
| [`docs/decisions.md`](docs/decisions.md) | ADR-001 … ADR-016 with status and outcomes |
| [`docs/measurements.md`](docs/measurements.md) | every number, with the machine and the date behind it |
| [`docs/release-1.0.md`](docs/release-1.0.md) | the release checklist, and the evidence that closes each line |

`tools/check_docs.py` is the gate over all of the above: it holds the documented ADR
range, crate list, dependency list and status lines to the repository, and CI fails on
any page that drifts.

## Start here

```sh
cargo test --workspace --all-targets   # 538 tests, no GPU, no audio device, no model, no Python
python3 tools/check_docs.py --tests    # the docs still describe this tree
```

`core` and `infer` are testable without a model, which is what makes the DSP and the
mixer developable on any laptop. The Python sidecar is the one seam that needs the
6 GB install (6.9 GiB with the CUDA wheels), and even it is tested against a fake worker
binary in CI.

`docs/decisions.md` records the accepted ADRs and, for ADR-002, the measurement that
reopened it.

## Reference material

- [`flashsr-stream`](https://github.com/gabriel-rivero/flashsr-stream) — the Python
  implementation and its measured numbers (the PoC this repo rewrites)
- [`docs/measurements.md`](docs/measurements.md) — every number in this repo, with the
  machine and date behind it
