# Vision

Status: draft, 2026-11-12.

## One sentence

A player that makes lossy music sound like it came from a better source, and proves
it to your ears in one keystroke.

## The problem, stated honestly

Lossy encoders discard everything above a cutoff (typically 16 kHz for a 128 kbps
MP3, higher for modern codecs) and quantise what remains. Bandwidth extension
models reconstruct plausible high-frequency content from the surviving low band.
They work well on music, which is why the spectrogram after enhancement looks like
a real full-band recording instead of an empty top half.

They do not recover your master. The generative part is a guess that is
statistically consistent with the low band. On most material the guess is good
enough that listeners cannot reliably tell enhanced from native lossless in a blind
test; on transient-heavy or already-bright material it can be worse than doing
nothing. Both outcomes are real, which is why flacify ships a blind A/B switch
instead of a marketing claim.

## What we claim, and what we refuse to claim

| claim | verdict |
|---|---|
| "The output is a 48 kHz FLAC" | true. FLAC is the container, nothing more. |
| "It sounds fuller on lossy sources" | testable, and the thing we measure. |
| "It restores the lost high frequencies" | **no.** They are generated. |
| "MP3 becomes lossless" | **no.** Information theory does not allow it. |
| "It sounds better on everything" | **no.** Already-full-band input is passed through unchanged. |

That last row is a product rule, not a caveat: if the detected cutoff is at or near
Nyquist, the enhancer is bypassed. Hallucinating a high band onto a FLAC file is
the fastest way to lose the audiophile audience.

## Who this is for

1. **The archive owner.** A local library of live recordings, vinyl rips, video
   rips, out-of-print releases, old 128 kbps downloads. The good version does not
   exist anywhere to buy or stream, so reconstruction is the only option.
2. **The curious listener.** Wants to hear what the model does to a familiar track,
   on their own headphones, in ten seconds. This is the demo, and it is what makes
   people install the thing.
3. **The tinkerer.** Wants a system-wide enhancer for everything, not just a player.

We are not competing with Spotify or Apple Music on catalogue, and we are not
building a library manager. Those fights are already lost or already won by
someone else.

## Principles

1. **Local first.** No account, no upload, no telemetry by default. The model runs on
   the user's machine; files never leave it.
2. **Prove it, don't claim it.** The A/B toggle is the product. Every quality claim
   in the UI links to a measurement in `docs/`.
3. **Degrade to passthrough, never to stutter.** Any condition that prevents
   enhancement (no GPU, thermal throttle, unrendered region, unsupported file)
   plays the original and says so. Silence and glitches are the only unacceptable
   failures.
4. **Be honest about the generated band.** Show it: the spectrum view shades the
   reconstructed region. Users who understand what they are hearing trust the tool
   more, not less.
5. **Cheap to start, cheap to leave.** First audio fast, no account, no lock-in. The
   cache is a plain FLAC file the user owns.

## Experience

One window. Drop a file on it. Playback starts. A spectrum analyser shows the top
octave filling in as the renderer catches up. `A` toggles original/enhanced with no
gap and no volume jump. The waveform underneath shows which regions are enhanced and
which are still passing through. Nothing else on screen unless you ask for it.

That is the whole MVP. Everything in [`roadmap.md`](roadmap.md) beyond it is optional.

## Where this could go

- **System-wide enhancement.** A virtual output device that any player can route
  into. This is the feature that makes it a utility rather than a demo, and it is
  also the one with the hardest platform work.
- **Library batch.** Render a folder overnight, keep the results in a sidecar cache,
  play them with zero GPU cost on the second listen.
- **Other models.** FlashSR is one point on a quality/speed curve. A `Upscaler` trait
  plus a model registry lets better models arrive without rewriting the player.
- **A local API.** HTTP and MCP endpoints so other tools and agents can call the
  enhancer. Cheap to add once the pipeline is a library, and it turns flacify into
  infrastructure.

## Success, measured

The MVP succeeds if all of these hold on the target machine (M5 Max, macOS):

- Cold start to first audio under 10 s, warm start under 2 s.
- Toggle latency under 30 ms, no click, no gap, no level jump.
- A 60-minute track plays to the end with zero dropouts.
- A blind A/B on five reference tracks is preferred for the enhanced version on at
  least four, and never rated worse on an already-lossless track.
- No crash, hang, or silent failure on a machine without a usable GPU.
