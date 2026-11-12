# Risks

Status: draft, 2026-11-12. Likelihood and impact are H/M/L. "Trigger" is the signal
that tells you it happened, which is more useful than the score.

## The five that decide whether this ships

**R1 — ONNX export does not reproduce PyTorch (H/H).** The forward pass wraps a
diffusers scheduler, a VAE, a UNet and a vocoder, and starts from sampled noise. Any
one of those can export wrongly and still produce audio that sounds plausible. This is
the single largest technical risk and it is why v0.0 exists. Mitigation: numeric gate
(≤ −40 dB per window against a fixed-seed PyTorch render), plus a spectrogram
difference image a human looks at once per export. Trigger: you cannot get below
−30 dB on one of the five tracks.

**R2 — Apple Silicon is not fast enough (M/H).** RTF on the M5 Max is unknown. The
4060 Ti numbers do not transfer: different width, different precision, different EP.
If sustained RTF lands above 1.0, real-time playback is dead and this becomes a
background renderer with a player attached. That is still a good product, but it is a
different roadmap. Mitigation: measure in week one, on 60 s of continuous inference,
not one window. Trigger: sustained RTF > 1.2.

**R3 — The model is stochastic and nobody noticed (M/H).** The one-step sampler draws
fresh noise per call, so the PoC renders a slightly different track every run.
Inherited naively, that breaks cache hits, A/B reproducibility, and bug reports.
Mitigation: noise as a graph input, seeded from a hash of the file and window index.
Trigger: two renders of the same file differ.

**R4 — The licence does not allow distribution (M/H).** `FlashSR_Inference` has no
LICENSE file and the weights dataset states no terms. "All rights reserved" is the
default. Mitigation: download-at-first-run from upstream, attribution, and ask the
author before v1.0. Trigger: the author says non-commercial, or says nothing and you
are about to ship a signed build.

**R5 — Quality regressions are invisible (M/H).** Without a measurement harness,
a preprocessing port that is 2 dB off sounds fine to the person who wrote it and
wrong to everyone else. Mitigation: port the eval metrics in v0.1, not v0.9; keep a
five-track corpus with native-lossless references; make the A/B report a release
artifact. Trigger: you catch yourself saying "it sounds better to me".

## Register

| id | risk | L | I | mitigation | trigger |
|---|---|---|---|---|---|
| R1 | ONNX export diverges from PyTorch | H | H | fixed-seed numeric gate, spectrogram diff | > −30 dB on a track |
| R2 | Apple Silicon too slow for real time | M | H | measure week one; offline-render fallback is a valid product | sustained RTF > 1.2 |
| R3 | stochastic output breaks reproducibility | M | H | noise as graph input, deterministic seeds | two renders differ |
| R4 | model/weights licence blocks shipping | M | H | download at first run, ask upstream, no bundling | "non-commercial" reply |
| R5 | quality regressions go unnoticed | M | H | eval harness + corpus + A/B report | no metrics in a release |
| R6 | preprocessing port differs from Python | H | M | golden fixtures, bin-exact cutoff test | cutoff differs on any file |
| R7 | thermal throttling on laptops | H | M | sustained 60 s probe; adaptive window; show the number | RTF p95 ≫ p50 after 2 min |
| R8 | CoreML EP falls back to CPU per node | M | H | log the partition, count CPU nodes, try candle-Metal | effective RTF near CPU RTF |
| R9 | underruns produce glitches, not errors | H | M | passthrough on underrun, dropout counter, no blocking writes | any audible gap |
| R10 | A/B desyncs and the comparison lies | M | H | fixed-depth backpressure, disable A/B when depth is short | toggle changes content position |
| R11 | cache grows without bound | M | M | LRU cap, manifest, per-key sizes, show usage | disk complaints |
| R12 | 3.3 GB download fails on flaky networks | H | M | resume, checksum, mirror, manual-file escape hatch | support threads about "download stuck" |
| R13 | macOS notarisation blocks first launch | M | M | Developer ID at v0.5, test on a clean machine | Gatekeeper dialog |
| R14 | loudness matching pumps on dynamic music | M | M | BS.1770 momentary, 1 s smoothing, dynamic-track test | audible breathing on toggle |
| R15 | stereo image shifts (channels enhanced independently) | M | M | evaluate mid/side; keep the stereo tensor joint through the model | narrow/wide A/B difference |
| R16 | hallucination on transients and sibilance | M | M | bypass on full-band input; artifact metrics; per-genre corpus | "sounds metallic/smeared" |
| R17 | audio device matrix (44.1-only, exclusive mode, Bluetooth) | H | M | resample in-app, shared mode default, device diagnostics | clicks or resampler errors |
| R18 | scope creep turns a demo into a media manager | H | M | the MVP cut table is a contract; new features go to `v2+` | playlists appear in the design |
| R19 | Python tooling rots (torch/CUDA drift) | M | L | pin the export environment in a container | export stops reproducing |
| R20 | solo-project bus factor | M | M | docs are the deliverable (this set), one-command `diagnose` | you cannot resume after 3 weeks away |

## Accepted risks

- **No playlist.** Deliberate. It is the fastest way to become a mediocre music
  player instead of a good enhancer.
- **No plugin architecture for models.** The `Upscaler` trait is enough; a registry
  can wait for the second model.
- **Unsigned Windows builds at v1.** SmartScreen warnings cost users; accept it until
  there is a reason to buy a certificate.
- **CPU-only machines get passthrough, not slow enhancement.** A 25–45 minute wait
  per track is not a feature; offer it as an explicit "export overnight" action later.

## What would make me stop

State these up front so the decision is not made in a bad week:

1. Sustained RTF above 1.5 on the M5 Max with no faster runtime in reach, **and** the
   target audience does not accept a background render.
2. The licence says non-commercial **and** the project's reason to exist is
   distribution.
3. The blind A/B shows no preference for the enhanced version on the reference
   corpus. At that point the model is the problem, not the player, and the right move
   is to change models, not polish the UI.
