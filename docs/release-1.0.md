# Release 1.0

The tag is the last step, not the first. Everything below happens before `v1.0.0`
exists, and each line names the evidence that closes it. The work itself is in
[`v1.0-plan.md`](v1.0-plan.md).

## 1. Acceptance runs, with numbers

Each run writes a row to [`measurements.md`](measurements.md) with machine and date. A run
without a row did not happen.

| id | run | evidence | owner |
|---|---|---|---|
| A1 | 60-minute file: zero dropouts, seek to 55:00 after the render, second play 0 s GPU | measurements row | agent |
| A2 | `play --live`, 10 minutes, preset 2: zero dropouts, flat memory | measurements row | agent |
| A3 | `sink`: 10 minutes through BlackHole, drift, A/B works, latency stated | measurements row | agent + human at the speakers |
| A4 | `batch`: a real 20-file folder, resume after a kill, 20 GB cap respected | measurements row | agent |
| A5 | blind `ab` report, one real run | the exported report, attached | human |
| A6 | this checklist, end to end | release page | human |

The GPU lane is exclusive: run these one at a time. A1 and A2 need a quiet machine;
do not run them while a render, a build or a video call is using the GPU.

S1's listening verdict gates the preset list. If a preset was dropped, say so in the
release notes rather than silently offering fewer.

## 2. The version has one source

`Cargo.toml` `[workspace.package] version` is the only place a release number lives.
The release workflow reads it and refuses to continue when it disagrees with the tag.
`flacify.iss` takes `AppVersion` as `/DMyAppVersion=`, and the `.dmg` plist is filled
from the same value.

To release 1.0.0:

1. Set `version = "1.0.0"` in the workspace `Cargo.toml`. Nothing else.
2. `cargo build --release` once locally and confirm `flacify --version` prints `1.0.0`.
3. Commit. Update `README.md` and `docs/roadmap.md` in the same commit if they quote a
   version or a status.

The gate is deliberate: a package whose menu says 0.4.0 is the failure mode this
replaces.

## 3. Dry run before the tag

Push the release commit to a branch, then run the **release** workflow manually
(`workflow_dispatch`) with `version: 1.0.0`. It builds every artifact and publishes
nothing. Check:

- every artifact name carries `1.0.0`: `flacify_1.0.0_amd64.deb`,
  `flacify-1.0.0-x86_64.tar.gz`, `flacify-bin-1.0.0-1-x86_64.pkg.tar.zst` after the Arch
  step;
- `Flacify.app/Contents/Info.plist` reports `1.0.0` in both keys;
- the installer's `AppVersion` is `1.0.0` (Inno reads it from the parameter, so check
  the built setup's properties rather than the script);
- `flacify --version` in each artifact prints `1.0.0`.

A dry run that passes is the only evidence that the packaging is single-sourced. Do not
tag on the strength of reading the workflow.

## 4. Install each artifact

| platform | check |
|---|---|
| macOS (this Mac) | open the `.dmg`, drag to Applications, right-click → Open, `flacify play --smoke` on a cached track, then launch the sink **from the app**, not a terminal: a shell-launched process gets digital silence with no prompt, because the permission belongs to the launching app. The first run asks for Microphone access as flacify. |
| Windows | from the record of 0.4.0: installer runs per-user, menu shows 1.0.0, a cached track plays |
| Linux (deb) | `sudo apt install ./flacify_1.0.0_amd64.deb`, `flacify play --smoke` |
| Arch | `sudo pacman -U flacify-bin-1.0.0-1-x86_64.pkg.tar.zst` |

macOS is the only platform the sink supports. Windows and Linux users get a player and a
renderer; the release notes say so. Nothing is signed: the quarantine removal line stays
in the notes.

## 5. What the release notes must say

The workflow writes the body. Before tagging, read it against the code and delete
anything that is not true of the shipped artifacts:

- live mode is opt-in, and the default first play still holds a whole track in memory;
- the sink needs a loopback device the user installs, and ships no driver;
- Windows loopback capture is unsupported;
- presets that S1 dropped are not offered;
- the weights are the FlashSR author's, carry no stated licence, and are downloaded at
  first run, never bundled.

## 6. Tag and verify

```sh
git tag -a v1.0.0 -m "flacify 1.0: live mode, the sink, batch and the blind report"
git push origin v1.0.0
```

Then watch the run to green, open the release page, and confirm every asset is attached
and named with 1.0.0. If the version gate fails, the tag and `Cargo.toml` disagree; fix
`Cargo.toml`, delete the tag, and retry. Never move a published tag.

## 7. After the tag

- `docs/roadmap.md`: mark v1.0 shipped with the date, and move anything cut into the
  deferred table with its trigger.
- `README.md`: status line, the new commands, and the numbers from the acceptance runs.
- `docs/measurements.md`: if a run was repeated after a late change, keep the shipped
  run and label the earlier one as superseded.
