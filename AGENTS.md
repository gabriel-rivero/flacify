# Working in this repository

Rules for an agent or a human changing flacify. The repo owns its own documentation, and
this file is how a change proves it kept them true.

## The docs are part of the change

`docs/` is not a summary written after the fact; for most readers it is the only
description of the code they will read before the code itself. A change that alters
behaviour, a crate, a dependency, a module, a number or a decision is not done until the
page that describes it is updated in the same commit.

| you changed | update |
|---|---|
| a module's job, or a new module file | `docs/architecture.md` (the Modules tables) |
| what a crate depends on | `docs/architecture.md` (Crates), `docs/dependencies.md` |
| a dependency, added or dropped | `docs/dependencies.md` (both tables) |
| a measured number (RTF, latency, drift, size) | `docs/measurements.md`, with machine and date |
| a design decision, or reversed one | a new ADR in `docs/decisions.md`; supersede, never rewrite |
| a test total | `README.md`, and the counter below agrees |
| what the app does, or how to use it | `README.md` |

## The gate

```sh
python3 tools/check_docs.py            # structural, no build
python3 tools/check_docs.py --tests    # also the counted test total
```

It prints one line per problem and exits non-zero. Run it before you call the work done.
CI runs both halves: the structural check in the `docs` job, the count in `rust`.

What it enforces, and why each one exists because it drifted once:

- **links and anchors** resolve, using the renderer's slug rules (punctuation dropped,
  one hyphen per space), so a hand-written `#anchor` cannot quietly point nowhere.
- **every docs page is linked** from another page, so a new page cannot be orphaned.
- **the ADR range** the README quotes equals what `decisions.md` holds (it said
  ADR-001…010 while the file held fourteen).
- **the crate table** in `architecture.md` equals `Cargo.toml`'s members (it listed a
  `crates/eval` that was never created).
- **the dependency tables** equal `Cargo.lock` and the direct dependencies (the docs
  credited `rubato`, `ort`, `reqwest`, `rtrb` and six more that were not in the tree).
- **every module file** is named in `architecture.md` (a reader could not find
  `live.rs`, `sink.rs` or `frontier.rs` from the map).
- **every doc has a dated `Status:` line**, so nobody has to guess its age.

## When the gate is wrong

The gate is code, and it will be wrong sometimes: a heading it slugs differently, a span
it reads as a crate name. Fix the checker in the same commit as the doc it misreads, and
add the case to `--self-test`. Do not edit the doc to satisfy a bug in the checker, and
do not silence a check you have not understood.

## Numbers have a machine and a date

A measurement without both is a claim, not evidence, and `measurements.md` is where the
repo keeps its evidence. If you cannot re-measure, say the number is unverified rather
than copying the old one forward.

## What the build must show

A change is done when the build and tests a reviewer would run have actually run:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
python3 tools/check_docs.py --tests
```

Paste the result, do not summarise it as "passes". If a check was skipped, say so.

## Green tests are not "it works"

The suite proves the units agree with each other; it does not prove the feature does the
thing a person bought it for. So a change is also done when **the real path has been run
on real input** and the output was inspected, not inferred:

- Run the binary, not only the tests. `flacify play` on an actual file, `flacify setup`
  on a clean home, the command the bug report used.
- Use the input that prompted the change. A synthetic sine is not a substitute for the
  file that failed: the track that broke FLAC writing here was ordinary music whose
  samples sat at both rails, and every synthetic signal round-tripped perfectly.
- Check the artifact, not the exit code. A segment that decodes to the wrong length, a
  window that renders once, a download that plays nothing — all of these exit 0.
- Reproduce the bug before fixing it, and reproduce it again after. "I changed the
  message" is not a fix, and "it compiles" is not evidence.

A release ships this evidence, so a fix is not ready on the strength of a passing test
run alone. What was run by hand, on what input, with what result, belongs in the commit
body and in `measurements.md` when it is a number.
