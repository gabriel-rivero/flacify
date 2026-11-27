#!/usr/bin/env python3
"""Check that the documentation still describes this repository.

The docs are the only description of the code that a reader gets before the code
itself, and nothing else in the pipeline notices when they stop being true. This
script is the notice. It reads the repository, never the internet, and prints one
line per problem; an empty list is a pass.

    python3 tools/check_docs.py            # structural checks, no build
    python3 tools/check_docs.py --tests    # also compare the counted test total
    python3 tools/check_docs.py --self-test

The checks, and what each one catches:

  1. links        a relative link or `#anchor` that no longer resolves
  2. index        a docs/ page nothing links to, or an index row naming a missing page
  3. adr          the ADR range quoted in README disagrees with decisions.md
  4. crates       the crate table in architecture.md disagrees with the workspace
  5. deps         a dependency documented as present but absent, or the reverse
  6. modules      a module in crates/ that architecture.md never names
  7. status       a doc with no dated `Status:` line, so nobody can tell its age

`--tests` needs a built workspace, so it is off by default: `cargo test -- --list`
compiles. CI runs it where the tests already ran.
"""
import json
import os
import re
import subprocess
import sys
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TOKEN_RE = re.compile(r"`([^`]+)`")

# Docs whose `Status:` line is deliberately absent: measurements are appended by date,
# and a release checklist is a procedure rather than a description.
NO_STATUS = {"measurements.md", "release-1.0.md"}


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def relative_markdown():
    """Every markdown file the docs promise to keep true, with its text."""
    paths = [os.path.join(ROOT, "README.md"), os.path.join(ROOT, "AGENTS.md")]
    paths += sorted(os.path.join(DOCS, n) for n in os.listdir(DOCS) if n.endswith(".md"))
    paths += sorted(os.path.join(ROOT, ".github", n) for n in ("PULL_REQUEST_TEMPLATE.md",)
                    if os.path.exists(os.path.join(ROOT, ".github", n)))
    return {p: read(p) for p in paths if os.path.exists(p)}


def slug(heading):
    """The anchor GitHub generates: drop punctuation, then one hyphen per space.

    A run of spaces becomes a run of hyphens, which is why an em dash between two
    spaces gives `v04--streaming`, not `v04-streaming`. Anchors in this repo are
    written by hand, so the check has to match the renderer the links are read in.
    """
    text = heading.strip().lower()
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"[^\w\s-]", "", text)
    return text.replace(" ", "-")


def anchors(text):
    found = set()
    for line in text.splitlines():
        heading = HEADING_RE.match(line)
        if heading:
            found.add(slug(heading.group(2)))
    return found


def check_links(files):
    problems = []
    for path, text in files.items():
        for target in LINK_RE.findall(text):
            if re.match(r"^(https?:|mailto:|#)", target):
                continue
            file_part, _, anchor = target.partition("#")
            resolved = os.path.normpath(os.path.join(os.path.dirname(path), file_part)) if file_part else path
            if not os.path.exists(resolved):
                problems.append(f"{os.path.relpath(path, ROOT)}: link to missing {target}")
            elif anchor and resolved.endswith(".md"):
                if anchor not in anchors(read(resolved)):
                    problems.append(f"{os.path.relpath(path, ROOT)}: link to missing anchor {target}")
    return problems


def check_index(files):
    """Every docs page is linked from somewhere, and every docs link names a real page."""
    problems = []
    readme = files[os.path.join(ROOT, "README.md")]
    linked = set()
    for path, text in files.items():
        for target in LINK_RE.findall(text):
            if re.match(r"^(https?:|mailto:|#)", target):
                continue
            file_part = target.partition("#")[0]
            if not file_part:
                continue
            resolved = os.path.normpath(os.path.join(os.path.dirname(path), file_part))
            linked.add(os.path.realpath(resolved))
    for name in sorted(os.listdir(DOCS)):
        if not name.endswith(".md"):
            continue
        page = os.path.realpath(os.path.join(DOCS, name))
        if page not in linked:
            problems.append(f"docs/{name}: no other page links it, so nobody will find it")
    if "## Documents" not in readme:
        problems.append("README.md: the Documents index is gone")
    return problems


def check_adr(files):
    """The README quotes an ADR range; it must be the range decisions.md actually holds."""
    problems = []
    decisions = files.get(os.path.join(DOCS, "decisions.md"), "")
    numbers = sorted(int(n) for n in re.findall(r"^##\s+ADR-(\d{3})", decisions, re.M))
    if not numbers:
        return ["docs/decisions.md: no `## ADR-NNN` headings found"]
    readme = files.get(os.path.join(ROOT, "README.md"), "")
    quoted = re.search(r"ADR-(\d{3})\s*[…\.]{1,3}\s*ADR-(\d{3})", readme)
    if not quoted:
        problems.append("README.md: the Decisions row no longer states the ADR range")
    else:
        low, high = int(quoted.group(1)), int(quoted.group(2))
        if (low, high) != (numbers[0], numbers[-1]):
            problems.append(
                f"README.md: quotes ADR-{low:03d} … ADR-{high:03d}, decisions.md holds "
                f"ADR-{numbers[0]:03d} … ADR-{numbers[-1]:03d}")
    gaps = [n for n in range(numbers[0], numbers[-1] + 1) if n not in numbers]
    for gap in gaps:
        # A gap is acceptable only when the page records it, so a later silent skip
        # cannot pass by leaving nothing behind at all.
        if not re.search(rf"ADR-{gap:03d} is unresolved|ADR-{gap:03d} does not exist", decisions):
            problems.append(
                f"docs/decisions.md: ADR-{gap:03d} is missing and the page does not say why")
    return problems


def workspace():
    root = tomllib.load(open(os.path.join(ROOT, "Cargo.toml"), "rb"))
    members = {os.path.relpath(os.path.normpath(os.path.join(ROOT, m)), ROOT): m
               for m in root["workspace"]["members"]}
    internal = {name for name in root["workspace"]["dependencies"] if name.startswith("flacify-")}
    return root, members, internal


def crate_deps(member_dir, internal):
    """External package names one crate actually depends on."""
    manifest = os.path.join(ROOT, member_dir, "Cargo.toml")
    data = tomllib.load(open(manifest, "rb"))
    names = set()
    for section in ("dependencies", "build-dependencies"):
        for name, spec in data.get(section, {}).items():
            if name in internal:
                continue
            names.add(spec.get("package", name) if isinstance(spec, dict) else name)
    return names


def table_rows(text, heading):
    """Rows of the first markdown table after `heading`, as cell lists."""
    if heading not in text:
        return []
    block = text.split(heading, 1)[1]
    rows, seen = [], False
    for line in block.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            rows.append(cells)
            seen = True
        elif seen:
            break
        elif line.startswith("#"):
            continue
    return rows[1:] if rows else []


def crate_tokens(cells):
    """The crate names a cell claims.

    Each name is its own code span. Paths and filenames in a span (`dependencies.md`,
    `flacify-core`) are prose, not crates, so they are dropped.
    """
    return {t.strip() for t in TOKEN_RE.findall(cells)
            if re.fullmatch(r"[a-z][a-z0-9_-]*", t.strip() or "")}


def check_crates(files):
    """architecture.md's crate table is the map a reader uses; it must be the real map."""
    problems = []
    text = files.get(os.path.join(DOCS, "architecture.md"), "")
    rows = table_rows(text, "## Crates")
    if not rows:
        return ["docs/architecture.md: the crate table is gone"]
    _, members, internal = workspace()
    documented = {}
    for cells in rows:
        token = TOKEN_RE.search(cells[0])
        if not token or not token.group(1).startswith("crates/"):
            continue
        documented[token.group(1)] = cells
    for missing in sorted(set(members) - set(documented)):
        problems.append(f"docs/architecture.md: crate {missing} is in the workspace but not the table")
    for extra in sorted(set(documented) - set(members)):
        problems.append(f"docs/architecture.md: crate {extra} is in the table but not the workspace")
    for member, cells in documented.items():
        if member not in members or len(cells) < 3:
            continue
        # Only the cell's own code spans name crates; a link inside it is prose.
        cell = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", cells[2])
        claimed = {t for t in crate_tokens(cell) if t not in internal}
        real = crate_deps(members[member], internal)
        for wrong in sorted(claimed - real):
            problems.append(f"docs/architecture.md: {member} does not depend on {wrong}")
    return problems


def check_deps(files):
    """dependencies.md must not credit a crate the build does not use, nor omit one it does.

    Two tables: "In the tree" claims presence, "Considered, not used" claims absence. The
    split is what lets a reader tell a decision from an oversight, so both are checked
    against `Cargo.lock`.
    """
    problems = []
    text = files.get(os.path.join(DOCS, "dependencies.md"), "")
    in_tree = table_rows(text, "### In the tree")
    not_used = table_rows(text, "### Considered, not used")
    if not in_tree:
        return ["docs/dependencies.md: the `### In the tree` crate table is gone"]
    lock = read(os.path.join(ROOT, "Cargo.lock"))
    locked = set(re.findall(r'^name = "([^"]+)"$', lock, re.M))
    _, members, internal = workspace()
    direct = set()
    for member in members:
        direct |= crate_deps(member, internal)

    present = set()
    for cells in in_tree:
        for name in crate_tokens(cells[0]):
            present.add(name)
            if name not in locked:
                problems.append(f"docs/dependencies.md: {name} is listed as used but not in Cargo.lock")
    for cells in not_used:
        for name in crate_tokens(cells[0]):
            if name in locked:
                problems.append(
                    f"docs/dependencies.md: {name} is under 'not used' but is in Cargo.lock")

    documented = {name.replace("-", "_") for name in present} | present
    for dep in sorted(direct - documented):
        problems.append(f"docs/dependencies.md: direct dependency {dep} is undocumented")
    return problems


def module_names():
    """Every module file the source tree holds, by stem."""
    names = set()
    for crate in ("core", "infer", "app"):
        base = os.path.join(ROOT, "crates", crate, "src")
        for directory, _, files in os.walk(base):
            for name in files:
                if not name.endswith(".rs"):
                    continue
                stem = name[:-3]
                if stem in ("mod", "lib", "main"):
                    continue
                names.add(stem)
    return names


def check_modules(files):
    """A module architecture.md never names is a module a reader cannot find."""
    text = files.get(os.path.join(DOCS, "architecture.md"), "")
    problems = []
    for name in sorted(module_names()):
        # A row may name a submodule as `dsp/cheby.rs` or group commands as
        # `setup`, `probe`, …, so match the stem anywhere, not the exact filename.
        if not re.search(rf"\b{re.escape(name)}\.rs\b", text):
            problems.append(f"docs/architecture.md: module {name}.rs is never named")
    return problems


def check_status(files):
    problems = []
    for path, text in files.items():
        if not path.startswith(DOCS + os.sep):
            continue
        name = os.path.basename(path)
        if name in NO_STATUS:
            continue
        head = "\n".join(text.splitlines()[:12])
        if not re.search(r"^Status:.*\d{4}-\d{2}-\d{2}", head, re.M):
            problems.append(f"docs/{name}: no `Status: ..., YYYY-MM-DD` line in the first lines")
    return problems


def counted_tests():
    """The workspace test total, as cargo lists it. Requires a build."""
    out = subprocess.run(["cargo", "test", "--workspace", "--all-targets", "--", "--list"],
                         cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip().splitlines()[-1:] or "cargo test --list failed")
    total = 0
    for line in out.stdout.splitlines():
        match = re.match(r"^(\d+) tests?,", line)
        if match:
            total += int(match.group(1))
    return total


def check_tests(files):
    """README quotes a test count; a stale number is the cheapest drift to catch."""
    readme = files.get(os.path.join(ROOT, "README.md"), "")
    quoted = re.search(r"#\s*(\d+)\s+tests\b", readme)
    if not quoted:
        return ["README.md: the test count line is gone"]
    actual = counted_tests()
    if int(quoted.group(1)) != actual:
        return [f"README.md: says {quoted.group(1)} tests, cargo lists {actual}"]
    return []


CHECKS = [
    ("links", check_links),
    ("index", check_index),
    ("adr", check_adr),
    ("crates", check_crates),
    ("deps", check_deps),
    ("modules", check_modules),
    ("status", check_status),
]


def self_test():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "docs"))
        page = os.path.join(tmp, "docs", "b.md")
        top = os.path.join(tmp, "a.md")
        write = lambda path, text: open(path, "w").write(text)

        write(top, "# A\n\n## Some Heading Here\n")
        write(page, "x [ok](../a.md#some-heading-here)\n")
        assert check_links({top: read(top), page: read(page)}) == []

        write(page, "x [no](../gone.md)\n")
        assert check_links({top: read(top), page: read(page)}), "a missing file must be caught"

        write(page, "x [no](../a.md#missing)\n")
        assert check_links({top: read(top), page: read(page)}), "a missing anchor must be caught"
    assert slug("The sink, end to end (A3), 2026-11-18") == "the-sink-end-to-end-a3-2026-11-18"
    text = "## Crates\n\n| a | b |\n|---|---|\n| `crates/x` | y |\n"
    assert table_rows(text, "## Crates") == [["`crates/x`", "y"]]
    print("self-test ok")


def main():
    if "--self-test" in sys.argv:
        return self_test()
    files = relative_markdown()
    problems = []
    for name, check in CHECKS:
        found = check(files)
        problems += [f"[{name}] {p}" for p in found]
    if "--tests" in sys.argv:
        problems += [f"[tests] {p}" for p in check_tests(files)]
    if problems:
        print("\n".join(problems))
        print(f"\n{len(problems)} documentation problem(s)")
        return 1
    print(f"docs ok ({len(files)} files, {len(CHECKS)} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
