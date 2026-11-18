#!/usr/bin/env python3
"""Build a long lossy file from a short playlist, for the acceptance runs.

The 60-minute runs need content that is (a) long enough to fill a cache entry,
(b) band-limited, so the app renders it rather than passing it through, and (c) not
committed, because it is an hour of audio. Concatenating the library files and
re-encoding once to 48 kHz MP3 gives all three, and the repetition does not affect
what the run measures (dropouts, memory, seek, second-play GPU time).

    python3 tools/make_hour.py --minutes 60 --out tools/fixtures/out/hour.mp3 \
        ~/Downloads/one.mp3 ~/Downloads/two.mp3
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sources", nargs="+", type=Path)
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--out", type=Path, default=Path("tools/fixtures/out/hour.mp3"))
    ap.add_argument("--bitrate", default="192k")
    return ap.parse_args(argv)


def duration_secs(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("ffmpeg and ffprobe are required", file=sys.stderr)
        return 2
    for source in args.sources:
        if not source.is_file():
            print(f"missing source: {source}", file=sys.stderr)
            return 2

    target = args.minutes * 60.0
    spans = [(source, duration_secs(source)) for source in args.sources]
    cycle = sum(span for _, span in spans)
    if cycle <= 0:
        print("sources have no duration", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    listing = args.out.with_suffix(".concat.txt")
    lines: list[str] = []
    total = 0.0
    while total < target:
        for source, span in spans:
            if total >= target:
                break
            lines.append(f"file '{source.resolve()}'")
            total += span
    listing.write_text("\n".join(lines) + "\n")

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(listing),
        "-t", f"{target:.3f}",
        "-ar", "48000", "-ac", "2", "-c:a", "libmp3lame", "-b:a", args.bitrate,
        str(args.out),
    ]
    subprocess.run(command, check=True)
    listing.unlink(missing_ok=True)

    size_mb = args.out.stat().st_size / 2**20
    print(f"{args.out}: {duration_secs(args.out) / 60:.1f} min, {size_mb:.1f} MB, 48 kHz stereo MP3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
