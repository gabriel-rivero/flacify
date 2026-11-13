#!/usr/bin/env python3
"""Drive tools/worker.py exactly the way the Rust client will, and time it.

This exists to answer one question before the player is built: does the 2.59 s per
window that eager PyTorch/Metal measured survive a real process boundary, a resident
model, and repeated calls? It speaks the same binary protocol, over the same pipes, so
the number it reports is the number the app will get.

It also separates compute from plumbing. The worker logs its own compute time per
window on stderr; the round trip measured here includes pipe transfer, so the difference
is the IPC cost of a 2 MB request and a 2 MB reply.

    python3 tools/worker_bench.py --iterations 3
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
HEADER = struct.Struct("<4sB3xQIII")
REPLY = struct.Struct("<4sBB2xQIII")
REQUEST_MAGIC = b"FLW1"
REPLY_MAGIC = b"FLR1"
CMD_HELLO = 1
CMD_WINDOW = 2
CMD_SHUTDOWN = 3
WINDOW_SECS = 245_760 / 48_000

COMPUTE_RE = re.compile(r"window (\d+): ([\d.]+) s")


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", default=str(Path.home() / "models" / "flashsr"))
    ap.add_argument("--flashsr-dir", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--report", default=str(TOOLS / "out" / "worker_bench_report.json"))
    return ap.parse_args(argv)


def read_exact(stream, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise SystemExit("worker closed the pipe")
        chunks.extend(chunk)
    return bytes(chunks)


def windows_rss_mb(pid: int) -> float:
    """RSS of `pid` and its children, because Windows has no `ps` and the venv's
    `python.exe` is a launcher that hands the work to a child interpreter."""
    try:
        import psutil
    except ImportError:
        return 0.0
    try:
        process = psutil.Process(pid)
        tree = [process, *process.children(recursive=True)]
        return round(sum(p.memory_info().rss for p in tree) / 2**20, 1)
    except (psutil.Error, OSError):
        return 0.0


class Client:
    def __init__(self, args: argparse.Namespace) -> None:
        command = [
            args.python,
            str(TOOLS / "worker.py"),
            "--model-dir",
            args.model_dir,
            "--device",
            args.device,
        ]
        if args.flashsr_dir:
            command += ["--flashsr-dir", args.flashsr_dir]

        self.proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self.compute: dict[int, float] = {}
        self.log: list[str] = []
        self._reader = threading.Thread(target=self._drain_stderr, daemon=True)
        self._reader.start()

    def _drain_stderr(self) -> None:
        assert self.proc.stderr is not None
        for raw in self.proc.stderr:
            line = raw.decode(errors="replace").rstrip()
            self.log.append(line)
            match = COMPUTE_RE.search(line)
            if match:
                self.compute[int(match.group(1))] = float(match.group(2))

    def request(self, cmd: int, window_id: int, channels: int, frames: int, noise_len: int, body: bytes = b""):
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(HEADER.pack(REQUEST_MAGIC, cmd, window_id, channels, frames, noise_len))
        if body:
            self.proc.stdin.write(body)
        self.proc.stdin.flush()
        header = read_exact(self.proc.stdout, REPLY.size)
        magic, reply_cmd, status, rid, rchannels, rframes, length = REPLY.unpack(header)
        if magic != REPLY_MAGIC:
            raise SystemExit(f"bad reply magic {magic!r}")
        payload = read_exact(self.proc.stdout, length) if length else b""
        return reply_cmd, status, rid, rchannels, rframes, payload

    def hello(self) -> dict:
        _, status, _, _, _, payload = self.request(CMD_HELLO, 0, 0, 0, 0)
        if status:
            raise SystemExit(f"hello failed: {payload.decode()}")
        return json.loads(payload)

    def rss_mb(self) -> float:
        if sys.platform == "win32":
            return windows_rss_mb(self.proc.pid)
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(self.proc.pid)], capture_output=True, text=True
        ).stdout.strip()
        return round(int(out) / 1024, 1) if out else 0.0

    def close(self) -> None:
        try:
            self.request(CMD_SHUTDOWN, 0, 0, 0, 0)
        except SystemExit:
            pass
        self.proc.wait(timeout=30)
        self._reader.join(timeout=10)


def fixture_windows(limit: int):
    import numpy as np

    paths = sorted((TOOLS / "fixtures" / "out").glob("*.in.f32"))[:limit]
    if not paths:
        raise SystemExit("no fixtures; run tools/fixtures/gen.py first")
    return [(path, np.fromfile(path, dtype="<f4").reshape(2, -1)) for path in paths]


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    import numpy as np

    t_spawn = time.perf_counter()
    client = Client(args)
    hello = client.hello()
    ready_secs = time.perf_counter() - t_spawn
    noise_len = hello["noise_len_stereo"]
    print(f"hello     {json.dumps(hello)}")

    rng = np.random.default_rng(1234)
    noise = rng.standard_normal(noise_len, dtype=np.float32)
    frames = hello["window"]

    first_audio = None
    rows = []
    window_id = 0
    for path, audio in fixture_windows(4):
        for _ in range(args.iterations):
            window_id += 1
            body = np.ascontiguousarray(audio[:, :frames]).tobytes() + noise.tobytes()
            t0 = time.perf_counter()
            cmd, status, rid, channels, rframes, payload = client.request(
                CMD_WINDOW, window_id, 2, frames, noise_len, body
            )
            round_trip = time.perf_counter() - t0
            if status:
                raise SystemExit(f"window failed: {payload.decode()}")
            if first_audio is None:
                first_audio = time.perf_counter() - t_spawn
            rows.append(
                {
                    "track": path.name,
                    "id": rid,
                    "round_trip_secs": round(round_trip, 3),
                    "bytes_out": len(payload),
                }
            )

    rss = client.rss_mb()
    client.close()

    # The worker's own compute times arrive on stderr, asynchronously; read them only
    # once the process has exited and the reader thread has drained.
    for row in rows:
        row["compute_secs"] = round(client.compute.get(row["id"], 0.0), 3)

    for row in rows:
        overhead = row["round_trip_secs"] - row["compute_secs"]
        row["ipc_overhead_secs"] = round(overhead, 3)
        row["rtf"] = round(row["round_trip_secs"] / WINDOW_SECS, 3)

    timed = rows[1:] if len(rows) > 1 else rows
    median = statistics.median(row["round_trip_secs"] for row in timed)
    compute = statistics.median(row["compute_secs"] for row in timed)
    p95 = sorted(row["round_trip_secs"] for row in timed)[-1]

    report = {
        "hello": hello,
        "ready_secs": round(ready_secs, 2),
        "time_to_first_audio_secs": round(first_audio or 0.0, 2),
        "window_secs": WINDOW_SECS,
        "rows": rows,
        "median_round_trip_secs": round(median, 3),
        "median_compute_secs": round(compute, 3),
        "median_ipc_overhead_secs": round(median - compute, 3),
        "median_rtf": round(median / WINDOW_SECS, 3),
        "worst_round_trip_secs": round(p95, 3),
        "worker_rss_mb": rss,
        "gate_rtf": 1.5,
        "passed": round(median / WINDOW_SECS, 3) <= 1.5,
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")

    for row in rows:
        print(
            f"{row['track']:22s} id {row['id']:2d}  round trip {row['round_trip_secs']:6.2f} s  "
            f"compute {row['compute_secs']:6.2f} s  ipc {row['ipc_overhead_secs']:5.3f} s  "
            f"RTF {row['rtf']:5.2f}"
        )
    print(f"ready     {ready_secs:.1f} s to first hello, {first_audio:.1f} s to first audio")
    print(f"steady    {median:.2f} s/window (compute {compute:.2f} s, ipc {median - compute:.3f} s)")
    print(f"rtf       {report['median_rtf']:.3f} (gate {report['gate_rtf']}), worker RSS {rss} MB")
    print(f"report    {args.report}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
