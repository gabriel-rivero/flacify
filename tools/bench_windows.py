#!/usr/bin/env python3
"""Render the S1 window presets through the resident worker, and time one call at each.

S1 asks two questions of the model: what a window of 1, 2, 4 or 8 granules costs per
second of audio, and whether it still sounds acceptable. This script answers the first
and produces the files a human needs for the second: one WAV per track and preset, plus
the dry source, all the same 30 s excerpt at 48 kHz stereo.

The window is a model input, so the noise lives on the VAE's latent grid, whose size the
HELLO handshake only reports for the trained 8-granule window. `latent_noise_len` derives
it for any window and is checked against HELLO before a single call is made.

    tools/bench_windows.py --presets 8 --iterations 1        # smoke
    tools/bench_windows.py                                   # the full matrix

The app's venv has the modules this needs (numpy, scipy, soundfile); a bare `python3`
does not necessarily.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from worker_bench import CMD_WINDOW, Client  # noqa: E402

SAMPLE_RATE = 48_000
GRANULE = 30_720
# The VAE's latent grid: one frame per 3 840 audio samples (80 ms), 16 channels and 32
# mel bins, so a stereo window needs channels * 16 * 32 * frames / 3840 samples of noise.
LATENT_HOP = 3_840
LATENT_CHANNELS = 16
LATENT_BINS = 32

# The app's own overlap, and the same clamp the pipeline applies.
OVERLAP = 48_000

# Numeric fields a failed row keeps at zero, so the table and the report line up.
FAILED_ROW_ZEROS = (
    "compute_p50",
    "compute_p95",
    "round_trip_p50",
    "round_trip_p95",
    "rtf_p50",
    "rtf_p95",
    "payload_bytes",
    "peak_rss_mb",
    "wav_bytes",
)

DATA_DIR = Path.home() / "Library" / "Application Support" / "flacify"
DEFAULT_TRACKS = [
    Path.home()
    / "Downloads"
    / "Weight of the World Kowaretasekainouta - Marina Kawano (NieR Automata"
    " Original Soundtrack)【Audio】.mp3",
    Path.home() / "Downloads" / "PROI PROI - Honkai Star Rail feat. Alexia Evellyn.mp3",
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tracks", nargs="+", default=[str(p) for p in DEFAULT_TRACKS])
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--presets", default="1,2,4,8", help="granules per window, comma separated")
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--out", default=str(TOOLS / "out" / "s1"))
    ap.add_argument("--python", default=str(DATA_DIR / "venv" / "bin" / "python"))
    ap.add_argument("--model-dir", default=str(DATA_DIR / "models"))
    ap.add_argument("--flashsr-dir", default=str(DATA_DIR / "runtime" / "flashsr"))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = ap.parse_args(argv)
    args.preset_list = [int(part) for part in args.presets.split(",") if part.strip()]
    if not args.preset_list or any(preset < 1 for preset in args.preset_list):
        raise SystemExit("--presets needs positive granule counts, e.g. 1,2,4,8")
    if args.seconds <= 0 or args.iterations < 1:
        raise SystemExit("--seconds and --iterations must be positive")
    return args


def latent_noise_len(channels: int, frames: int) -> int:
    """Samples of noise the VAE's latent grid needs for a `channels` x `frames` window."""
    if frames % LATENT_HOP:
        raise ValueError(f"{frames} frames is not a whole number of {LATENT_HOP}-sample latents")
    latents = frames // LATENT_HOP
    return channels * LATENT_CHANNELS * LATENT_BINS * latents


def check_latent_against_hello(hello: dict) -> None:
    """HELLO's count is ground truth for one window; the formula has to agree with it."""
    expected = latent_noise_len(2, hello["window"])
    if expected != hello["noise_len_stereo"]:
        raise SystemExit(
            f"latent formula says {expected} samples for a {hello['window']}-frame stereo "
            f"window, the worker says {hello['noise_len_stereo']}"
        )


def decode_excerpt(path: Path, seconds: float) -> "object":
    """First `seconds` of `path` as planar float32 stereo at `SAMPLE_RATE`."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    info = sf.info(str(path))
    wanted = int(round(seconds * info.samplerate))
    audio, rate = sf.read(str(path), frames=wanted, dtype="float32", always_2d=True)
    if audio.shape[1] != 2:
        raise SystemExit(f"{path} has {audio.shape[1]} channels; S1 needs stereo")
    planar = np.ascontiguousarray(audio.T)
    if rate != SAMPLE_RATE:
        planar = np.ascontiguousarray(resample_poly(planar, SAMPLE_RATE, rate, axis=1))
    target = int(round(seconds * SAMPLE_RATE))
    if planar.shape[1] < target:
        planar = np.pad(planar, ((0, 0), (0, target - planar.shape[1])))
    return np.ascontiguousarray(planar[:, :target].astype(np.float32))


def grain_frames(preset: int) -> int:
    return preset * GRANULE


def overlap_for(window: int) -> int:
    """The app's overlap, clamped so two windows still cover new input."""
    return min(OVERLAP, window // 2 - 1)


def rise(count: int):
    import numpy as np

    t = np.arange(count, dtype=np.float32) / count
    return (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)


def fall(count: int):
    import numpy as np

    return (1.0 - rise(count)).astype(np.float32)


def window_chunk(audio, start: int, window: int):
    """`window` frames from `start`, zero-padded the way the pipeline pads a short tail."""
    import numpy as np

    chunk = np.zeros((2, window), dtype=np.float32)
    take = min(window, audio.shape[1] - start)
    if take > 0:
        chunk[:, :take] = audio[:, start : start + take]
    return chunk


class Resident:
    """One worker, kept alive across every preset, restarted if it dies."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.ids = itertools.count(1)
        self.client = None
        self.hello = None
        self.load_secs: list[float] = []
        self.start()

    def start(self) -> None:
        namespace = argparse.Namespace(
            python=self.args.python,
            model_dir=self.args.model_dir,
            flashsr_dir=self.args.flashsr_dir,
            device=self.args.device,
        )
        self.client = Client(namespace)
        self.hello = self.client.hello()
        self.load_secs.append(float(self.hello["load_secs"]))
        check_latent_against_hello(self.hello)

    def is_alive(self) -> bool:
        return self.client is not None and self.client.proc.poll() is None

    def ensure(self) -> None:
        if not self.is_alive():
            print("worker exited; starting a fresh one", file=sys.stderr)
            self.start()

    def run(self, chunk, noise):
        """One WINDOW call. Returns the generated audio and the timing of the exchange."""
        import numpy as np

        frames = chunk.shape[1]
        rid = next(self.ids)
        body = np.ascontiguousarray(chunk).tobytes() + np.ascontiguousarray(noise).tobytes()
        started = time.perf_counter()
        try:
            _, status, reply_id, channels, reply_frames, payload = self.client.request(
                CMD_WINDOW, rid, 2, frames, noise.size, body
            )
        except SystemExit as error:
            raise RuntimeError(f"worker stopped during window {rid}: {error}") from error
        round_trip = time.perf_counter() - started
        if status:
            raise RuntimeError(f"window {rid} ({frames} frames) failed: {payload.decode()}")
        compute = wait_compute(self.client, reply_id)
        audio = np.frombuffer(payload, dtype="<f4").reshape(channels, reply_frames)
        return {
            "id": reply_id,
            "frames": frames,
            "compute_secs": compute,
            "round_trip_secs": round_trip,
            "payload_bytes": len(payload),
            "audio": np.array(audio, dtype=np.float32, copy=True),
        }

    def rss_mb(self) -> float:
        return self.client.rss_mb()

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None


def wait_compute(client: Client, window_id: int, timeout: float = 30.0) -> float:
    """The worker logs its compute time after replying, so the line lands after the call."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if window_id in client.compute:
            return client.compute[window_id]
        time.sleep(0.002)
    raise RuntimeError(f"no compute line for window {window_id} within {timeout:.0f} s")


def noise_for(count: int, seed: int):
    import numpy as np

    return np.random.default_rng(seed).standard_normal(count, dtype=np.float32)


def measure(worker: Resident, audio, window: int, iterations: int) -> dict:
    """Warm up once, then time `iterations` calls on the same first window."""
    noise_len = latent_noise_len(2, window)
    chunk = window_chunk(audio, 0, window)
    warm = worker.run(chunk, noise_for(noise_len, seed=0))
    calls = []
    peak_rss = 0.0
    for index in range(iterations):
        result = worker.run(chunk, noise_for(noise_len, seed=index + 1))
        calls.append(result)
        peak_rss = max(peak_rss, worker.rss_mb())
    timings = describe_calls(calls, window, peak_rss)
    timings["warmup_compute_secs"] = round(warm["compute_secs"], 3)
    return timings


def describe_calls(calls: list[dict], window: int, peak_rss: float) -> dict:
    audio_secs = window / SAMPLE_RATE
    compute = [call["compute_secs"] for call in calls]
    round_trip = [call["round_trip_secs"] for call in calls]
    rtf = [value / audio_secs for value in compute]
    return {
        "calls": len(calls),
        "compute_p50": round(percentile(compute, 0.50), 3),
        "compute_p95": round(percentile(compute, 0.95), 3),
        "round_trip_p50": round(percentile(round_trip, 0.50), 3),
        "round_trip_p95": round(percentile(round_trip, 0.95), 3),
        "rtf_p50": round(percentile(rtf, 0.50), 3),
        "rtf_p95": round(percentile(rtf, 0.95), 3),
        "payload_bytes": calls[0]["payload_bytes"],
        "peak_rss_mb": peak_rss,
    }


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def render(worker: Resident, audio, window: int) -> "object":
    """Overlap-add the excerpt through the model, with the app's crossfade and step."""
    import numpy as np

    overlap = overlap_for(window)
    step = window - overlap
    total = audio.shape[1]
    noise_len = latent_noise_len(2, window)
    out = np.zeros_like(audio)
    weight = np.zeros_like(audio)
    head = rise(overlap)
    tail = fall(overlap)
    index = 0
    start = 0
    while start < total:
        chunk = window_chunk(audio, start, window)
        result = worker.run(chunk, noise_for(noise_len, seed=10_000 + index))
        gains = np.ones(window, dtype=np.float32)
        if index > 0:
            gains[:overlap] *= head
        if start + window < total:
            gains[window - overlap :] *= tail
        span = min(window, total - start)
        out[:, start : start + span] += result["audio"][:, :span] * gains[:span]
        weight[:, start : start + span] += gains[:span]
        index += 1
        start += step
    covered = weight > 0
    out[covered] /= weight[covered]
    return out


def write_wav(path: Path, planar) -> int:
    import soundfile as sf

    sf.write(str(path), planar.T, SAMPLE_RATE, subtype="FLOAT")
    return path.stat().st_size


def track_slug(path: Path) -> str:
    name = path.name.lower()
    if "nier" in name or "weight of the world" in name:
        return "nier"
    if "proi" in name:
        return "proi"
    return "".join(char if char.isalnum() else "-" for char in path.stem.lower())[:24].strip("-")


def table_row(label: str, preset: int, row: dict) -> str:
    window = grain_frames(preset)
    return (
        f"{label:6s} {preset:2d}g {window:8d} {window / SAMPLE_RATE:5.2f}s  "
        f"{row['compute_p50']:6.2f} {row['compute_p95']:6.2f}  "
        f"{row['rtf_p50']:5.2f} {row['rtf_p95']:5.2f}  "
        f"{row['payload_bytes']:8d} {row['wav_bytes']:9d} {row['peak_rss_mb']:7.1f}"
    )


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "date": time.strftime("%Y-%m-%d"),
        "machine": platform.platform(),
        "processor": platform.processor(),
        "seconds": args.seconds,
        "presets": args.preset_list,
        "iterations": args.iterations,
        "tracks": [str(path) for path in args.tracks],
        "rows": [],
        "listening_files": {},
        "worker": {},
    }

    worker = Resident(args)
    hello = worker.hello
    report["worker"] = {
        "device": hello["device"],
        "torch": hello["torch"],
        "sample_rate": hello["sample_rate"],
        "load_secs": hello["load_secs"],
        "warmup_secs": hello["warmup_secs"],
        "latent_shape_stereo": hello["latent_shape_stereo"],
    }
    print(
        f"worker    {hello['device']} torch {hello['torch']}  load {hello['load_secs']:.1f} s  "
        f"warmup {hello['warmup_secs']:.2f} s  latent {hello['latent_shape_stereo']}"
    )

    print(
        f"{'track':6s} {'win':>3s} {'frames':>8s} {'secs':>6s}  "
        f"{'cmp50':>6s} {'cmp95':>6s}  {'rtf50':>5s} {'rtf95':>5s}  "
        f"{'payload':>8s} {'wav':>9s} {'rssMB':>7s}"
    )

    for path in (Path(item) for item in args.tracks):
        if not path.is_file():
            raise SystemExit(f"no such track: {path}")
        label = track_slug(path)
        audio = decode_excerpt(path, args.seconds)
        dry_path = out_dir / f"{label}-dry.wav"
        dry_bytes = write_wav(dry_path, audio)
        report["listening_files"].setdefault(label, {})["dry"] = {
            "path": dry_path.name,
            "bytes": dry_bytes,
        }
        print(f"excerpt   {label} {audio.shape[1]} frames from {path.name}")

        for preset in args.preset_list:
            window = grain_frames(preset)
            row = {
                "track": label,
                "source": path.name,
                "preset": preset,
                "frames": window,
                "seconds": round(window / SAMPLE_RATE, 2),
                "listening_verdict": "pending",
            }
            try:
                worker.ensure()
                row.update(measure(worker, audio, window, args.iterations))
                wet = render(worker, audio, window)
                wet_path = out_dir / f"{label}-{preset}g.wav"
                row["wav_bytes"] = write_wav(wet_path, wet)
                row["status"] = "ok"
                report["listening_files"][label][f"{preset}g"] = {
                    "path": wet_path.name,
                    "bytes": row["wav_bytes"],
                }
            except (RuntimeError, ValueError, OSError) as error:
                row["status"] = "failed"
                row["error"] = str(error)
                for key in FAILED_ROW_ZEROS:
                    row.setdefault(key, 0)
            report["rows"].append(row)
            print(table_row(label, preset, row) + ("" if row["status"] == "ok" else "  FAILED"))
            if row["status"] != "ok":
                print(f"          {row['error']}", file=sys.stderr)

    worker.close()
    report["worker"]["loads_secs"] = worker.load_secs
    report["worker"]["peak_rss_mb"] = max((row["peak_rss_mb"] for row in report["rows"]), default=0.0)

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report    {report_path}")
    print(f"listening {out_dir}")
    failures = [row for row in report["rows"] if row["status"] != "ok"]
    if failures:
        print(f"failed    {len(failures)} of {len(report['rows'])} rows", file=sys.stderr)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
