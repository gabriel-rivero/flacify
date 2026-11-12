#!/usr/bin/env python3
"""Time the exported graph on this machine: the number ADR-002 lives or dies on.

The build criterion is sustained RTF <= 1.5 for a 5.12 s stereo window, and the
earlier numbers come from an RTX 4060 Ti, which says nothing about a laptop. This
script needs no torch and no FlashSR package: it loads the .onnx file, reads the
shapes it declares, and runs it.

    python3 tools/bench_onnx.py                              # coreml, if available
    python3 tools/bench_onnx.py --provider cpu --iterations 10

RTF here is seconds of wall clock per second of audio, so lower is better and
anything above 1.0 cannot keep up with playback on its own. A first iteration is
discarded as warmup (weight paging, kernel selection); the rest are reported with
median, p95 and the median of the back half, which is the sustained figure after
the GPU has had time to throttle.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SAMPLE_RATE = 48_000
WINDOW = 245_760
WINDOW_SECS = WINDOW / SAMPLE_RATE


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--onnx", default=str(TOOLS / "out" / "flashsr.onnx"))
    ap.add_argument("--provider", default="coreml", choices=["coreml", "cpu", "auto"])
    ap.add_argument("--iterations", type=int, default=5, help="timed runs, after one warmup")
    ap.add_argument(
        "--coreml-mlprogram",
        action="store_true",
        help="ask CoreML for an MLProgram instead of the default NeuralNetwork; "
        "MLProgram is the format that can use the GPU and ANE on current macOS",
    )
    ap.add_argument(
        "--coreml-units",
        default="ALL",
        choices=["CPU", "GPU", "CPUAndGPU", "ALL"],
        help="which CoreML compute units are allowed",
    )
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--report", default=str(TOOLS / "out" / "bench_report.json"))
    ap.add_argument("--gate-rtf", type=float, default=1.5, help="ADR-002 kill criterion")
    return ap.parse_args(argv)


def providers_for(args: argparse.Namespace) -> list[str] | list[tuple[str, dict]]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if args.provider == "cpu":
        return ["CPUExecutionProvider"]
    if args.provider == "coreml" and "CoreMLExecutionProvider" not in available:
        raise SystemExit(
            f"CoreMLExecutionProvider is not available here: {available}\n"
            "  onnxruntime on macOS ships it; check the wheel"
        )

    options: dict[str, str] = {"MLComputeUnits": args.coreml_units}
    if args.coreml_mlprogram:
        options["ModelFormat"] = "MLProgram"
    return [("CoreMLExecutionProvider", options), "CPUExecutionProvider"]


def inputs_for(session, seed: int):
    """Zero audio and seeded noise, shaped from what the graph declares."""
    import numpy as np

    shapes, dtypes = {}, {}
    for entry in session.get_inputs():
        shapes[entry.name] = entry.shape
        dtypes[entry.name] = entry.type
    if "cond" not in shapes or "noise" not in shapes:
        raise SystemExit(f"graph inputs are {list(shapes)}, expected cond and noise")

    cond = np.zeros(shapes["cond"], dtype=np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(shapes["noise"], dtype=np.float32)
    return {"cond": cond, "noise": noise}, shapes


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    import onnxruntime as ort

    providers = providers_for(args)
    options = ort.SessionOptions()
    options.log_severity_level = 3

    started = time.perf_counter()
    session = ort.InferenceSession(args.onnx, sess_options=options, providers=providers)
    load_secs = time.perf_counter() - started

    feeds, shapes = inputs_for(session, args.seed)

    timings: list[float] = []
    for run in range(args.iterations + 1):
        t0 = time.perf_counter()
        session.run(None, feeds)
        elapsed = time.perf_counter() - t0
        if run == 0:
            print(f"warmup    {elapsed:.2f} s (discarded)")
        else:
            timings.append(elapsed)
            print(f"run {run:2d}    {elapsed:.2f} s   RTF {elapsed / WINDOW_SECS:.3f}")

    median = statistics.median(timings)
    ordered = sorted(timings)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    back_half = statistics.median(timings[len(timings) // 2 :])

    report = {
        "onnx": str(Path(args.onnx).resolve()),
        "providers_requested": providers,
        "providers_active": session.get_providers(),
        "load_secs": round(load_secs, 2),
        "input_shapes": shapes,
        "window_secs": WINDOW_SECS,
        "runs_secs": [round(value, 3) for value in timings],
        "median_secs": round(median, 3),
        "p95_secs": round(p95, 3),
        "sustained_secs": round(back_half, 3),
        "median_rtf": round(median / WINDOW_SECS, 3),
        "p95_rtf": round(p95 / WINDOW_SECS, 3),
        "sustained_rtf": round(back_half / WINDOW_SECS, 3),
        "gate_rtf": args.gate_rtf,
        "passed": round(back_half / WINDOW_SECS, 3) <= args.gate_rtf,
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")

    print(f"provider  {', '.join(report['providers_active'])}")
    print(f"load      {load_secs:.2f} s")
    print(
        f"steady    {median:.2f} s/window, RTF {report['median_rtf']:.3f} "
        f"(gate {args.gate_rtf})"
    )
    print(f"sustained RTF {report['sustained_rtf']:.3f}   report {args.report}")
    if not report["passed"]:
        print(f"FAILED the sustained gate: RTF {report['sustained_rtf']} > {args.gate_rtf}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
