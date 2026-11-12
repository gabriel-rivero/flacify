#!/usr/bin/env python3
"""Compare the exported ONNX graph against PyTorch on every track, and fail loudly.

This is E1-3. For each track it takes one model window, band-limits it the way upstream
does (the same two stages Rust reproduces), then runs:

  * upstream `FlashSR.forward` with the internal noise pinned to a fixed tensor;
  * the ONNX graph with the same tensor.

and reports the difference relative to the reference in dB. The gate is ADR-002's kill
criterion: worse than -40 dB on any track and the ONNX path is not good enough to ship.
The report is JSON so a regression can be diffed, not squinted at.

    python3 tools/compare_onnx.py --onnx tools/out/flashsr.onnx --model-dir ~/models/flashsr

Defaults to the four scipy fixtures. E1-3 asks for five real tracks; point `--tracks` at
them when E8-1 lands. Fewer than five tracks prints a warning, because the sample is
what the gate is worth.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import flashsr_onnx as h  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    h.add_common_args(ap)
    ap.add_argument("--onnx", default=str(TOOLS / "out" / "flashsr.onnx"))
    ap.add_argument(
        "--tracks",
        nargs="+",
        default=[str(TOOLS / "fixtures" / "out" / "*.in.f32")],
        help="audio files or globs; .f32 is planar raw, anything else goes through WAV",
    )
    ap.add_argument("--report", default=str(TOOLS / "out" / "compare_report.json"))
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gate-db", type=float, default=-40.0)
    ap.add_argument("--allow-gate-fail", action="store_true")
    return ap.parse_args(argv)


def expand_tracks(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(p) for p in glob.glob(pattern))
        paths.extend(matches)
    if not paths:
        raise SystemExit(f"no tracks matched {patterns}")
    return paths


def band_limit(window):
    """Upstream's `UtilAudioLowPassFilter.lowpass`, with the near-silence guard.

    The graph starts after this stage, so a comparison that skipped it would feed the
    model audio it never sees. `find_cutoff_freq` reports 24 kHz for a window with no
    usable low band, which `cheby1` rejects at Nyquist, so that case passes through.
    """
    import numpy as np
    import torch
    from FlashSR.Util.UtilAudioSR import UtilAudioSR
    from FlashSR.Util.UtilAudioLowPassFilter import UtilAudioLowPassFilter

    # `find_cutoff_freq` takes a tensor (it runs a torch STFT and reads channel 0);
    # `lowpass` takes numpy. Upstream does the same conversion in `FlashSR.forward`.
    cutoff = int(UtilAudioSR.find_cutoff_freq(torch.from_numpy(window)))
    if cutoff >= h.SAMPLE_RATE // 2 or cutoff < 1:
        return window, cutoff, False
    filtered = UtilAudioLowPassFilter.lowpass(
        window, h.SAMPLE_RATE, filter_name="cheby", filter_order=8, cutoff_freq=cutoff
    )
    return np.ascontiguousarray(filtered, dtype=np.float32), cutoff, True


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    import numpy as np

    torch = h.require_torch()
    import onnxruntime as ort

    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.set_grad_enabled(False)

    model = h.load_model(
        h.resolve_model_dir(args.model_dir),
        h.resolve_package_dir(args.flashsr_dir),
        args.device,
    )
    h.force_deterministic_vae(model)
    shape = h.latent_shape(model, args.channels, h.WINDOW, args.device)
    noise = h.noise_tensor(shape, args.seed, args.device)

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    tracks = expand_tracks(args.tracks)

    rows = []
    for path in tracks:
        window = h.window_from_track(path, args.channels)
        cond_values, cutoff, lowpassed = band_limit(window)
        if cond_values.shape != (args.channels, h.WINDOW):
            raise SystemExit(f"{path}: expected {(args.channels, h.WINDOW)}, got {cond_values.shape}")
        cond = torch.from_numpy(cond_values).to(device=args.device, dtype=torch.float32)

        reference = h.reference_output(model, cond, noise).cpu().numpy()
        onnx_out = session.run(
            None, {"cond": cond_values, "noise": noise.cpu().numpy()}
        )[0]

        level = h.db(onnx_out - reference, reference)
        row = {
            "track": path.name,
            "cutoff_hz": cutoff,
            "lowpassed": lowpassed,
            "max_abs_diff": float(np.max(np.abs(onnx_out - reference))),
            "db": round(float(level), 2),
            "passed": bool(level <= args.gate_db and np.isfinite(onnx_out).all()),
        }
        rows.append(row)
        status = "ok" if row["passed"] else "FAIL"
        print(
            f"{path.name:28s} cutoff {cutoff:5d} Hz  "
            f"{row['db']:+7.2f} dB  max {row['max_abs_diff']:.3e}  {status}"
        )

    # dB values are negative, so the worst track is the largest, not the smallest.
    worst = max(row["db"] for row in rows)
    report = {
        "onnx": str(Path(args.onnx).resolve()),
        "seed": args.seed,
        "gate_db": args.gate_db,
        "latent_shape": list(shape),
        "tracks": rows,
        "tracks_required": 5,
        "worst_db": worst,
        # `passed` is the whole gate, sample size included; the per-track `passed` flags
        # above are the numeric answer. With four fixtures this is False until E8-1.
        "passed": all(row["passed"] for row in rows) and len(rows) >= 5,
    }
    if len(rows) < 5:
        print(
            f"warning: {len(rows)} tracks, E1-3 wants 5. The gate is advisory until "
            "E8-1's corpus lands.",
            file=sys.stderr,
        )

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(f"worst      {worst:+.2f} dB (gate {args.gate_db:+.0f})")
    print(f"report     {args.report}")

    if not all(row["passed"] for row in rows) and not args.allow_gate_fail:
        print("FAILED the numeric gate", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
