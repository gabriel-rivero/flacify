#!/usr/bin/env python3
"""The numeric gate for the sidecar: worker output vs in-process Metal output.

The worker changes where the model runs, not how. If it ever stops being the same
function, this says so in dB before anyone listens to it. Same fixed noise tensor and
same window both sides, so the only difference left is process boundary and device
plumbing.

    python3 tools/worker_check.py            # exit 1 if the difference is above -60 dB
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", default=str(Path.home() / "models" / "flashsr"))
    ap.add_argument("--flashsr-dir", default=None)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gate-db", type=float, default=-60.0)
    ap.add_argument("--report", default=str(TOOLS / "out" / "worker_check_report.json"))
    return ap.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    import numpy as np

    import flashsr_onnx as h
    from worker_bench import CMD_WINDOW, Client, fixture_windows

    path, audio = fixture_windows(1)[0]

    # In-process reference, same code the export compared against.
    torch = h.require_torch()
    model = h.load_model(
        h.resolve_model_dir(args.model_dir),
        h.resolve_package_dir(args.flashsr_dir),
        args.device,
    )
    h.force_deterministic_vae(model)
    h.patch_mel_front_end()
    model.to(args.device)
    module = h.make_sampler_module(model)
    shape = h.latent_shape(model, 2, h.WINDOW, args.device)
    noise_np = np.random.default_rng(args.seed).standard_normal(
        int(np.prod(shape)), dtype=np.float32
    )
    noise = torch.from_numpy(noise_np.reshape(shape)).to(args.device)
    cond = torch.from_numpy(np.ascontiguousarray(audio[:, : h.WINDOW])).to(args.device)
    with torch.inference_mode():
        expected = module(cond, noise).float().cpu().numpy()

    # The same window through the pipe.
    client = Client(args)
    hello = client.hello()
    body = cond.cpu().numpy().tobytes() + noise_np.tobytes()
    _, status, _, _, _, payload = client.request(
        CMD_WINDOW, 1, 2, hello["window"], hello["noise_len_stereo"], body
    )
    client.close()
    if status:
        raise SystemExit(f"worker failed: {payload.decode()}")
    got = np.frombuffer(payload, dtype="<f4").reshape(expected.shape)

    level = h.db(got - expected, expected)
    report = {
        "track": path.name,
        "device": hello["device"],
        "torch": hello["torch"],
        "window": hello["window"],
        "latent_shape": hello["latent_shape_stereo"],
        "db": round(float(level), 2),
        "max_abs_diff": float(np.max(np.abs(got - expected))),
        "gate_db": args.gate_db,
        "passed": bool(level < args.gate_db),
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(f"{path.name}  {report['db']:+.2f} dB  max {report['max_abs_diff']:.3e}  "
          f"device {report['device']}  {'ok' if report['passed'] else 'FAILED'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
