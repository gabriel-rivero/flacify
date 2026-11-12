#!/usr/bin/env python3
"""Export FlashSR to ONNX with the sampler's noise as an explicit input.

This is E1-1 and E1-2 of the spike. The graph it writes is the one `crates/infer` will
load: audio in, audio out, both `[channels, 245760]` fp32, with a second input for the
noise that upstream draws internally. The lowpass preprocessing is *not* in the graph;
the Rust side does it and `crates/core/tests/fixtures.rs` pins it against scipy.

What it checks before it calls the export good:

  * the graph has no random op left (a leftover `RandomNormalLike` would make every
    render differ, and CoreML cannot run it);
  * the graph has no `STFT`/`MelWeightMatrix`/`Loop`/`Scan`/`If` without saying so,
    because those are the ops that decide whether an execution provider can take it;
  * onnxruntime reproduces the eager PyTorch output at a fixed noise tensor, to better
    than the gate (default -40 dB, ADR-002's kill criterion).

    python3 tools/export_onnx.py --model-dir ~/models/flashsr
    python3 tools/export_onnx.py --model-dir ~/models/flashsr --allow-gate-fail

Writes tools/out/flashsr.onnx and tools/out/export_report.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import flashsr_onnx as h  # noqa: E402

# Ops that must not survive into the shipping graph, and why.
NONDETERMINISTIC_OPS = {
    "RandomNormal",
    "RandomNormalLike",
    "RandomUniform",
    "RandomUniformLike",
    "Multinomial",
    "Bernoulli",
}
# Ops a backend may refuse. The spike reports them rather than hiding them.
PORTABILITY_RISK_OPS = {
    "STFT",
    "DFT",
    "MelWeightMatrix",
    "Loop",
    "Scan",
    "If",
    "GridSample",
    "NonMaxSuppression",
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    h.add_common_args(ap)
    ap.add_argument("--out", default=str(TOOLS / "out" / "flashsr.onnx"))
    ap.add_argument("--report", default=str(TOOLS / "out" / "export_report.json"))
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset; the dynamo exporter needs 18 or newer and silently raises "
        "anything lower, then tries to convert back, which fails on this graph",
    )
    ap.add_argument(
        "--gate-db",
        type=float,
        default=-40.0,
        help="fail if the ONNX output is worse than this, relative to PyTorch (ADR-002)",
    )
    ap.add_argument(
        "--allow-gate-fail",
        action="store_true",
        help="report the number and exit 0 anyway; for measuring a known-bad graph",
    )
    return ap.parse_args(argv)


def fixture_cond(channels: int):
    """A realistic model input, if the scipy fixtures are present.

    `tools/fixtures/gen.py` writes exactly one window of preprocessed audio, which is
    what the graph expects. Falling back to a synthetic signal keeps the script usable
    without them, but the numbers are only meaningful with a real spectrum.
    """
    import numpy as np

    path = TOOLS / "fixtures" / "out" / "music_cut16k.out.f32"
    if path.is_file():
        data = h.read_f32(path, channels)
        if data.shape[1] == h.WINDOW:
            return data, f"fixtures/{path.name}"
    t = np.arange(h.WINDOW) / h.SAMPLE_RATE
    rng = np.random.default_rng(0)
    signal = sum(0.1 * np.sin(2 * np.pi * f * t) for f in (220.0, 880.0, 3520.0))
    signal = signal + 0.02 * rng.standard_normal(h.WINDOW)
    stereo = np.stack([signal, np.roll(signal, 17)]).astype(np.float32)
    return stereo[:channels], "synthetic"


def external_data_path(out_path: Path) -> Path:
    """Where the dynamo exporter puts initializers past protobuf's 2 GB limit."""
    return out_path.with_name(out_path.name + ".data")


def sha256_of(path: Path) -> str:
    """Chunked so a multi-gigabyte graph does not have to fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def export_graph(module, inputs, out_path: Path, opset: int) -> str:
    """Write the ONNX graph and say which exporter produced it.

    torch's TorchScript exporter serialises the whole ModelProto through protobuf,
    which is capped at 2 GB. FlashSR's weights are 3.3 GB, so that path cannot write
    this graph at all; the dynamo exporter can, putting the weights in a sidecar
    `flashsr.onnx.data`. Dynamo goes first and TorchScript stays as a fallback, so a
    smaller model, or a torch that predates dynamo, still exports.
    """
    import torch

    errors: list[str] = []
    if h._torch_at_least(2, 6):
        try:
            torch.onnx.export(
                module,
                inputs,
                str(out_path),
                input_names=["cond", "noise"],
                output_names=["audio"],
                opset_version=opset,
                dynamo=True,
                external_data=True,
            )
            require_written(out_path)
            return "dynamo"
        except SystemExit as exc:
            # Upstream helpers call exit() on a type mismatch, which would otherwise
            # end the process with a clean code and no model to show for it.
            errors.append(f"dynamo: upstream called exit({exc.code}) during export")
            out_path.unlink(missing_ok=True)
            external_data_path(out_path).unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001 - a failed attempt is what fallback is for
            errors.append(f"dynamo: {type(exc).__name__}: {exc}")
            out_path.unlink(missing_ok=True)
            external_data_path(out_path).unlink(missing_ok=True)

    try:
        torch.onnx.export(
            module,
            inputs,
            str(out_path),
            input_names=["cond", "noise"],
            output_names=["audio"],
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
        require_written(out_path)
        return "torchscript"
    except SystemExit as exc:
        errors.append(f"torchscript: upstream called exit({exc.code}) during export")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"torchscript: {type(exc).__name__}: {exc}")
    raise RuntimeError("both ONNX exporters failed: " + " | ".join(errors))


def require_written(out_path: Path) -> None:
    """An exporter that returns without a file has failed, whatever it returned."""
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise RuntimeError(f"{out_path} was not written")


def export(args: argparse.Namespace) -> tuple[int, dict]:
    import numpy as np

    torch = h.require_torch()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32))

    model_dir = h.resolve_model_dir(args.model_dir)
    pkg_dir = h.resolve_package_dir(args.flashsr_dir)
    model = h.load_model(model_dir, pkg_dir, args.device)
    patched = h.force_deterministic_vae(model)
    patched_mel = h.patch_mel_front_end()
    module = h.make_sampler_module(model)

    shape = h.latent_shape(model, args.channels, h.WINDOW, args.device)
    noise = h.noise_tensor(shape, args.seed, args.device)
    cond_values, cond_source = fixture_cond(args.channels)
    cond = torch.from_numpy(cond_values).to(device=args.device, dtype=torch.float32)

    with torch.no_grad():
        reference = h.reference_output(model, cond, noise).cpu().numpy()
        eager = module(cond, noise).cpu().numpy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    backend = export_graph(module, (cond, noise), out_path, args.opset)

    import onnx

    # The weights are ~2.5 GB, past protobuf's 2 GB limit, so they live in a sidecar
    # `.data` file. Read the graph structure without it for the histogram, but check the
    # model by path: the checker resolves external data relative to the file and cannot
    # verify a bare ModelProto that has none loaded.
    graph = onnx.load(str(out_path), load_external_data=False)
    onnx.checker.check_model(str(out_path))
    # torch's dynamo exporter raises the opset to 18 whatever is asked for, so record
    # what the file actually contains rather than what was requested.
    exported_opset = max(
        (entry.version for entry in graph.opset_import if entry.domain in ("", "ai.onnx")),
        default=args.opset,
    )
    histogram: dict[str, int] = {}
    for node in graph.graph.node:
        histogram[node.op_type] = histogram.get(node.op_type, 0) + 1

    nondeterministic = sorted(NONDETERMINISTIC_OPS & histogram.keys())
    risky = sorted(PORTABILITY_RISK_OPS & histogram.keys())

    import onnxruntime as ort

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(
        None,
        {
            "cond": cond_values,
            "noise": noise.cpu().numpy(),
        },
    )[0]

    diff = float(np.max(np.abs(onnx_out - reference)))
    level = h.db(onnx_out - reference, reference)
    eager_level = h.db(eager - reference, reference)
    finite = bool(np.isfinite(onnx_out).all())

    report = {
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "opset": exported_opset,
        "opset_requested": args.opset,
        "device": args.device,
        "seed": args.seed,
        "channels": args.channels,
        "window": h.WINDOW,
        "cond_source": cond_source,
        "latent_shape": list(shape),
        "output_shape": list(onnx_out.shape),
        "onnx_sha256": sha256_of(out_path),
        "onnx_bytes": out_path.stat().st_size,
        "external_data_bytes": sidecar.stat().st_size if (sidecar := external_data_path(out_path)).is_file() else 0,
        "export_backend": backend,
        "node_count": sum(histogram.values()),
        "op_histogram": dict(sorted(histogram.items(), key=lambda kv: -kv[1])),
        "nondeterministic_ops": nondeterministic,
        "portability_risk_ops": risky,
        "deterministic_vae": patched,
        "traceable_mel_front_end": patched_mel,
        "eager_vs_upstream_db": round(eager_level, 2),
        "onnx_vs_upstream_db": round(level, 2),
        "max_abs_diff": diff,
        "output_is_finite": finite,
        "gate_db": args.gate_db,
    }

    failures = []
    if nondeterministic:
        failures.append(f"random ops survived export: {', '.join(nondeterministic)}")
    if not finite:
        failures.append("the graph produced NaN or Inf")
    if eager_level > -80.0:
        failures.append(
            f"the explicit-noise wrapper already differs from upstream by {eager_level:.1f} dB"
        )
    if level > args.gate_db:
        failures.append(f"ONNX is {level:.1f} dB from PyTorch, gate is {args.gate_db:.0f} dB")

    report["failures"] = failures
    report["passed"] = not failures
    return (0 if (not failures or args.allow_gate_fail) else 1), report


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        code, report = export(args)
    except SystemExit:
        raise
    except Exception as exc:  # export failures are the spike's answer, not a crash
        print(f"export failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")

    print(f"wrote      {args.out}  ({report['onnx_bytes'] / 1e9:.2f} GB + "
          f"{report['external_data_bytes'] / 1e9:.2f} GB external, {report['export_backend']})")
    print(f"opset      {report['opset']}   nodes {report['node_count']}")
    print(f"latent     {report['latent_shape']}   output {report['output_shape']}")
    print(f"cond       {report['cond_source']}")
    print(f"eager      {report['eager_vs_upstream_db']:+.1f} dB vs upstream")
    print(f"onnx       {report['onnx_vs_upstream_db']:+.1f} dB vs upstream (gate {args.gate_db:+.0f})")
    if report["portability_risk_ops"]:
        print(f"risk ops   {', '.join(report['portability_risk_ops'])}")
    if report["nondeterministic_ops"]:
        print(f"random ops {', '.join(report['nondeterministic_ops'])}")
    for failure in report["failures"]:
        print(f"FAILED     {failure}", file=sys.stderr)
    print(f"report     {args.report}")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
