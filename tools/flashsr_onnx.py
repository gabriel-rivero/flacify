#!/usr/bin/env python3
"""Shared harness for the FlashSR -> ONNX spike.

`export_onnx.py` and `compare_onnx.py` both need the same three things, and they have
to agree exactly or the comparison measures the harness instead of the graph:

  * the model, loaded the way the reference inference package loads it;
  * one sampler step written as an [`nn.Module`] with the noise as an explicit input,
    replacing `torch.randn` inside `DiffusersWrapper.infer` (ADR-006);
  * a reference run that goes through upstream's own `FlashSR.forward`, with the noise
    pinned to the same tensor so the only difference is eager PyTorch vs ONNX.

Nothing here imports torch at module import time: the scripts must fail with a useful
message on a machine without it, not with an ImportError traceback at parse time.

The window is fixed at 5.12 s of 48 kHz stereo, the length FlashSR was trained on.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SAMPLE_RATE = 48_000
WINDOW = 245_760  # 5.12 s, the model's trained length
WINDOW_SECS = WINDOW / SAMPLE_RATE


def add_common_args(parser) -> None:
    """Model location and device, shared by both scripts."""
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("FLASHSR_MODEL_DIR"),
        help="directory holding student_ldm.pth, sr_vocoder.pth and vae.pth "
        "(default: $FLASHSR_MODEL_DIR)",
    )
    parser.add_argument(
        "--flashsr-dir",
        default=os.environ.get("FLASHSR_PKG_DIR"),
        help="directory that contains the FlashSR/ and TorchJaekwon/ packages "
        "(default: $FLASHSR_PKG_DIR, then the installed package)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="device for the PyTorch reference (cpu keeps the comparison honest)",
    )


def require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise SystemExit(
            "this spike needs torch, which is not installed here.\n"
            "  python3 -m venv .venv && . .venv/bin/activate\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  pip install -r tools/requirements-spike.txt\n"
            f"(import error: {exc})"
        ) from exc
    return torch


def resolve_package_dir(explicit: str | None) -> Path:
    """Directory to put on sys.path so that `import FlashSR` works."""
    import importlib.util

    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not (path / "FlashSR" / "FlashSR.py").is_file():
            raise SystemExit(f"{path} does not contain FlashSR/FlashSR.py")
        return path

    spec = importlib.util.find_spec("FlashSR")
    if spec is not None and spec.origin is not None:
        candidate = Path(spec.origin).resolve().parent.parent
        if (candidate / "FlashSR" / "FlashSR.py").is_file():
            return candidate

    raise SystemExit(
        "cannot find the FlashSR inference package.\n"
        "  git clone https://github.com/jakeoneijk/FlashSR_Inference ./flashsr\n"
        "  --flashsr-dir ./flashsr      (or set FLASHSR_PKG_DIR)"
    )


def resolve_model_dir(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        raise SystemExit(
            "no model directory: pass --model-dir or set FLASHSR_MODEL_DIR.\n"
            "  weights: https://huggingface.co/jakeoneijk/FlashSR_weights"
        )
    missing = [n for n in ("student_ldm.pth", "sr_vocoder.pth", "vae.pth") if not (path / n).is_file()]
    if missing:
        raise SystemExit(f"{path} is missing {', '.join(missing)}")
    return path


def load_model(model_dir: Path, pkg_dir: Path, device: str):
    """Build FlashSR exactly as the reference wrapper does, with CPU-safe loading."""
    torch = require_torch()
    if str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))

    from FlashSR.FlashSR import FlashSR

    # Published checkpoints were saved from CUDA tensors and read with a bare
    # torch.load(), which fails on a CPU-only machine. Upstream also relies on the
    # full pickle, so opt out of the weights_only default that torch >= 2.6 picked up.
    original_load = torch.load

    def pinned_load(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu" if torch.device(device).type == "cpu" else device)
        if _torch_at_least(2, 6):
            kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = pinned_load
    try:
        model = FlashSR(
            str(model_dir / "student_ldm.pth"),
            str(model_dir / "sr_vocoder.pth"),
            str(model_dir / "vae.pth"),
        )
    finally:
        torch.load = original_load

    return model.eval().to(device)


def force_deterministic_vae(model) -> str:
    """Make the VAE encode deterministic by using the posterior mean.

    `VAEWrapper.encode_to_z` calls `DiagonalGaussianDistribution.sample()`, which draws
    an unseeded `torch.randn` every window. Upstream therefore adds a second RNG stream
    nobody can reproduce or A/B against. At inference the posterior mean is the usual
    choice, and it leaves the sampler's noise as the single explicit input ADR-006
    specifies. Returns the class that was patched, for the report.
    """
    from FlashSR.AudioSR.autoencoder import DiagonalGaussianDistribution

    if getattr(DiagonalGaussianDistribution, "_flacify_mode", False):
        return "FlashSR.AudioSR.autoencoder.DiagonalGaussianDistribution"
    DiagonalGaussianDistribution.sample = lambda self: self.mean  # type: ignore[method-assign]
    DiagonalGaussianDistribution._flacify_mode = True
    return "FlashSR.AudioSR.autoencoder.DiagonalGaussianDistribution"


def make_sampler_module(model):
    """One DPMSolver step, as an `nn.Module` whose noise is an input.

    This mirrors `DiffusersWrapper.infer` line for line: preprocess (mel -> VAE encode),
    scale the noise by `init_noise_sigma`, run one scheduler step, postprocess
    (VAE decode -> SR vocoder -> denormalise), then trim to the input length.
    """
    torch = require_torch()
    from TorchJaekwon.Model.Diffusion.External.diffusers.schedulers.scheduling_dpmsolver_multistep import (
        DPMSolverMultistepScheduler,
    )
    from TorchJaekwon.Model.Diffusion.External.diffusers.DiffusersWrapper import (
        DiffusersWrapper,
    )

    scheduler_config = DiffusersWrapper.get_diffusers_scheduler_config(
        model, {"timestep_spacing": "trailing"}
    )
    scheduler = DPMSolverMultistepScheduler(**scheduler_config)
    scheduler.set_timesteps(1)

    class SamplerStep(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            self.scheduler = scheduler

        def forward(self, cond, noise):
            _, latent_cond, extra = self.model.preprocess(x_start=None, cond=cond)
            x = noise * self.scheduler.init_noise_sigma
            for t in self.scheduler.timesteps:
                denoiser_input = self.scheduler.scale_model_input(x, t)
                model_output = self.model.apply_model(
                    denoiser_input,
                    torch.full((x.shape[0],), t, device=x.device, dtype=torch.long),
                    latent_cond,
                    False,
                    cfg_scale=self.model.cfg_scale,
                )
                x = self.scheduler.step(model_output, t, x, return_dict=False)[0]
            audio = self.model.postprocess(x, extra)
            return audio[..., : cond.shape[-1]]

    return SamplerStep().eval()


def latent_shape(model, channels: int, frames: int, device: str) -> tuple[int, ...]:
    """Shape of the noise tensor: the VAE latent grid, not the audio window."""
    torch = require_torch()
    with torch.no_grad():
        cond = torch.zeros(channels, frames, device=device)
        _, z, _ = model.preprocess(x_start=None, cond=cond)
    return tuple(z.shape)


def noise_tensor(shape: tuple[int, ...], seed: int, device: str):
    """The explicit sampler noise, identical for both paths."""
    import numpy as np

    torch = require_torch()
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(shape, dtype=np.float32)
    return torch.from_numpy(values).to(device=device, dtype=torch.float32)


def reference_output(model, cond, noise):
    """Upstream `FlashSR.forward`, with its internal `torch.randn` pinned to `noise`.

    Patching the global is the only way to make upstream's unseeded draw reproducible;
    the wrapper asserts exactly one draw, so a silent change upstream turns into a
    failure here instead of a wrong number.
    """
    torch = require_torch()
    calls = {"n": 0}
    original = torch.randn

    def pinned(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise AssertionError("upstream drew more than one noise tensor")
        return noise

    torch.randn = pinned
    try:
        with torch.no_grad():
            out = model(cond, num_steps=1, lowpass_input=False)
    finally:
        torch.randn = original
    if calls["n"] != 1:
        raise AssertionError(f"upstream drew {calls['n']} noise tensors, expected 1")
    return out


def db(diff, reference) -> float:
    """Difference relative to the reference, in dB (negative is better)."""
    import numpy as np

    diff = np.asarray(diff, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    rms_diff = float(np.sqrt(np.mean(diff**2) + 1e-30))
    rms_ref = float(np.sqrt(np.mean(reference**2) + 1e-30))
    return 20.0 * np.log10(rms_diff / rms_ref + 1e-30)


def read_f32(path: Path, channels: int = 2) -> "object":
    """Planar little-endian f32, the fixture format."""
    import numpy as np

    data = np.fromfile(path, dtype="<f4")
    if data.size % channels:
        raise SystemExit(f"{path} holds {data.size} samples, not a multiple of {channels}")
    return data.reshape(channels, -1).astype(np.float32)


def window_from_track(path: Path, channels: int = 2):
    """First model window of a track, zero-padded if the track is shorter.

    `*.f32` is read as planar raw audio (the fixture format). Anything else goes
    through scipy's WAV reader and is downmixed or truncated to `channels`.
    """
    import numpy as np

    if path.suffix == ".f32":
        audio = read_f32(path, channels)
    else:
        from scipy.io import wavfile

        rate, data = wavfile.read(path)
        if rate != SAMPLE_RATE:
            raise SystemExit(f"{path} is {rate} Hz; this spike only handles {SAMPLE_RATE} Hz")
        if data.ndim == 1:
            data = data[None, :]
        if np.issubdtype(data.dtype, np.integer):
            scale = float(np.iinfo(data.dtype).max)
            data = data.astype(np.float32) / scale
        audio = data.T.astype(np.float32)
        if audio.shape[0] > channels:
            audio = audio[:channels]
        elif audio.shape[0] < channels:
            audio = np.vstack([audio, np.zeros((channels - audio.shape[0], audio.shape[1]), np.float32)])

    window = np.zeros((channels, WINDOW), dtype=np.float32)
    take = min(WINDOW, audio.shape[1])
    window[:, :take] = audio[:, :take]
    return window


def _torch_at_least(major: int, minor: int) -> bool:
    import torch

    parts = torch.__version__.split("+")[0].split(".")
    return (int(parts[0]), int(parts[1])) >= (major, minor)
