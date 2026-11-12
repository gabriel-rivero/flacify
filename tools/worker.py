#!/usr/bin/env python3
"""FlashSR as a resident inference worker: one model, one pipe, no protocol library.

The player spawns this, keeps it alive, and feeds it one window at a time. It holds
the model in memory, so a window costs one forward pass instead of a 15 s import, and
it runs on Metal (MPS), where the measured cost is 2.6 s per 5.12 s window (RTF 0.51)
against 110 s for the ONNX Runtime CPU path.

stdout carries the protocol and nothing else. Upstream's `print` and tqdm output would
corrupt it, so fd 1 is duplicated into PROTO and then pointed at stderr: every stray
print from FlashSR lands in the log instead of inside a header.

Protocol, little-endian:

    request  28 bytes  b"FLW1" u8 cmd  3x pad  u64 id  u32 channels  u32 frames  u32 noise_len
    reply    28 bytes  b"FLR1" u8 cmd  u8 status 2x pad u64 id  u32 channels  u32 frames  u32 len
    payload  `len` bytes: f32 planar audio (status 0), UTF-8 message (status 1),
             or JSON (cmd HELLO)

Commands: HELLO = 1, WINDOW = 2, SHUTDOWN = 3.

    FLASHSR_MODEL_DIR=~/models/flashsr python3 tools/worker.py --device mps
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

HEADER = struct.Struct("<4sB3xQIII")
REPLY = struct.Struct("<4sBB2xQIII")

CMD_HELLO = 1
CMD_WINDOW = 2
CMD_SHUTDOWN = 3

REQUEST_MAGIC = b"FLW1"
REPLY_MAGIC = b"FLR1"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", default=os.environ.get("FLASHSR_MODEL_DIR", str(Path.home() / "models" / "flashsr")))
    ap.add_argument("--flashsr-dir", default=os.environ.get("FLASHSR_PKG_DIR"))
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    return ap.parse_args(argv)


def pick_device(requested: str):
    import torch

    if requested == "mps":
        if not torch.backends.mps.is_available():
            log("mps is not available; falling back to cpu")
            return torch.device("cpu"), "cpu"
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


class Worker:
    def __init__(self, args: argparse.Namespace) -> None:
        import flashsr_onnx as h

        self.h = h
        self.torch = h.require_torch()
        started = time.perf_counter()

        model_dir = h.resolve_model_dir(args.model_dir)
        pkg_dir = h.resolve_package_dir(args.flashsr_dir)
        self.device, self.device_name = pick_device(args.device)
        model = h.load_model(model_dir, pkg_dir, self.device_name)
        # Same two patches the export needed: the VAE's second RNG must not exist, and
        # the mel front-end's type() checks and print guards must not be traced.
        h.force_deterministic_vae(model)
        h.patch_mel_front_end()
        model.to(self.device)
        self.module = h.make_sampler_module(model)
        self.load_secs = time.perf_counter() - started
        self.shapes: dict[tuple[int, int], tuple[int, ...]] = {}
        self.warmup_secs = self._warmup()

    def latent_shape(self, channels: int, frames: int) -> tuple[int, ...]:
        key = (channels, frames)
        if key not in self.shapes:
            self.shapes[key] = self.h.latent_shape(self.module.model, channels, frames, self.device_name)
        return self.shapes[key]

    def _warmup(self) -> float:
        """Compile the Metal kernels before anyone is waiting on a real window."""
        torch = self.torch
        noise = torch.zeros(self.latent_shape(2, self.h.WINDOW), device=self.device)
        cond = torch.zeros(2, self.h.WINDOW, device=self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            self.module(cond, noise)
        elapsed = time.perf_counter() - started
        log(f"warmup {elapsed:.2f} s on {self.device_name}")
        return elapsed

    def hello(self) -> dict:
        return {
            "protocol": 1,
            "device": self.device_name,
            "torch": self.torch.__version__,
            "window": self.h.WINDOW,
            "sample_rate": self.h.SAMPLE_RATE,
            "noise_len_stereo": int(
                self.torch.tensor(self.latent_shape(2, self.h.WINDOW)).prod().item()
            ),
            "latent_shape_stereo": list(self.latent_shape(2, self.h.WINDOW)),
            "load_secs": round(self.load_secs, 2),
            "warmup_secs": round(self.warmup_secs, 2),
        }

    def window(self, cond_bytes: bytes, noise_bytes: bytes, channels: int, frames: int):
        torch = self.torch
        cond = torch.frombuffer(bytearray(cond_bytes), dtype=torch.float32).reshape(channels, frames)
        shape = self.latent_shape(channels, frames)
        noise = torch.frombuffer(bytearray(noise_bytes), dtype=torch.float32).reshape(shape)
        with torch.inference_mode():
            audio = self.module(cond.to(self.device), noise.to(self.device))
        return audio[..., :frames].float().cpu().numpy().tobytes()


def read_exact(stream, size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            return None
        chunks.extend(chunk)
    return bytes(chunks)


def serve(worker: Worker, stream_in, stream_out) -> int:
    windows = 0
    started = time.perf_counter()
    while True:
        header = read_exact(stream_in, HEADER.size)
        if header is None:
            log("stdin closed; exiting")
            return 0
        magic, cmd, window_id, channels, frames, noise_len = HEADER.unpack(header)
        if magic != REQUEST_MAGIC:
            log(f"bad request magic {magic!r}; exiting")
            return 2

        if cmd == CMD_SHUTDOWN:
            log(f"shutdown after {windows} windows")
            return 0

        if cmd == CMD_HELLO:
            body = json.dumps(worker.hello()).encode()
            stream_out.write(REPLY.pack(REPLY_MAGIC, cmd, 0, window_id, 0, 0, len(body)) + body)
            continue

        if cmd != CMD_WINDOW:
            body = f"unknown command {cmd}".encode()
            stream_out.write(REPLY.pack(REPLY_MAGIC, cmd, 1, window_id, 0, 0, len(body)) + body)
            continue

        expected_noise = len(worker.latent_shape(channels, frames))
        payload = read_exact(stream_in, channels * frames * 4)
        noise = read_exact(stream_in, noise_len * 4)
        if payload is None or noise is None:
            log("truncated window; exiting")
            return 2

        t0 = time.perf_counter()
        try:
            if expected_noise and noise_len != int(
                worker.torch.tensor(worker.latent_shape(channels, frames)).prod().item()
            ):
                raise ValueError(
                    f"noise_len {noise_len} does not match the graph's latent grid "
                    f"{worker.latent_shape(channels, frames)}"
                )
            audio = worker.window(payload, noise, channels, frames)
        except Exception as exc:  # a bad window must not kill the session
            body = f"{type(exc).__name__}: {exc}".encode()
            stream_out.write(REPLY.pack(REPLY_MAGIC, cmd, 1, window_id, 0, 0, len(body)) + body)
            log(f"window {window_id} failed: {exc}")
            continue

        elapsed = time.perf_counter() - t0
        stream_out.write(
            REPLY.pack(REPLY_MAGIC, cmd, 0, window_id, channels, frames, len(audio)) + audio
        )
        windows += 1
        audio_secs = frames / worker.h.SAMPLE_RATE
        log(
            f"window {window_id}: {elapsed:.2f} s  RTF {elapsed / audio_secs:.2f}  "
            f"({windows} served, {time.perf_counter() - started:.0f} s total)"
        )


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    # Take the real descriptors first, then point the process's stdout at stderr so
    # nothing upstream can write into the protocol stream.
    stream_in = os.fdopen(os.dup(0), "rb", buffering=0)
    stream_out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    worker = Worker(args)
    log(f"ready on {worker.device_name} (load {worker.load_secs:.1f} s)")
    return serve(worker, stream_in, stream_out)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
