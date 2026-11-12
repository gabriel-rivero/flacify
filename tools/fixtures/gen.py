#!/usr/bin/env python3
"""Generate golden fixtures for the Rust port of FlashSR's preprocessing.

The Rust side must reproduce two Python functions, both of which change every
sample the model sees:

  1. FlashSR.Util.UtilAudioSR.find_cutoff_freq  (STFT, 98.3% cumulative energy)
  2. FlashSR.Util.UtilAudioLowPassFilter.lowpass (cheby1 order 8 rp 0.1, then a
     resample_poly round trip through the cutoff rate)

Both are reimplemented here with numpy/scipy, mirroring the upstream code line for
line, so fixtures can be produced without torch or a GPU. The intermediate after
the zero-phase filter is saved too: when the Rust output diverges, the first
diverging stage is what matters.

Outputs land in tools/fixtures/out/ (gitignored) as raw little-endian f32 plus a
JSON descriptor.

    python3 tools/fixtures/gen.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import cheby1, resample_poly, sosfiltfilt

SR = 48_000
WINDOW = 245_760  # 5.12 s, the model's trained length
N_FFT = 2_048
HOP = 480
PERCENTILE = 0.983
ORDER = 8
RIPPLE_DB = 0.1


def periodic_hann(n: int) -> np.ndarray:
    """torch.hann_window(n) is periodic; np.hanning is symmetric. Not the same."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def stft_magnitude(x: np.ndarray) -> np.ndarray:
    """[time, freq] magnitude, center=False, matching torch.stft in UtilAudioSR."""
    window = periodic_hann(N_FFT)
    frames = 1 + (len(x) - N_FFT) // HOP
    if frames <= 0:
        raise ValueError(f"signal of {len(x)} samples is shorter than one frame")
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    framed = x[idx] * window
    return np.abs(np.fft.rfft(framed, n=N_FFT, axis=1))


def find_cutoff_freq(audio: np.ndarray) -> int:
    """Port of UtilAudioSR.find_cutoff_freq, including its edge cases."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 2:
        x = x[0]  # upstream uses stft_spec[0]
    magnitude = stft_magnitude(x)
    energy = np.cumsum(magnitude.sum(axis=0))
    threshold = energy[-1] * PERCENTILE
    index = 0
    for i in range(1, len(energy)):
        if energy[-i] < threshold:
            index = len(energy) - i
            break
    cutoff = (index / 1024) * (SR // 2)
    if cutoff < 1000:
        cutoff = SR // 2
    return int(cutoff)


def lowpass(audio: np.ndarray, cutoff: int) -> np.ndarray:
    """Port of UtilAudioLowPassFilter.lowpass(audio, 48000, 'cheby', 8, cutoff)."""
    sos = cheby1(ORDER, RIPPLE_DB, cutoff / (SR / 2), btype="low", output="sos")
    out = np.empty_like(audio, dtype=np.float64)
    for c in range(audio.shape[0]):
        filtered = sosfiltfilt(sos, audio[c])
        out[c] = band_limit(filtered, cutoff)
    return out


def band_limit(x: np.ndarray, cutoff: int) -> np.ndarray:
    """resample_poly down to the cutoff rate and back, as upstream does."""
    fs_down = int((cutoff / (SR / 2)) * SR)
    if fs_down <= 0 or fs_down >= SR:
        return x.copy()
    down = resample_poly(x, fs_down, SR)
    up = resample_poly(down, SR, fs_down)
    if len(up) != len(x):
        up = np.pad(up, (0, len(x) - len(up))) if len(up) < len(x) else up[: len(x)]
    return up


def make_signal(rng: np.random.Generator, cutoff: int | None) -> np.ndarray:
    """Stereo, music-ish: pink noise plus a few drifting partials."""
    n = WINDOW
    t = np.arange(n) / SR
    base = rng.standard_normal(n)
    # 1/f tilt via a crude integrator/leak, then normalise.
    pink = np.zeros(n)
    acc = 0.0
    for i in range(n):
        acc = 0.98 * acc + base[i] * 0.02
        pink[i] = acc
    pink /= np.max(np.abs(pink)) + 1e-12
    music = 0.35 * pink
    for k, (freq, amp) in enumerate([(220.0, 0.25), (880.0, 0.15), (3520.0, 0.08), (11000.0, 0.05)]):
        music += amp * np.sin(2 * np.pi * (freq + 0.7 * k) * t + rng.uniform(0, 6.28))
    music += 0.02 * rng.standard_normal(n)
    music /= np.max(np.abs(music)) + 1e-12

    left = music
    right = 0.9 * np.roll(music, 17) + 0.05 * rng.standard_normal(n)
    stereo = np.stack([left, right]).astype(np.float64)

    if cutoff is not None:
        sos = cheby1(ORDER, RIPPLE_DB, cutoff / (SR / 2), btype="low", output="sos")
        stereo = np.stack([sosfiltfilt(sos, ch) for ch in stereo])
        stereo /= np.max(np.abs(stereo)) + 1e-12
    return stereo


CASES = [
    ("music_cut16k", 16_000),
    ("music_cut8k", 8_000),
    ("music_cut20k", 20_000),
    ("music_fullband", None),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "out"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {"sample_rate": SR, "window": WINDOW, "n_fft": N_FFT, "hop": HOP,
                "percentile": PERCENTILE, "order": ORDER, "ripple_db": RIPPLE_DB,
                "cases": []}

    for index, (name, true_cutoff) in enumerate(CASES):
        rng = np.random.default_rng(0xF1AC + index)
        audio = make_signal(rng, true_cutoff)
        detected = find_cutoff_freq(audio)

        filtered = np.empty_like(audio)
        sos = cheby1(ORDER, RIPPLE_DB, detected / (SR / 2), btype="low", output="sos")
        for c in range(audio.shape[0]):
            filtered[c] = sosfiltfilt(sos, audio[c])
        post = np.stack([band_limit(filtered[c], detected) for c in range(audio.shape[0])])

        audio.astype("<f4").tofile(out / f"{name}.in.f32")
        filtered.astype("<f4").tofile(out / f"{name}.filtered.f32")
        post.astype("<f4").tofile(out / f"{name}.out.f32")

        entry = {
            "name": name,
            "true_cutoff": true_cutoff,
            "detected_cutoff": detected,
            "channels": int(audio.shape[0]),
            "frames": int(audio.shape[1]),
            "rms_in": float(np.sqrt(np.mean(audio**2))),
            "rms_filtered": float(np.sqrt(np.mean(filtered**2))),
            "rms_out": float(np.sqrt(np.mean(post**2))),
            "max_abs_out": float(np.max(np.abs(post))),
        }
        manifest["cases"].append(entry)
        print(f"{name:16s} true={true_cutoff!s:>6s} detected={detected:5d} "
              f"rms_in={entry['rms_in']:.6f} rms_out={entry['rms_out']:.6f}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {len(CASES)} cases to {out}")


if __name__ == "__main__":
    main()
