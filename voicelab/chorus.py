"""WAV mixing for "chorus" mode: several individually cloned voices summed
into one simultaneous ensemble — not blended in embedding space (that's
communal mixing, see embeddings.py) and not concatenated one after
another, but genuinely overlaid so multiple voices say the same text at
the same time.

Pure stdlib `wave` + numpy, deliberately avoiding soundfile/librosa as a
hard dependency — chorus mode should work with just the base
requirements.txt plus whichever TTS backend actually produced the clips.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as f:
        sr = f.getframerate()
        n_frames = f.getnframes()
        sampwidth = f.getsampwidth()
        channels = f.getnchannels()
        raw = f.readframes(n_frames)
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth)
    if dtype is None:
        raise ValueError(f"unsupported WAV sample width: {sampwidth} bytes ({path})")
    data = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    max_val = float(2 ** (8 * sampwidth - 1))
    return data / max_val, sr


def _resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear-interpolation resample — good enough for mixing
    spoken-word clips together; not worth a librosa/soundfile dependency
    just for this."""
    if orig_sr == target_sr or len(data) == 0:
        return data
    duration = len(data) / orig_sr
    target_len = max(1, int(round(duration * target_sr)))
    orig_idx = np.linspace(0, len(data) - 1, num=len(data))
    target_idx = np.linspace(0, len(data) - 1, num=target_len)
    return np.interp(target_idx, orig_idx, data)


def _write_wav(path: Path, data: np.ndarray, sr: int) -> None:
    clipped = np.clip(data, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def mix_chorus(clips: list[tuple[Path, float]], out_path: Path) -> None:
    """`clips`: [(wav_path, relative_gain), ...]. Sums every clip as a
    simultaneous ensemble. Resamples to the highest sample rate among the
    clips and zero-pads shorter ones so they all start together; the
    output plays for as long as the longest clip. Peak-normalizes the sum
    so combining several voices never clips/distorts.
    """
    if len(clips) < 2:
        raise ValueError("chorus mixing needs at least 2 clips")

    loaded = [(_read_wav(p), gain) for p, gain in clips]
    target_sr = max(sr for (_, sr), _ in loaded)
    resampled = [(_resample(data, sr, target_sr), gain) for (data, sr), gain in loaded]
    max_len = max(len(data) for data, _ in resampled)

    mixed = np.zeros(max_len, dtype=np.float64)
    for data, gain in resampled:
        padded = np.pad(data, (0, max_len - len(data)))
        mixed += padded * gain

    peak = float(np.max(np.abs(mixed))) if max_len else 0.0
    if peak > 0.99:
        mixed = mixed / peak * 0.99

    _write_wav(out_path, mixed, target_sr)
