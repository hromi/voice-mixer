"""Shared WAV primitives: read/resample/write/concat, plus MP3 compression
via ffmpeg. Pure stdlib `wave` + numpy for everything except the MP3 step
(no soundfile/librosa dependency needed for any of this) — used by both
chorus.py (simultaneous mixing) and manifesto.py (sequential concatenation
of a full reading).
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np


def read_wav(path: Path) -> tuple[np.ndarray, int]:
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


def resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear-interpolation resample — good enough for mixing/
    concatenating spoken-word clips; not worth a librosa/soundfile
    dependency just for this."""
    if orig_sr == target_sr or len(data) == 0:
        return data
    duration = len(data) / orig_sr
    target_len = max(1, int(round(duration * target_sr)))
    orig_idx = np.linspace(0, len(data) - 1, num=len(data))
    target_idx = np.linspace(0, len(data) - 1, num=target_len)
    return np.interp(target_idx, orig_idx, data)


def write_wav(path: Path, data: np.ndarray, sr: int) -> None:
    clipped = np.clip(data, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def concat_clips(clips: list[Path], out_path: Path, gap_s: float = 0.4) -> None:
    """Sequential concatenation (not a simultaneous mix — see chorus.py for
    that) with a short silence between clips, resampling everything to the
    highest sample rate among them first."""
    if not clips:
        raise ValueError("no clips to concatenate")
    loaded = [read_wav(p) for p in clips]
    target_sr = max(sr for _, sr in loaded)
    resampled = [resample(data, sr, target_sr) for data, sr in loaded]
    gap = np.zeros(int(gap_s * target_sr))

    pieces = []
    for i, clip in enumerate(resampled):
        pieces.append(clip)
        if i != len(resampled) - 1:
            pieces.append(gap)
    write_wav(out_path, np.concatenate(pieces), target_sr)


class FfmpegNotFound(RuntimeError):
    pass


def compress_to_mp3(wav_path: Path, mp3_path: Path, bitrate: str = "96k") -> Path:
    """Compress a WAV to MP3 via ffmpeg (much smaller for emailing than
    raw PCM — a 6-minute mono 24kHz WAV is ~17MB; at 96kbps MP3 it's
    ~4MB)."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FfmpegNotFound("ffmpeg not found on PATH — required to compress audio for email")
    result = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(wav_path), "-b:a", bitrate, str(mp3_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to compress {wav_path}: {result.stderr}")
    return mp3_path
