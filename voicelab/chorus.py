"""WAV mixing for "chorus" mode: several individually cloned voices summed
into one simultaneous ensemble — not blended in embedding space (that's
communal mixing, see embeddings.py) and not concatenated one after
another (see audio.concat_clips for that), but genuinely overlaid so
multiple voices say the same text at the same time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio import read_wav, resample, write_wav


def mix_chorus(clips: list[tuple[Path, float]], out_path: Path) -> None:
    """`clips`: [(wav_path, relative_gain), ...]. Sums every clip as a
    simultaneous ensemble. Resamples to the highest sample rate among the
    clips and zero-pads shorter ones so they all start together; the
    output plays for as long as the longest clip. Peak-normalizes the sum
    so combining several voices never clips/distorts.
    """
    if len(clips) < 2:
        raise ValueError("chorus mixing needs at least 2 clips")

    loaded = [(read_wav(p), gain) for p, gain in clips]
    target_sr = max(sr for (_, sr), _ in loaded)
    resampled = [(resample(data, sr, target_sr), gain) for (data, sr), gain in loaded]
    max_len = max(len(data) for data, _ in resampled)

    mixed = np.zeros(max_len, dtype=np.float64)
    for data, gain in resampled:
        padded = np.pad(data, (0, max_len - len(data)))
        mixed += padded * gain

    peak = float(np.max(np.abs(mixed))) if max_len else 0.0
    if peak > 0.99:
        mixed = mixed / peak * 0.99

    write_wav(out_path, mixed, target_sr)
