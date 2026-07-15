"""Tone-color (speaker) embedding extraction and communal blending.

OpenVoice's `ToneColorConverter.extract_se` reduces one or more reference
recordings to a single embedding tensor by averaging per-segment
embeddings. We reuse that same idea one level up: a speaker's embedding is
the average of their per-file embeddings, and a *communal* embedding is a
weighted average across multiple speakers (optionally restricted to
specific files each), which performs a crossover in tone-color space
between two or more human voices.

Per-file embeddings are cached to disk (keyed by file checksum) since
extraction requires running the reference encoder and is comparatively
expensive; recomputation is only triggered when a recording's contents
change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from . import config, storage


@dataclass
class MixEntry:
    speaker: str
    weight: float = 1.0
    files: Optional[list[str]] = None  # filenames within voices/<speaker>/; None = all


def _embedding_cache_path(speaker: str, file_checksum: str) -> Path:
    d = config.EMBEDDING_CACHE_DIR / storage.sanitize_speaker_name(speaker)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{file_checksum}.pt"


def get_file_embedding(converter: Any, speaker: str, file_path: Path):
    """Return the cached (or freshly extracted) tone-color embedding for one recording."""
    import torch

    checksum = storage.checksum_file(file_path)
    cache_path = _embedding_cache_path(speaker, checksum)
    if cache_path.exists():
        return torch.load(cache_path, map_location=converter.device)

    se = converter.extract_se([str(file_path)])
    torch.save(se, cache_path)
    return se


def speaker_embedding(converter: Any, speaker: str, files: Optional[list[str]] = None):
    """Average embedding across a speaker's recordings (or a chosen subset)."""
    all_recordings = {p.name: p for p in storage.list_recordings(speaker)}
    if not all_recordings:
        raise ValueError(f"speaker {speaker!r} has no recordings in {storage.speaker_dir(speaker)}")

    if files:
        missing = [f for f in files if f not in all_recordings]
        if missing:
            raise ValueError(f"unknown recordings for {speaker!r}: {missing}")
        selected = [all_recordings[f] for f in files]
    else:
        selected = list(all_recordings.values())

    embeddings = [get_file_embedding(converter, speaker, p) for p in selected]
    used_filenames = [p.name for p in selected]
    if len(embeddings) == 1:
        return embeddings[0], used_filenames

    stacked_sum = embeddings[0]
    for e in embeddings[1:]:
        stacked_sum = stacked_sum + e
    return stacked_sum / len(embeddings), used_filenames


def blend_embeddings(weighted: list[tuple[Any, float]]):
    """Weighted average of embeddings (weights normalized to sum to 1)."""
    if not weighted:
        raise ValueError("no embeddings to blend")
    total_weight = sum(w for _, w in weighted)
    if total_weight <= 0:
        raise ValueError("total weight must be positive")

    result = None
    for embedding, weight in weighted:
        term = embedding * (weight / total_weight)
        result = term if result is None else result + term
    return result


def communal_embedding(converter: Any, mix: list[MixEntry]):
    """Resolve a communal mix spec into a single blended target embedding.

    Returns (blended_embedding, detail) where `detail` records exactly which
    recordings and weights were used per speaker, for the metadata sidecar.
    """
    if not mix:
        raise ValueError("mix must contain at least one speaker")

    weighted = []
    detail = []
    for entry in mix:
        se, used_files = speaker_embedding(converter, entry.speaker, entry.files)
        weighted.append((se, entry.weight))
        detail.append({"speaker": entry.speaker, "weight": entry.weight, "files": used_files})

    total_weight = sum(e.weight for e in mix)
    for d in detail:
        d["normalized_weight"] = d["weight"] / total_weight

    return blend_embeddings(weighted), detail
