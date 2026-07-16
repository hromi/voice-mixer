"""Filesystem storage layer for human recordings and synthetic (cloned) audio.

Layout on disk::

    voices/<Speaker>/<recording files>
    synthetic/<Speaker or Speaker+Speaker2>/<id>.wav
    synthetic/<Speaker or Speaker+Speaker2>/<id>.json   (metadata sidecar)

Every synthetic file's metadata sidecar records the original prompt plus
every parameter needed to reproduce the clone unambiguously (see
`SynthesisMetadata` in engine.py). This module also implements the
content-addressed cache lookup: if a request's canonical parameters hash to
a cache key already present in the target speaker folder, the cached result
is reused instead of resynthesizing.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import config

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")


class InvalidSpeakerName(ValueError):
    pass


def sanitize_speaker_name(name: str) -> str:
    """Normalize a speaker/mix-folder name and guard against path traversal.

    Raises InvalidSpeakerName if nothing usable remains after sanitizing,
    which also blocks names like "..", "/etc", or "" from ever reaching the
    filesystem.
    """
    name = name.strip()
    cleaned = _SAFE_NAME_RE.sub("", name)
    if not cleaned or cleaned in {".", ".."}:
        raise InvalidSpeakerName(f"invalid speaker name: {name!r}")
    return cleaned


def mix_folder_name(speaker_names: list[str]) -> str:
    """Deterministic folder name for a (possibly communal) speaker mix."""
    cleaned = sorted({sanitize_speaker_name(n) for n in speaker_names})
    return "+".join(cleaned)


def speaker_dir(speaker: str, create: bool = False) -> Path:
    path = config.VOICES_DIR / sanitize_speaker_name(speaker)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def synthetic_dir(folder_name: str, create: bool = False) -> Path:
    path = config.SYNTHETIC_DIR / sanitize_speaker_name(folder_name)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def list_speakers() -> list[str]:
    if not config.VOICES_DIR.exists():
        return []
    return sorted(p.name for p in config.VOICES_DIR.iterdir() if p.is_dir())


def list_recordings(speaker: str) -> list[Path]:
    d = speaker_dir(speaker)
    if not d.exists():
        return []
    return sorted(
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in config.AUDIO_EXTENSIONS
    )


def checksum_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def add_recording(speaker: str, source_path: Path, original_filename: Optional[str] = None,
                   source: str = "upload", move: bool = False, note: Optional[str] = None) -> Path:
    """Copy (or move) an audio file into voices/<speaker>/.

    `note` is a free-text label (e.g. a transcript of what was said, or any
    other description) stored in the recording's metadata sidecar — purely
    informational, it plays no role in embedding extraction.

    Returns the path of the stored recording. The stored filename is
    timestamped to avoid collisions while staying human-readable.
    """
    source_path = Path(source_path)
    ext = source_path.suffix.lower() or ".wav"
    if ext not in config.AUDIO_EXTENSIONS:
        raise ValueError(f"unsupported audio extension: {ext}")
    size = source_path.stat().st_size
    if size < config.MIN_RECORDING_BYTES:
        raise ValueError(
            f"recording is empty or failed to upload ({size} bytes) — this is a known "
            "browser quirk (especially Safari/macOS's microphone recorder finishing the "
            "upload after the Save button is clicked); wait a moment after the waveform "
            "appears and try saving again"
        )
    d = speaker_dir(speaker, create=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(original_filename or source_path.name).stem)[:60] or "recording"
    dest = d / f"{stamp}_{stem}{ext}"
    counter = 1
    while dest.exists():
        dest = d / f"{stamp}_{stem}-{counter}{ext}"
        counter += 1
    if move:
        shutil.move(str(source_path), dest)
    else:
        shutil.copyfile(source_path, dest)

    meta = {
        "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "original_filename": original_filename or source_path.name,
        "source": source,  # "upload" | "microphone"
        "checksum": checksum_file(dest),
        "note": note or "",
    }
    dest.with_suffix(dest.suffix + ".json").write_text(json.dumps(meta, indent=2))
    return dest


def recording_metadata(speaker: str, filename: str) -> dict[str, Any]:
    """Read a recording's metadata sidecar (uploaded_at, note, etc.)."""
    sidecar = speaker_dir(speaker) / f"{filename}.json"
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def remove_recording(speaker: str, filename: str) -> None:
    d = speaker_dir(speaker)
    target = (d / filename).resolve()
    if d.resolve() not in target.parents:
        raise InvalidSpeakerName("refusing to delete outside speaker directory")
    if target.exists():
        target.unlink()
    sidecar = target.with_suffix(target.suffix + ".json")
    if sidecar.exists():
        sidecar.unlink()


@dataclass
class SyntheticItem:
    id: str
    folder: str
    audio_path: Path
    metadata_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_cache_key(canonical_params: dict[str, Any]) -> str:
    payload = json.dumps(canonical_params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def find_cached(folder_name: str, cache_key: str) -> Optional[SyntheticItem]:
    d = synthetic_dir(folder_name)
    if not d.exists():
        return None
    for meta_path in d.glob("*.json"):
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if meta.get("cache_key") == cache_key:
            audio_path = meta_path.with_suffix(".wav")
            if audio_path.exists():
                return SyntheticItem(
                    id=meta.get("id", meta_path.stem),
                    folder=folder_name,
                    audio_path=audio_path,
                    metadata_path=meta_path,
                    metadata=meta,
                )
    return None


def save_synthetic(folder_name: str, audio_source: Path, metadata: dict[str, Any]) -> SyntheticItem:
    """Persist a newly generated clone plus its reproducibility metadata."""
    d = synthetic_dir(folder_name, create=True)
    item_id = metadata.get("id") or uuid.uuid4().hex[:12]
    metadata["id"] = item_id
    audio_dest = d / f"{item_id}.wav"
    meta_dest = d / f"{item_id}.json"
    shutil.copyfile(audio_source, audio_dest)
    meta_dest.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return SyntheticItem(
        id=item_id, folder=folder_name, audio_path=audio_dest,
        metadata_path=meta_dest, metadata=metadata,
    )


def list_synthetic_folders() -> list[str]:
    if not config.SYNTHETIC_DIR.exists():
        return []
    return sorted(p.name for p in config.SYNTHETIC_DIR.iterdir() if p.is_dir())


def list_synthetic_items(folder_name: str) -> list[SyntheticItem]:
    d = synthetic_dir(folder_name)
    if not d.exists():
        return []
    items = []
    for meta_path in sorted(d.glob("*.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        audio_path = meta_path.with_suffix(".wav")
        if audio_path.exists():
            items.append(SyntheticItem(
                id=meta.get("id", meta_path.stem), folder=folder_name,
                audio_path=audio_path, metadata_path=meta_path, metadata=meta,
            ))
    return items


def get_synthetic_item(folder_name: str, item_id: str) -> Optional[SyntheticItem]:
    safe_id = _SAFE_NAME_RE.sub("", item_id)
    d = synthetic_dir(folder_name)
    meta_path = d / f"{safe_id}.json"
    audio_path = d / f"{safe_id}.wav"
    if not (meta_path.exists() and audio_path.exists()):
        return None
    meta = json.loads(meta_path.read_text())
    return SyntheticItem(id=safe_id, folder=folder_name, audio_path=audio_path,
                          metadata_path=meta_path, metadata=meta)
