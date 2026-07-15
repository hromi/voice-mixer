"""Shared parser for the speaker-mix syntax used by the CLI, the HTTP API's
convenience query form, and the web UI:

    SolarPunk0
    SolarPunk0:0.6,JaneDoe:0.4
    SolarPunk0[20260715T141259_audio.wav]
    SolarPunk0[fileA.wav|fileB.wav]:0.6,JaneDoe:0.4

The optional `[file1|file2]` suffix pins the embedding to specific
recordings instead of averaging every recording the speaker has — useful
for reproducing a clone from an exact source file, or for a communal mix
where only some of a speaker's recordings should contribute.
"""

from __future__ import annotations

import re

from .embeddings import MixEntry

_ENTRY_RE = re.compile(
    r"^(?P<name>[^\[\]:]+)"
    r"(\[(?P<files>[^\[\]]*)\])?"
    r"(:(?P<weight>-?[0-9.]+))?$"
)


def parse_mix_spec(spec: str) -> list[MixEntry]:
    """Parse a speaker-mix string into MixEntry objects.

    A bare speaker name (no ":weight") defaults to weight 1.0; weights are
    normalized relative to each other, so "A,B" and "A:1,B:1" are
    equivalent. A bare name (no "[files]") uses all of that speaker's
    recordings, averaged.
    """
    entries: list[MixEntry] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue

        match = _ENTRY_RE.match(chunk)
        if not match:
            raise ValueError(f"invalid mix entry {chunk!r}")

        name = match.group("name").strip()
        if not name:
            raise ValueError(f"invalid mix entry {chunk!r}")

        files = None
        if match.group("files") is not None:
            files = [f.strip() for f in match.group("files").split("|") if f.strip()]
            if not files:
                raise ValueError(f"empty file list in mix entry {chunk!r}")

        weight_str = match.group("weight")
        try:
            weight = float(weight_str) if weight_str is not None else 1.0
        except ValueError:
            raise ValueError(f"invalid weight in mix entry {chunk!r}")

        entries.append(MixEntry(speaker=name, weight=weight, files=files))

    if not entries:
        raise ValueError("empty speaker mix")
    return entries
