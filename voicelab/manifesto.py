"""The Solarpunk Manifesto, split into sentences, plus helpers to pick a
random excerpt for the teapunk installation and to generate a full
single-speaker reading (sentence-by-sentence, concatenated).
"""

from __future__ import annotations

import random
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

from . import audio, config, storage

MANIFESTO_PATH = Path(__file__).with_name("data") / "manifesto.txt"

MIN_EXCERPT_WORDS = 35
MAX_EXCERPT_WORDS = 70


def _load_text() -> str:
    return MANIFESTO_PATH.read_text()


def split_sentences(text: Optional[str] = None) -> list[str]:
    text = text if text is not None else _load_text()
    text = (text.replace("“", '"').replace("”", '"')
                 .replace("‘", "'").replace("’", "'")
                 .replace("—", "-"))
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    sentences = []
    for block in blocks:
        block = re.sub(r"\s+", " ", block).strip()
        for part in re.split(r"(?<=[.!?])\s+", block):
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def pick_random_excerpt(
    sentences: Optional[list[str]] = None,
    min_words: int = MIN_EXCERPT_WORDS,
    max_words: int = MAX_EXCERPT_WORDS,
    rng: Optional[random.Random] = None,
) -> str:
    """A random contiguous run of sentences long enough to give a decent
    voice sample when read aloud (roughly 15-30s of speech), short enough
    to not be a chore for a walk-up installation visitor. Falls back to
    the longest available run if the manifesto has nothing that long."""
    sentences = sentences if sentences is not None else split_sentences()
    rng = rng or random
    if not sentences:
        raise ValueError("manifesto has no sentences")

    n = len(sentences)
    candidates = []
    for start in range(n):
        words = 0
        for end in range(start, n):
            words += len(sentences[end].split())
            if words >= min_words:
                if words <= max_words:
                    candidates.append((start, end))
                break

    if not candidates:
        # nothing fits the window; just take the single longest sentence
        longest = max(range(n), key=lambda i: len(sentences[i].split()))
        return sentences[longest]

    start, end = rng.choice(candidates)
    return " ".join(sentences[start:end + 1])


def generate_full_manifesto(
    engine,
    speaker: str,
    *,
    language: str = config.DEFAULT_LANGUAGE,
    backend: Optional[str] = None,
    qwen_clone_method: Optional[str] = None,
    progress_cb=None,
) -> storage.SyntheticItem:
    """Clones the entire manifesto, sentence by sentence, in a single
    speaker's own voice (no blending/chorus — just them), then
    concatenates it into one full reading. `progress_cb(i, n, text)` is
    called after each sentence if given, for status reporting from a
    long-running background job.
    """
    sentences = split_sentences()
    clip_paths = []

    for i, text in enumerate(sentences):
        result = engine.synthesize(
            text, [{"speaker": speaker, "weight": 1.0}],
            language=language, backend=backend, qwen_clone_method=qwen_clone_method,
        )
        clip_paths.append(result.item.audio_path)
        if progress_cb:
            progress_cb(i + 1, len(sentences), text)

    with tempfile.TemporaryDirectory(prefix="voicelab-manifesto-") as tmp_str:
        full_wav = Path(tmp_str) / "manifesto_full.wav"
        audio.concat_clips(clip_paths, full_wav)

        metadata = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "mode": "manifesto",
            "speaker": speaker,
            "language": language,
            "sentence_count": len(sentences),
        }
        item = storage.save_synthetic(speaker, full_wav, metadata)

    return item
