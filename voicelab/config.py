"""Central path and constant configuration for voicelab.

All paths default to living next to the repository root (the parent of this
package) so that `./voices` and `./synthetic` match the layout described in
the project README. Every path can be overridden with an environment
variable, which is useful when running the API/web server from elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_path(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


VOICES_DIR = _env_path("VOICELAB_VOICES_DIR", PROJECT_ROOT / "voices")
SYNTHETIC_DIR = _env_path("VOICELAB_SYNTHETIC_DIR", PROJECT_ROOT / "synthetic")
CHECKPOINTS_DIR = _env_path("VOICELAB_CHECKPOINTS_DIR", PROJECT_ROOT / "checkpoints")
CACHE_DIR = _env_path("VOICELAB_CACHE_DIR", PROJECT_ROOT / ".cache")
EMBEDDING_CACHE_DIR = CACHE_DIR / "embeddings"

# OpenVoice V2 checkpoint layout (see scripts/download_checkpoints.sh).
CONVERTER_DIR = CHECKPOINTS_DIR / "converter"
BASE_SPEAKER_SE_DIR = CHECKPOINTS_DIR / "base_speakers" / "ses"

DEFAULT_LANGUAGE = "EN"
DEFAULT_SPEED = 1.0
DEFAULT_TAU = 0.3
DEFAULT_WATERMARK_MESSAGE = "@voicelab"

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}

# Gradio's built-in login wall for the web app (voicelab serve web).
# No password ships in this (public) repo — VOICELAB_WEB_PASSWORD must be
# set in the environment the server actually runs in. Read lazily (via
# get_web_auth_password(), only called at server-start time) rather than
# as a module-level constant, so importing voicelab.config — e.g. for
# `voicelab --help` or the CLI/API, which don't need the web app's
# credentials at all — doesn't require it to be set.
WEB_AUTH_USERNAME = os.environ.get("VOICELAB_WEB_USERNAME", "tts")


def get_web_auth_password() -> str:
    password = os.environ.get("VOICELAB_WEB_PASSWORD")
    if not password:
        raise RuntimeError(
            "VOICELAB_WEB_PASSWORD is not set. Set it before running the web app, e.g.:\n"
            "  export VOICELAB_WEB_PASSWORD='...'\n"
            "  voicelab serve web"
        )
    return password

# MeloTTS (the default base-speaker engine) doesn't cover every language
# OpenVoice's tone-color conversion could otherwise work with. For
# languages in this map, Qwen3-TTS generates the base utterance instead —
# same downstream tone-color-conversion step, just a different source of
# base audio. Values are the language names Qwen3-TTS itself expects
# (validated case-insensitively against the loaded checkpoint at runtime).
#
# QWEN_ONLY_LANGUAGES (a subset) is where MeloTTS has *no* voice at all —
# used to pick a backend automatically when the caller doesn't force one.
# QWEN_LANGUAGES is everything Qwen3-TTS supports, including languages
# MeloTTS also covers, for when a backend is explicitly requested.
QWEN_LANGUAGES = {
    "EN": "English", "ES": "Spanish", "FR": "French", "ZH": "Chinese",
    "JP": "Japanese", "KR": "Korean", "EN_NEWEST": "English",
    "DE": "German", "RU": "Russian", "PT": "Portuguese", "IT": "Italian",
}
QWEN_ONLY_LANGUAGES = {
    "DE": "German",
    "RU": "Russian",
    "PT": "Portuguese",
    "IT": "Italian",
}
MELO_LANGUAGES = {"EN", "EN_NEWEST", "ES", "FR", "ZH", "JP", "KR"}
QWEN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
QWEN_DEFAULT_SPEAKER = "Ryan"

# Qwen3-TTS's "Base" checkpoint (distinct from -CustomVoice above) supports
# native speaker-embedding voice cloning: extract an x-vector from
# reference audio via model.extract_speaker_embedding(...) and condition
# generation on it directly — no separate OpenVoice tone-color-conversion
# pass needed. This is the "native" qwen_clone_method; "openvoice" (the
# default) is the original preset-voice-then-convert pipeline, which is
# the only option for MeloTTS since it has no native cloning of its own.
QWEN_BASE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

# MeloTTS's base-speaker keys per language (its model.hps.data.spk2id),
# queried directly from the loaded checkpoints — most languages have
# exactly one voice, English has several accents plus a "Newest" model.
MELO_BASE_SPEAKERS = {
    "EN": ["EN-Default", "EN-US", "EN-BR", "EN-AU", "EN_INDIA"],
    "EN_NEWEST": ["EN-Newest"],
    "ES": ["ES"],
    "FR": ["FR"],
    "ZH": ["ZH"],
    "JP": ["JP"],
    "KR": ["KR"],
}

# Qwen3-TTS's 9 built-in preset voices (Qwen3-TTS-12Hz-*-CustomVoice),
# same list for every language it supports.
QWEN_SPEAKERS = [
    "Ryan", "Aiden", "Vivian", "Serena", "Uncle_Fu",
    "Dylan", "Eric", "Ono_Anna", "Sohee",
]

# qwen-tts uses `X | None` union-type syntax at module level, which needs
# Python 3.10+ — incompatible with this project's main (3.9) interpreter.
# It runs as a subprocess in its own venv instead; see qwen_worker.py and
# scripts/setup_qwen_venv.sh.
QWEN_VENV_DIR = _env_path("VOICELAB_QWEN_VENV_DIR", PROJECT_ROOT / ".venv-qwen")
QWEN_VENV_PYTHON = QWEN_VENV_DIR / "bin" / "python3"

for _dir in (VOICES_DIR, SYNTHETIC_DIR, CHECKPOINTS_DIR, EMBEDDING_CACHE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
