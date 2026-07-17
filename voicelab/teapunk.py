"""Orchestrates the teapunk installation's walk-up flow: a visitor reads a
random manifesto excerpt aloud, enters their email, and — in the
background, since it takes many minutes — we clone the full manifesto in
their voice, compress it, and email it to them.

Privacy note: the visitor's email address is *never* written into
voicelab's own public storage (voices/synthetic, browsable via the admin
Library tab) — only into a private, non-web-served local log
(.cache/teapunk_log.jsonl) kept for operational purposes (e.g. resending
after a bounce). The speaker folder created for their recording uses an
anonymous generated id, not their name or email.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

from . import audio, config, mailer, manifesto, storage

MIN_RECORDING_BYTES = 20_000  # a few real seconds of audio; catches an empty/failed browser recording


def log_path() -> Path:
    # Computed fresh each call (not a module-level constant) so it always
    # reflects the current config.CACHE_DIR — matters for test isolation
    # (VOICELAB_CACHE_DIR env override) and for VOICELAB_CACHE_DIR changing
    # after this module is first imported.
    return config.CACHE_DIR / "teapunk_log.jsonl"


def new_guest_slug() -> str:
    return f"Guest{time.strftime('%Y%m%d%H%M%S')}{secrets.token_hex(3)}"


def _log(event: dict) -> None:
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **event}
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _background_job(speaker: str, email: str, language: str) -> None:
    from .engine import get_shared_engine  # deferred: heavy ML import

    try:
        engine = get_shared_engine()

        def progress(i: int, n: int, text: str) -> None:
            if i == 1 or i == n or i % 10 == 0:
                _log({"event": "progress", "speaker": speaker, "sentence": i, "total": n})

        item = manifesto.generate_full_manifesto(engine, speaker, language=language, progress_cb=progress)

        mp3_path = item.audio_path.with_suffix(".mp3")
        audio.compress_to_mp3(item.audio_path, mp3_path)

        mailer.send_manifesto_email(email, mp3_path)
        _log({"event": "sent", "speaker": speaker, "email": email})
    except Exception as exc:  # noqa: BLE001 - a background thread's exceptions vanish silently otherwise
        _log({
            "event": "error", "speaker": speaker, "email": email,
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
        })


def submit_recording(
    audio_path: Optional[str], email: str, language: str = config.DEFAULT_LANGUAGE,
    on_started: Optional[Callable[[str], None]] = None,
) -> str:
    """Registers the visitor's recording under a fresh anonymous speaker
    id and kicks off background generation + email delivery on a daemon
    thread. Returns the guest slug immediately — the actual ~15-20 minutes
    of work happens asynchronously, so the kiosk UI doesn't block on it.
    """
    if not audio_path:
        raise ValueError("Please record yourself reading the excerpt first.")
    if not mailer.is_valid_email(email):
        raise ValueError("Please enter a valid email address.")

    size = Path(audio_path).stat().st_size
    if size < MIN_RECORDING_BYTES:
        raise ValueError(
            "That recording looks too short or empty — please try recording again, "
            "reading the full excerpt aloud."
        )

    speaker = new_guest_slug()
    storage.add_recording(speaker, audio_path, source="teapunk", note="Teapunk installation recording")
    _log({"event": "submitted", "speaker": speaker, "email": email})

    if on_started:
        on_started(speaker)

    threading.Thread(target=_background_job, args=(speaker, email, language), daemon=True).start()
    return speaker
