import wave
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_project_dirs(tmp_path, monkeypatch):
    """Point voicelab's config at a scratch directory for every test so
    tests never touch the real ./voices or ./synthetic folders."""
    monkeypatch.setenv("VOICELAB_VOICES_DIR", str(tmp_path / "voices"))
    monkeypatch.setenv("VOICELAB_SYNTHETIC_DIR", str(tmp_path / "synthetic"))
    monkeypatch.setenv("VOICELAB_CHECKPOINTS_DIR", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("VOICELAB_CACHE_DIR", str(tmp_path / ".cache"))

    import importlib

    from voicelab import config
    importlib.reload(config)

    from voicelab import storage
    importlib.reload(storage)

    from voicelab import embeddings
    importlib.reload(embeddings)

    yield config, storage, embeddings


def _write_wav(path: Path, seconds: float = 0.2, framerate: int = 16000) -> Path:
    """Write a tiny silent WAV file — enough to exercise storage/checksum
    logic without needing real audio or any ML dependency."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = int(seconds * framerate)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(framerate)
        f.writeframes(b"\x00\x00" * n_frames)
    return path


@pytest.fixture
def make_wav():
    return _write_wav
