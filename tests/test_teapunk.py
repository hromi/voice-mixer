import json
import wave

import pytest

from voicelab import storage, teapunk


def _write_tone(path, sr=16000, seconds=1.0):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(b"\x00\x10" * int(sr * seconds))
    return path


def test_new_guest_slug_is_unique_and_anonymous():
    a = teapunk.new_guest_slug()
    b = teapunk.new_guest_slug()
    assert a != b
    assert a.startswith("Guest")


def test_submit_recording_requires_audio_path():
    with pytest.raises(ValueError, match="record yourself"):
        teapunk.submit_recording(None, "test@example.com")


def test_submit_recording_requires_valid_email(tmp_path):
    clip = _write_tone(tmp_path / "clip.wav")
    with pytest.raises(ValueError, match="valid email"):
        teapunk.submit_recording(str(clip), "not-an-email")


def test_submit_recording_rejects_too_short_recording(tmp_path):
    tiny = tmp_path / "tiny.wav"
    tiny.write_bytes(b"RIFF" + b"\x00" * 10)
    with pytest.raises(ValueError, match="too short or empty"):
        teapunk.submit_recording(str(tiny), "test@example.com")


def test_submit_recording_registers_speaker_and_logs_without_running_ml(tmp_path, monkeypatch, isolated_project_dirs):
    # Don't let the real (GPU-requiring) background job run in a unit test.
    started = []
    monkeypatch.setattr(teapunk, "_background_job", lambda *a, **k: started.append(a))

    clip = _write_tone(tmp_path / "clip.wav", seconds=3.0)
    speaker = teapunk.submit_recording(str(clip), "visitor@example.com")

    assert speaker.startswith("Guest")
    assert len(storage.list_recordings(speaker)) == 1

    assert teapunk.log_path().exists()
    lines = [json.loads(line) for line in teapunk.log_path().read_text().splitlines()]
    submitted = [e for e in lines if e["event"] == "submitted" and e["speaker"] == speaker]
    assert len(submitted) == 1
    # the email must never end up in voicelab's own public storage metadata
    note = storage.recording_metadata(speaker, storage.list_recordings(speaker)[0].name).get("note", "")
    assert "visitor@example.com" not in note
