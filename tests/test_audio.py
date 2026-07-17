import shutil
import wave

import numpy as np
import pytest

from voicelab import audio


def _write_tone(path, freq_hz, duration_s, sr=16000, amplitude=0.5):
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    data = (amplitude * np.sin(2 * np.pi * freq_hz * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(data.tobytes())
    return path


def test_concat_clips_duration_is_sum_plus_gaps(tmp_path):
    a = _write_tone(tmp_path / "a.wav", 440, 1.0)
    b = _write_tone(tmp_path / "b.wav", 220, 2.0)
    out = tmp_path / "out.wav"

    audio.concat_clips([a, b], out, gap_s=0.5)

    with wave.open(str(out), "rb") as f:
        duration = f.getnframes() / f.getframerate()
    assert duration == pytest.approx(1.0 + 0.5 + 2.0, abs=0.01)


def test_concat_clips_is_sequential_not_mixed(tmp_path):
    a = _write_tone(tmp_path / "a.wav", 440, 0.5, amplitude=0.5)
    b = _write_tone(tmp_path / "b.wav", 220, 0.5, amplitude=0.5)
    out = tmp_path / "out.wav"

    audio.concat_clips([a, b], out, gap_s=0.1)

    data_a, sr = audio.read_wav(a)
    data_out, _ = audio.read_wav(out)
    # The first clip's samples should appear verbatim at the start of the
    # concatenation (sequential placement, not summed with anything).
    assert np.allclose(data_out[: len(data_a)], data_a, atol=1e-3)


def test_concat_clips_requires_at_least_one(tmp_path):
    with pytest.raises(ValueError):
        audio.concat_clips([], tmp_path / "out.wav")


def test_resample_preserves_duration():
    data = np.sin(np.linspace(0, 10, 16000))
    out = audio.resample(data, orig_sr=16000, target_sr=24000)
    assert len(out) == pytest.approx(24000, rel=0.01)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_compress_to_mp3_produces_smaller_playable_file(tmp_path):
    wav_path = _write_tone(tmp_path / "tone.wav", 440, 3.0, sr=24000)
    mp3_path = tmp_path / "tone.mp3"

    audio.compress_to_mp3(wav_path, mp3_path, bitrate="64k")

    assert mp3_path.exists()
    assert mp3_path.stat().st_size > 0
    assert mp3_path.stat().st_size < wav_path.stat().st_size
