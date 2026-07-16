import wave

import numpy as np
import pytest

from voicelab import chorus


def _write_tone(path, freq_hz, duration_s, sr=16000, amplitude=0.5):
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    data = (amplitude * np.sin(2 * np.pi * freq_hz * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(data.tobytes())
    return path


def test_mix_chorus_requires_at_least_two_clips(tmp_path):
    a = _write_tone(tmp_path / "a.wav", 440, 0.2)
    with pytest.raises(ValueError):
        chorus.mix_chorus([(a, 1.0)], tmp_path / "out.wav")


def test_mix_chorus_output_duration_matches_longest_clip(tmp_path):
    short = _write_tone(tmp_path / "short.wav", 440, 0.2)
    long = _write_tone(tmp_path / "long.wav", 220, 0.5)
    out = tmp_path / "out.wav"

    chorus.mix_chorus([(short, 1.0), (long, 1.0)], out)

    with wave.open(str(out), "rb") as f:
        duration = f.getnframes() / f.getframerate()
    assert duration == pytest.approx(0.5, abs=0.01)


def test_mix_chorus_is_a_genuine_sum_not_either_input(tmp_path):
    a = _write_tone(tmp_path / "a.wav", 440, 0.2, amplitude=0.3)
    b = _write_tone(tmp_path / "b.wav", 440, 0.2, amplitude=0.3)
    out = tmp_path / "out.wav"

    chorus.mix_chorus([(a, 1.0), (b, 1.0)], out)

    data_a, sr_a = chorus._read_wav(a)
    data_out, sr_out = chorus._read_wav(out)
    assert sr_out == sr_a
    # Two identical in-phase 0.3-amplitude tones should sum to ~0.6 peak
    # (below the 0.99 clipping threshold, so untouched by normalization)
    # — a genuine sum, not equal to either single input's own amplitude.
    assert np.max(np.abs(data_out)) == pytest.approx(0.6, abs=0.02)
    assert not np.allclose(data_out[: len(data_a)], data_a, atol=1e-3)


def test_mix_chorus_never_clips(tmp_path):
    clips = [_write_tone(tmp_path / f"{i}.wav", 300 + i * 50, 0.2, amplitude=0.9) for i in range(4)]
    out = tmp_path / "out.wav"

    chorus.mix_chorus([(c, 1.0) for c in clips], out)

    data_out, _ = chorus._read_wav(out)
    assert np.max(np.abs(data_out)) <= 1.0


def test_mix_chorus_respects_relative_gain(tmp_path):
    loud = _write_tone(tmp_path / "loud.wav", 440, 0.2, amplitude=0.1)
    quiet = _write_tone(tmp_path / "quiet.wav", 220, 0.2, amplitude=0.1)
    out_equal = tmp_path / "equal.wav"
    out_weighted = tmp_path / "weighted.wav"

    chorus.mix_chorus([(loud, 1.0), (quiet, 1.0)], out_equal)
    chorus.mix_chorus([(loud, 10.0), (quiet, 1.0)], out_weighted)

    data_equal, _ = chorus._read_wav(out_equal)
    data_weighted, _ = chorus._read_wav(out_weighted)
    # Both get peak-normalized, but the *shape* differs: the heavily
    # weighted version should correlate more strongly with the loud tone
    # alone than the equal-weight version does.
    data_loud, _ = chorus._read_wav(loud)
    n = min(len(data_loud), len(data_equal), len(data_weighted))
    corr_equal = np.corrcoef(data_equal[:n], data_loud[:n])[0, 1]
    corr_weighted = np.corrcoef(data_weighted[:n], data_loud[:n])[0, 1]
    assert corr_weighted > corr_equal
