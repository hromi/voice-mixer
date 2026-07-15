import pytest

from voicelab.mixspec import parse_mix_spec


def test_single_speaker():
    entries = parse_mix_spec("SolarPunk0")
    assert len(entries) == 1
    assert entries[0].speaker == "SolarPunk0"
    assert entries[0].weight == 1.0
    assert entries[0].files is None


def test_weighted_two_speakers():
    entries = parse_mix_spec("SolarPunk0:0.6,JaneDoe:0.4")
    assert [e.speaker for e in entries] == ["SolarPunk0", "JaneDoe"]
    assert [e.weight for e in entries] == [0.6, 0.4]
    assert all(e.files is None for e in entries)


def test_pinpoint_single_file():
    entries = parse_mix_spec("SolarPunk0[a.wav]")
    assert entries[0].files == ["a.wav"]
    assert entries[0].weight == 1.0


def test_pinpoint_multiple_files_with_weight():
    entries = parse_mix_spec("SolarPunk0[a.wav|b.wav]:0.6,JaneDoe:0.4")
    assert entries[0].files == ["a.wav", "b.wav"]
    assert entries[0].weight == 0.6
    assert entries[1].files is None


def test_empty_spec_rejected():
    with pytest.raises(ValueError):
        parse_mix_spec("")
    with pytest.raises(ValueError):
        parse_mix_spec("   ")


def test_empty_file_list_rejected():
    with pytest.raises(ValueError):
        parse_mix_spec("SolarPunk0[]")


def test_malformed_entry_rejected():
    with pytest.raises(ValueError):
        parse_mix_spec("SolarPunk0:not-a-number")
    with pytest.raises(ValueError):
        parse_mix_spec("Solar:Punk0[a.wav]:0.5")
