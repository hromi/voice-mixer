import pytest


def test_sanitize_speaker_name_blocks_traversal(isolated_project_dirs):
    _, storage, _ = isolated_project_dirs
    assert storage.sanitize_speaker_name("SolarPunk0") == "SolarPunk0"
    # Path separators and dots are stripped entirely, so no traversal
    # sequence can ever reach the filesystem call — it degrades to a plain
    # (safe) name rather than needing to be rejected.
    assert storage.sanitize_speaker_name("../../etc") == "etc"
    with pytest.raises(storage.InvalidSpeakerName):
        storage.sanitize_speaker_name("..")
    with pytest.raises(storage.InvalidSpeakerName):
        storage.sanitize_speaker_name("../..")
    with pytest.raises(storage.InvalidSpeakerName):
        storage.sanitize_speaker_name("///")


def test_add_recording_and_list(isolated_project_dirs, tmp_path, make_wav):
    _, storage, _ = isolated_project_dirs
    src = make_wav(tmp_path / "src" / "sample.wav")

    dest = storage.add_recording("SolarPunk0", src, original_filename="sample.wav")

    assert dest.exists()
    assert dest.parent.name == "SolarPunk0"
    assert storage.list_speakers() == ["SolarPunk0"]
    assert storage.list_recordings("SolarPunk0") == [dest]

    sidecar = dest.with_suffix(dest.suffix + ".json")
    assert sidecar.exists()


def test_add_recording_rejects_unsupported_extension(isolated_project_dirs, tmp_path):
    _, storage, _ = isolated_project_dirs
    bogus = tmp_path / "notes.txt"
    bogus.write_text("hello")
    with pytest.raises(ValueError):
        storage.add_recording("SolarPunk0", bogus)


def test_mix_folder_name_is_sorted_and_deduped(isolated_project_dirs):
    _, storage, _ = isolated_project_dirs
    assert storage.mix_folder_name(["JaneDoe", "SolarPunk0"]) == "JaneDoe+SolarPunk0"
    assert storage.mix_folder_name(["SolarPunk0", "JaneDoe"]) == "JaneDoe+SolarPunk0"
    assert storage.mix_folder_name(["SolarPunk0", "SolarPunk0"]) == "SolarPunk0"


def test_synthetic_cache_round_trip(isolated_project_dirs, tmp_path, make_wav):
    _, storage, _ = isolated_project_dirs
    audio = make_wav(tmp_path / "clone.wav")

    params = {"text": "hello", "speaker": "SolarPunk0"}
    cache_key = storage.compute_cache_key(params)
    assert storage.find_cached("SolarPunk0", cache_key) is None

    item = storage.save_synthetic("SolarPunk0", audio, {"text": "hello", "cache_key": cache_key})
    assert item.audio_path.exists()
    assert item.metadata["cache_key"] == cache_key

    hit = storage.find_cached("SolarPunk0", cache_key)
    assert hit is not None
    assert hit.id == item.id

    listed = storage.list_synthetic_items("SolarPunk0")
    assert [i.id for i in listed] == [item.id]

    fetched = storage.get_synthetic_item("SolarPunk0", item.id)
    assert fetched is not None
    assert fetched.metadata["text"] == "hello"


def test_compute_cache_key_is_order_independent_for_dict_keys(isolated_project_dirs):
    _, storage, _ = isolated_project_dirs
    a = storage.compute_cache_key({"x": 1, "y": 2})
    b = storage.compute_cache_key({"y": 2, "x": 1})
    assert a == b

    c = storage.compute_cache_key({"x": 1, "y": 3})
    assert a != c
