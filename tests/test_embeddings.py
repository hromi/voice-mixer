import numpy as np
import pytest

from voicelab import embeddings


def test_blend_embeddings_weighted_average():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])

    blended = embeddings.blend_embeddings([(a, 0.5), (b, 0.5)])
    np.testing.assert_allclose(blended, [0.5, 0.5])

    blended_unnormalized = embeddings.blend_embeddings([(a, 3.0), (b, 1.0)])
    np.testing.assert_allclose(blended_unnormalized, [0.75, 0.25])


def test_blend_embeddings_single_entry_is_identity():
    a = np.array([2.0, 4.0])
    np.testing.assert_allclose(embeddings.blend_embeddings([(a, 1.0)]), a)


def test_blend_embeddings_rejects_empty_or_zero_weight():
    with pytest.raises(ValueError):
        embeddings.blend_embeddings([])
    with pytest.raises(ValueError):
        embeddings.blend_embeddings([(np.array([1.0]), 0.0)])


torch = pytest.importorskip("torch", reason="torch not installed; skipping model-shaped embedding tests")


class FakeConverter:
    """Stands in for openvoice.api.ToneColorConverter: deterministic
    per-file embeddings derived from the file path, so blend math is
    verifiable without the real model or checkpoints."""

    device = "cpu"

    def extract_se(self, ref_wav_list, se_save_path=None):
        path = ref_wav_list[0]
        seed = abs(hash(path)) % 1000
        return torch.full((4, 1), float(seed))


def test_speaker_embedding_averages_files(isolated_project_dirs, tmp_path, make_wav):
    _, storage, _ = isolated_project_dirs
    f1 = make_wav(tmp_path / "a" / "one.wav")
    f2 = make_wav(tmp_path / "a" / "two.wav")
    storage.add_recording("SolarPunk0", f1, original_filename="one.wav")
    storage.add_recording("SolarPunk0", f2, original_filename="two.wav")

    converter = FakeConverter()
    se, used_files = embeddings.speaker_embedding(converter, "SolarPunk0")
    assert len(used_files) == 2

    per_file = [embeddings.get_file_embedding(converter, "SolarPunk0", p)
                for p in storage.list_recordings("SolarPunk0")]
    expected = (per_file[0] + per_file[1]) / 2
    torch.testing.assert_close(se, expected)


def test_speaker_embedding_requires_recordings(isolated_project_dirs):
    _, storage, _ = isolated_project_dirs
    with pytest.raises(ValueError):
        embeddings.speaker_embedding(FakeConverter(), "NoOneHome")


def test_communal_embedding_blends_two_speakers(isolated_project_dirs, tmp_path, make_wav):
    _, storage, _ = isolated_project_dirs
    storage.add_recording("SolarPunk0", make_wav(tmp_path / "j.wav"), original_filename="j.wav")
    storage.add_recording("JaneDoe", make_wav(tmp_path / "d.wav"), original_filename="d.wav")

    converter = FakeConverter()
    mix = [
        embeddings.MixEntry(speaker="SolarPunk0", weight=0.5),
        embeddings.MixEntry(speaker="JaneDoe", weight=0.5),
    ]
    blended, detail = embeddings.communal_embedding(converter, mix)

    assert {d["speaker"] for d in detail} == {"SolarPunk0", "JaneDoe"}
    for d in detail:
        assert d["normalized_weight"] == pytest.approx(0.5)

    john_se, _ = embeddings.speaker_embedding(converter, "SolarPunk0")
    jane_se, _ = embeddings.speaker_embedding(converter, "JaneDoe")
    torch.testing.assert_close(blended, (john_se + jane_se) / 2)
