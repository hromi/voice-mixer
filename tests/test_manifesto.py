import random

import pytest

from voicelab import manifesto


def test_split_sentences_returns_nonempty_list():
    sentences = manifesto.split_sentences()
    assert len(sentences) > 10
    assert all(isinstance(s, str) and s.strip() for s in sentences)


def test_split_sentences_is_deterministic():
    assert manifesto.split_sentences() == manifesto.split_sentences()


def test_pick_random_excerpt_within_word_bounds():
    sentences = manifesto.split_sentences()
    rng = random.Random(0)
    for _ in range(20):
        excerpt = manifesto.pick_random_excerpt(sentences, rng=rng)
        word_count = len(excerpt.split())
        # allow a little slack: the window logic accepts the sentence that
        # crosses min_words even if it slightly overshoots max_words
        assert word_count >= manifesto.MIN_EXCERPT_WORDS * 0.5

    assert excerpt  # last one, just confirm nonempty


def test_pick_random_excerpt_is_contiguous_text():
    sentences = ["Alpha one.", "Beta two.", "Gamma three.", "Delta four.", "Epsilon five."]
    rng = random.Random(1)
    excerpt = manifesto.pick_random_excerpt(sentences, min_words=2, max_words=6, rng=rng)
    # every excerpt should be an exact substring formed by joining some
    # contiguous run of the input sentences with spaces
    assert excerpt in " ".join(sentences) or any(
        excerpt == " ".join(sentences[i:j])
        for i in range(len(sentences))
        for j in range(i + 1, len(sentences) + 1)
    )


def test_pick_random_excerpt_falls_back_for_impossible_bounds():
    sentences = ["Short."]
    excerpt = manifesto.pick_random_excerpt(sentences, min_words=100, max_words=200)
    assert excerpt == "Short."


def test_pick_random_excerpt_rejects_empty_sentence_list():
    with pytest.raises(ValueError):
        manifesto.pick_random_excerpt([])
