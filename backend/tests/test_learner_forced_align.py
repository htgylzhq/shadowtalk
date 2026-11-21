import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.alignment import (  # noqa: E402
    AlignedUnit,
    AlignmentResult,
    Language,
    force_to_reference,
    score_utterance,
)
from app.services.gop import CTCPosteriors  # noqa: E402

# 0 blank, 1 t, 2 iː, 3 aɪ, 4 l, 5 k, 6 ɡ, 7 ɹ, 8 n
V = 9
LABELS = {1: "t", 2: "iː", 3: "aɪ", 4: "l", 5: "k", 6: "ɡ", 7: "ɹ", 8: "n"}


def frames(*best, p=0.9):
    rows = np.full((len(best), V), (1 - p) / (V - 1))
    rows[np.arange(len(best)), best] = p
    return np.log(rows)


def lesson(*words):
    """Reference units from lesson text: (word, [token ids])."""
    units, t = [], 0.0
    for word, ids in words:
        for i in ids:
            units.append(AlignedUnit(LABELS[i], t, t + 0.1, 0.9, token_ids=[i], word=word))
            t += 0.1
    return AlignmentResult(Language.ENGLISH, " ".join(w for w, _ in words), units)


def learner(log_probs):
    return AlignmentResult(
        Language.ENGLISH, "", [], posteriors=CTCPosteriors(log_probs, 0.02, 0)
    )


GREEN_TEA = (("green", [6, 7, 2, 8]), ("tea", [1, 2]))


def test_learner_units_are_the_lesson_phonemes_on_learner_frames():
    # Learner says "ɹ iː n t aɪ": ɡ is not decoded, the tea vowel is wrong.
    lp = frames(0, 7, 0, 2, 2, 0, 8, 0, 1, 0, 3, 3, 0)
    forced, pairs = force_to_reference(learner(lp), lesson(*GREEN_TEA))
    assert [(u.label, u.word) for u in forced.units] == [
        ("ɡ", "green"), ("ɹ", "green"), ("iː", "green"), ("n", "green"), ("t", "tea"), ("iː", "tea"),
    ]
    assert pairs == [0, 1, 2, 3, 4, 5]


def test_undecoded_phoneme_inside_a_heard_word_is_scored_not_skipped():
    lp = frames(0, 7, 0, 2, 2, 0, 8, 0, 1, 0, 3, 3, 0)
    result = score_utterance(learner(lp), lesson(*GREEN_TEA))
    assert result.skipped == []
    by = [(u.label, u.word, u.gop) for u in result.units]
    assert by[-1][0] == "iː" and by[-1][2] is not None and by[-1][2] < 0.2  # said aɪ
    assert by[-2][2] == pytest.approx(1.0, abs=0.05)  # t is right


def test_word_the_learner_skipped_is_reported_skipped():
    # Learner says only "t iː": "green" gets nothing but blank frames.
    lp = frames(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 2, 0)
    result = score_utterance(learner(lp), lesson(*GREEN_TEA))
    assert result.skipped == ["ɡ", "ɹ", "iː", "n"]
    assert [(u.label, u.word) for u in result.units] == [("t", "tea"), ("iː", "tea")]
    assert all(u.gop == pytest.approx(1.0, abs=0.05) for u in result.units)


def test_reference_without_lesson_words_keeps_decoded_pairing():
    ref = lesson(*GREEN_TEA)
    for u in ref.units:
        u.word = None
    assert force_to_reference(learner(frames(0, 1, 0)), ref) is None


def test_learner_without_posteriors_keeps_decoded_pairing():
    no_post = AlignmentResult(Language.ENGLISH, "", [AlignedUnit("t", 0, 0.1, 0.9)])
    assert force_to_reference(no_post, lesson(*GREEN_TEA)) is None


def test_silent_learner_skips_every_word():
    result = score_utterance(learner(frames(0, 0, 0, 0, 0, 0)), lesson(*GREEN_TEA))
    assert result.units == []
    assert result.skipped == ["ɡ", "ɹ", "iː", "n", "t", "iː"]


def test_unrelated_speech_is_skipped_or_scored_low():
    # Learner says "like" to the lesson "green tea".
    result = score_utterance(learner(frames(0, 4, 0, 3, 3, 0, 5, 0)), lesson(*GREEN_TEA))
    assert len(result.units) + len(result.skipped) == 6
    assert all(u.gop is None or u.gop < 0.2 for u in result.units)


def test_word_skipped_mid_sentence_does_not_steal_the_next_words_audio():
    # Lesson "like green tea", learner says "like tea".
    lp = frames(0, 4, 0, 3, 3, 0, 5, 0, 0, 1, 0, 2, 2, 0)
    result = score_utterance(learner(lp), lesson(("like", [4, 3, 5]), *GREEN_TEA))
    assert result.skipped == ["ɡ", "ɹ", "iː", "n"]
    assert [u.word for u in result.units] == ["like"] * 3 + ["tea"] * 2
    assert all(u.gop == pytest.approx(1.0, abs=0.05) for u in result.units)


def test_mispronounced_word_is_scored_not_skipped():
    # Lesson "like tea", learner says "like tie": tea is heard, just wrong.
    lp = frames(0, 4, 0, 3, 3, 0, 5, 0, 1, 0, 3, 3, 0)
    result = score_utterance(learner(lp), lesson(("like", [4, 3, 5]), ("tea", [1, 2])))
    assert result.skipped == []
    assert result.units[-1].label == "iː" and result.units[-1].gop < 0.2


def test_skipped_opening_word_with_no_leading_silence():
    # Learner says "tea" straight away, no blank frame before it.
    result = score_utterance(learner(frames(1, 1, 0, 2, 2, 0)), lesson(*GREEN_TEA))
    assert result.skipped == ["ɡ", "ɹ", "iː", "n"]
    assert [u.label for u in result.units] == ["t", "iː"]
    assert all(u.gop == pytest.approx(1.0, abs=0.05) for u in result.units)


def test_skipped_closing_word_with_no_trailing_silence():
    lp = frames(0, 1, 0, 2, 2)
    result = score_utterance(learner(lp), lesson(("tea", [1, 2]), ("like", [4, 3, 5])))
    assert result.skipped == ["l", "aɪ", "k"]
    assert all(u.gop == pytest.approx(1.0, abs=0.05) for u in result.units)


def test_two_words_skipped_back_to_back():
    # Lesson "like green tea", learner says only "like".
    lp = frames(0, 4, 0, 3, 0, 5, 0, 0)
    result = score_utterance(learner(lp), lesson(("like", [4, 3, 5]), *GREEN_TEA))
    assert result.skipped == ["ɡ", "ɹ", "iː", "n", "t", "iː"]
    assert [u.word for u in result.units] == ["like"] * 3
