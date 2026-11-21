import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.alignment import (  # noqa: E402
    AlignedUnit,
    AlignmentResult,
    Language,
    pair_units,
    score_utterance,
)


def units(*labels, dur=0.2):
    return [AlignedUnit(l, i * dur, (i + 1) * dur, 0.9) for i, l in enumerate(labels)]


def result(*labels, dur=0.2):
    return AlignmentResult(Language.ENGLISH, " ".join(labels), units(*labels, dur=dur))


def test_skipped_word_does_not_shift_later_pairs():
    assert pair_units(units("i", "like", "tea"), units("i", "really", "like", "tea")) == [0, 2, 3]


def test_inserted_word_is_left_unpaired():
    assert pair_units(units("i", "um", "like", "tea"), units("i", "like", "tea")) == [0, None, 1, 2]


def test_substitution_pairs_with_the_word_it_replaced():
    assert pair_units(units("i", "lake", "tea"), units("i", "like", "tea")) == [0, 1, 2]


def test_substitution_by_a_repeated_word_still_pairs_with_the_replaced_word():
    assert pair_units(units("no", "no"), units("oh", "no")) == [0, 1]


def test_equal_cost_alignment_keeps_exact_matches():
    assert pair_units(units("i", "um", "like", "tea"), units("i", "like", "hot", "tea")) == [0, None, 1, 3]


def test_case_and_punctuation_do_not_break_pairing():
    assert pair_units(units("Hello,", "WORLD."), units("hello", "world")) == [0, 1]


def test_no_reference_units_pairs_nothing():
    assert pair_units(units("a", "b"), []) == [None, None]


def test_score_reports_skipped_reference_units_and_pairs_durations():
    learner = result("i", "like", "tea")
    reference = result("i", "really", "like", "tea")
    scored = score_utterance(learner, reference)
    assert scored.skipped == ["really"]
    # Rate-normalised, every paired unit keeps the reference's rhythm.
    assert [u.duration_ratio for u in scored.units] == [pytest.approx(r) for r in (1.0, 1.0, 1.0)]
    assert [u.label for u in scored.units] == ["i", "like", "tea"]
