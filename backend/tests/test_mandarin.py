import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import g2p, mandarin  # noqa: E402
from app.services.alignment import (  # noqa: E402
    AlignedUnit,
    AlignmentResult,
    Language,
    Wav2Vec2PhonemeAligner,
    force_to_reference,
    score_utterance,
)
from app.services.gop import CTCPosteriors  # noqa: E402
from app.services.pitch import PitchContour  # noqa: E402

# A slice of the phoneme head's vocabulary: Mandarin finals are whole
# toned tokens (espeak cmn IPA: ɜ = rising, 2 = low, 5 = high level or falling).
VOCAB = {
    "<pad>": 0, "n": 1, "iɜ": 2, "x": 3, "ɑu2": 4, "ɕ": 5, "iɛ5": 6, "i2": 7,
    "m": 8, "ɑ5": 9, "t": 10, "iɛ2": 11, "ərɜ": 12, "k": 13, "o2": 14,
}
ESPEAK = {
    "你好": ("nˈiɜ xˈɑu2", "n'i35_| x'Au21_|"),
    "谢谢你": ("ɕˈiɛ5 ɕˈiɛ5 nˈi2", "S;'iE51_| S;'iE51_| n'i214_|"),
    "妈骂": ("mˈɑ5 mˈɑ5", "m'A55_| m'A51_|"),
    "点儿": ("tˈiɛ2n ˈərɜ", "t'iE21n_| '@r35_|"),
    "个": ("ko-2", "ko-22_|"),
    "你好吗": ("nˈiɜ xˈɑu2", "n'i35_| x'Au21_| mA11_|"),  # IPA lost a syllable
}


def fake_espeak(text):
    return ESPEAK[text]


def units(text):
    return [
        (u.word, u.word_index, u.label, u.token_ids, u.tone)
        for u in mandarin.lesson_units(text, VOCAB, fake_espeak)
    ]


def frames(*best, p=0.9):
    rows = np.full((len(best), len(VOCAB)), (1 - p) / (len(VOCAB) - 1))
    rows[np.arange(len(best)), best] = p
    return np.log(rows)


def test_each_syllable_is_an_initial_and_a_toned_final():
    assert units("你好") == [
        ("你", 0, "n", [1], None),
        ("你", 0, "iɜ", [2], 2),
        ("好", 1, "x", [3], None),
        ("好", 1, "ɑu2", [4], 3),
    ]


def test_tone_comes_from_the_mnemonics_not_the_lossy_ipa_mark():
    # 妈 (tone 1) and 骂 (tone 4) are both "ɑ5" in espeak's IPA.
    assert [u[4] for u in units("妈骂")] == [None, 1, None, 4]


def test_third_tone_sandhi_from_espeak_is_kept():
    # 你好: espeak already applies 3-3 sandhi, so 你 is scored as tone 2.
    assert [u[4] for u in units("你好") if u[4]] == [2, 3]


def test_repeated_characters_are_separate_words():
    assert [(u[0], u[1]) for u in units("谢谢你")][::2] == [("谢", 0), ("谢", 1), ("你", 2)]


def test_coda_stays_in_the_final_and_zero_initial_syllables_have_no_initial():
    assert [(u[2], u[3], u[4]) for u in units("点儿")] == [
        ("t", [10], None),
        ("iɛ2n", [11, 1], 3),
        ("ərɜ", [12], 2),
    ]


def test_neutral_tone_marker_is_tokenized():
    assert [(u[2], u[4]) for u in units("个")] == [("k", None), ("o2", 5)]


def test_syllable_count_mismatch_keeps_the_chunk_as_one_word_without_tones():
    assert {(u[0], u[1]) for u in units("你好吗")} == {("你好吗", 0)}
    assert all(u[4] is None for u in units("你好吗"))


@pytest.mark.parametrize(
    "syllable, tone",
    [("m'A55_|", 1), ("m'A35", 2), ("m'A21", 3), ("n'i214", 3), ("m'A51", 4), ("mA11", 5), ("ko-22", 5)],
)
def test_tone_from_chao_digits(syllable, tone):
    assert mandarin.tone_from_mnemonic(syllable) == tone


def test_lesson_units_routes_mandarin_and_keeps_one_token_units_elsewhere(monkeypatch):
    monkeypatch.setattr(mandarin, "espeak_cmn", fake_espeak)
    assert [u.label for u in g2p.lesson_units("你好", Language.MANDARIN, VOCAB)] == [
        "n", "iɜ", "x", "ɑu2",
    ]
    en = g2p.lesson_units(
        "tea tea", Language.ENGLISH, {"<pad>": 0, "t": 1, "iː": 2}, ipa=lambda w, lang: "tˈiː"
    )
    assert [(u.word, u.word_index, u.label, u.token_ids, u.tone) for u in en] == [
        ("tea", 0, "t", [1], None),
        ("tea", 0, "iː", [2], None),
        ("tea", 1, "t", [1], None),
        ("tea", 1, "iː", [2], None),
    ]


def test_decoded_tokens_group_into_initials_and_finals():
    labels = ["n", "iɜ", "x", "ɑu2", "t", "iɛ2", "n", "n", "i2", "ərɜ", "f", "k"]
    decoded = [AlignedUnit(lab, i * 0.1, i * 0.1 + 0.1, 0.8, token_ids=[i]) for i, lab in enumerate(labels)]
    grouped = mandarin.group_initial_final(decoded)
    assert [u.label for u in grouped] == ["n", "iɜ", "x", "ɑu2", "t", "iɛ2n", "n", "i2", "ərɜ", "f", "k"]
    final = grouped[5]
    assert (final.start_s, final.end_s, final.token_ids) == (0.5, pytest.approx(0.7), [5, 6])
    assert all(u.tone is None for u in grouped)


def test_decoded_coda_before_zero_initial_er_or_a_toned_final_stays_in_the_final():
    def labels(*tokens):
        decoded = [AlignedUnit(t, i * 0.1, i * 0.1 + 0.1, 0.8, token_ids=[i]) for i, t in enumerate(tokens)]
        return [u.label for u in mandarin.group_initial_final(decoded)]

    assert labels("t", "iɛ2", "n", "ərɜ") == ["t", "iɛ2n", "ərɜ"]
    assert labels("t", "ɑ5", "ŋ", "ɑ5") == ["t", "ɑ5ŋ", "ɑ5"]


def test_align_text_spans_a_multi_token_final(monkeypatch):
    aligner = Wav2Vec2PhonemeAligner()
    log_probs = frames(0, 10, 11, 11, 1, 0, 12, 12, 0)
    monkeypatch.setattr(aligner, "_posteriors", lambda samples: (log_probs, 0, VOCAB))
    monkeypatch.setattr(mandarin, "espeak_cmn", fake_espeak)
    result = aligner.align_text(b"\x00\x00" * 800, Language.MANDARIN, "点儿")
    assert [(u.label, u.token_ids, u.word, u.word_index, u.tone) for u in result.units] == [
        ("t", [10], "点", 0, None),
        ("iɛ2n", [11, 1], "点", 0, 3),
        ("ərɜ", [12], "儿", 1, 2),
    ]
    final = result.units[1]
    assert final.start_s == pytest.approx(0.04)
    assert final.end_s == pytest.approx(result.units[2].start_s)
    assert final.ctc_confidence > 0.5


def test_decode_fallback_groups_mandarin_units(monkeypatch):
    aligner = Wav2Vec2PhonemeAligner()
    log_probs = frames(0, 1, 1, 0, 2, 2, 0)
    monkeypatch.setattr(aligner, "_posteriors", lambda samples: (log_probs, 0, VOCAB))
    result = aligner.align(b"\x00\x00" * 800, Language.MANDARIN)
    assert [u.label for u in result.units] == ["n", "iɜ"]


def xiexie_ni_reference():
    labels = [("ɕ", 5, None), ("iɛ5", 6, 4)] * 2 + [("n", 1, None), ("i2", 7, 3)]
    words = ["谢", "谢", "谢", "谢", "你", "你"]
    return AlignmentResult(
        Language.MANDARIN,
        "谢谢你",
        [
            AlignedUnit(lab, j * 0.1, j * 0.1 + 0.1, 0.9, token_ids=[t], word=words[j], word_index=j // 2, tone=tone)
            for j, (lab, t, tone) in enumerate(labels)
        ],
    )


def test_learner_who_says_one_xie_skips_one_word_not_both():
    learner = AlignmentResult(
        Language.MANDARIN, "", [],
        posteriors=CTCPosteriors(frames(0, 5, 5, 0, 6, 6, 0, 1, 0, 7, 7, 0), 0.02, 0),
    )
    forced, pairs = force_to_reference(learner, xiexie_ni_reference())
    assert [u.label for u in forced.units] == ["ɕ", "iɛ5", "n", "i2"]
    assert len({forced.units[0].word_index, forced.units[2].word_index}) == 2
    assert forced.units[3].tone == 3


def contour(hz):
    return PitchContour(np.asarray(hz, dtype=float), 0.01)


def ma_result(start=0.0):
    return AlignmentResult(
        Language.MANDARIN,
        "妈",
        [
            AlignedUnit("m", start, start + 0.1, 0.9),
            AlignedUnit("ɑ5", start + 0.1, start + 0.3, 0.9, tone=1),
        ],
    )


def test_mandarin_final_gets_tone_and_shape_distance():
    level = contour([200.0] * 30)
    rising = contour([150.0] * 10 + list(np.geomspace(150.0, 300.0, 20)))
    result = score_utterance(ma_result(), ma_result(), learner_pitch=rising, reference_pitch=level)
    initial, final = result.units
    assert (initial.tone, initial.tone_distance) == (None, None)
    assert final.tone == 1
    assert final.tone_distance > 2.0  # a rise of an octave over a level tone


def test_same_tone_in_a_different_register_is_not_a_tone_error():
    reference = contour([200.0] * 30)
    learner = contour([320.0] * 30)
    result = score_utterance(ma_result(), ma_result(), learner_pitch=learner, reference_pitch=reference)
    assert result.units[1].tone_distance == pytest.approx(0.0, abs=1e-9)


def test_tone_distance_is_none_without_enough_voicing_or_off_mandarin():
    voiced_initial_only = contour([200.0] * 10 + [np.nan] * 20)
    result = score_utterance(
        ma_result(), ma_result(), learner_pitch=voiced_initial_only, reference_pitch=contour([200.0] * 30)
    )
    assert result.units[1].tone == 1 and result.units[1].tone_distance is None
    english = AlignmentResult(Language.ENGLISH, "", [AlignedUnit("ɑ5", 0.0, 0.2, 0.9)])
    scored = score_utterance(english, english, learner_pitch=contour([200.0] * 20), reference_pitch=contour([200.0] * 20))
    assert scored.units[0].tone_distance is None


@pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng not installed")
def test_espeak_cmn_gives_toned_syllables():
    ipa, mnemonics = mandarin.espeak_cmn("妈马")
    assert len(ipa.split()) == 2
    assert [mandarin.tone_from_mnemonic(s) for s in mnemonics.replace("_|", " ").split()] == [1, 3]
