import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import g2p  # noqa: E402
from app.services.alignment import (  # noqa: E402
    AlignedUnit,
    AlignmentResult,
    Language,
    Wav2Vec2PhonemeAligner,
    score_utterance,
)
from app.services.gop import ctc_forced_align  # noqa: E402
from app.services.reference import ReferenceStore  # noqa: E402

VOCAB = {"<pad>": 0, "t": 1, "iː": 2, "aɪ": 3, "a": 4, "ɪ": 5, "l": 6, "k": 7}
IPA = {"I": "ˈaɪ", "like": "lˈaɪk", "tea": "tˈiː"}


def fake_ipa(word, language):
    return IPA[word]


def frames(*best, p=0.9):
    rows = np.full((len(best), len(VOCAB)), (1 - p) / (len(VOCAB) - 1))
    rows[np.arange(len(best)), best] = p
    return np.log(rows)


def test_lesson_words_drop_punctuation_but_keep_apostrophes():
    assert g2p.lesson_words("I don't like tea, really!") == ["I", "don't", "like", "tea", "really"]


def test_lesson_words_keep_curly_apostrophe_contractions():
    assert g2p.lesson_words("I don\u2019t know") == ["I", "don't", "know"]


def test_tokenize_ipa_prefers_the_longest_vocabulary_token():
    # "aɪ" is one model token, not "a" + "ɪ"; stress marks carry no phoneme.
    assert g2p.tokenize_ipa("lˈaɪk", VOCAB) == [6, 3, 7]


def test_tokenize_ipa_drops_symbols_the_model_lacks():
    assert g2p.tokenize_ipa("tʃiː", VOCAB) == [1, 2]


def test_lesson_targets_tag_phonemes_with_their_word():
    targets = g2p.lesson_targets("I like tea.", Language.ENGLISH, VOCAB, ipa=fake_ipa)
    assert targets == [("I", [3]), ("like", [6, 3, 7]), ("tea", [1, 2])]


@pytest.mark.skipif(not g2p.espeak_available(), reason="espeak-ng not installed")
def test_espeak_cli_gives_ipa():
    assert "iː" in g2p.espeak_ipa("tea", Language.ENGLISH)


def test_forced_align_spans_each_target():
    log_probs = frames(0, 1, 1, 0, 2, 2, 0)
    assert ctc_forced_align(log_probs, [1, 2], 0) == [(1, 2), (4, 5)]


def test_forced_align_keeps_the_intended_phoneme_where_the_audio_differs():
    # The audio says "t aɪ"; the lesson says "t iː": iː still lands on a vowel frame.
    spans = ctc_forced_align(frames(0, 1, 0, 3, 3, 0), [1, 2], 0)
    assert spans[0] == (1, 1) and spans[1][0] in (3, 4)

def test_forced_align_needs_a_blank_between_repeats():
    assert ctc_forced_align(frames(1, 1), [1, 1], 0) is None
    assert ctc_forced_align(frames(1, 0, 1), [1, 1], 0) == [(0, 0), (2, 2)]


def test_align_text_force_aligns_lesson_phonemes(monkeypatch):
    aligner = Wav2Vec2PhonemeAligner()
    log_probs = frames(0, 1, 1, 0, 3, 3, 0)  # the learner-style audio says "t aɪ"
    monkeypatch.setattr(aligner, "_posteriors", lambda samples: (log_probs, 0, VOCAB))
    monkeypatch.setattr(g2p, "espeak_ipa", fake_ipa)
    result = aligner.align_text(b"\x00\x00" * 800, Language.ENGLISH, "tea")
    assert [(u.label, u.word, u.token_ids) for u in result.units] == [
        ("t", "tea", [1]),
        ("iː", "tea", [2]),
    ]
    assert result.transcript == "tea"
    assert result.posteriors is not None
    # Intended iː over frames the model heard as aɪ: low confidence, not a skip.
    assert result.units[1].ctc_confidence < 0.1


def test_align_text_falls_back_to_decoding_when_g2p_gives_nothing(monkeypatch):
    aligner = Wav2Vec2PhonemeAligner()
    log_probs = frames(0, 1, 1, 0, 2, 2, 0)
    monkeypatch.setattr(aligner, "_posteriors", lambda samples: (log_probs, 0, VOCAB))
    monkeypatch.setattr(g2p, "espeak_ipa", lambda word, language: "")
    monkeypatch.setattr(aligner, "align", lambda *a: pytest.fail("ran the model twice"))
    result = aligner.align_text(b"\x00\x00" * 800, Language.ENGLISH, "tea")
    assert [u.label for u in result.units] == ["t", "iː"]
    assert all(u.word is None for u in result.units)


def test_score_takes_word_from_the_paired_reference_unit():
    ref = AlignmentResult(
        Language.ENGLISH,
        "tea",
        [AlignedUnit("t", 0.0, 0.2, 0.9, word="tea"), AlignedUnit("iː", 0.2, 0.4, 0.9, word="tea")],
    )
    learner = AlignmentResult(
        Language.ENGLISH, "", [AlignedUnit("t", 0.0, 0.2, 0.9), AlignedUnit("aɪ", 0.2, 0.4, 0.9)]
    )
    result = score_utterance(learner, ref)
    assert [u.word for u in result.units] == ["tea", "tea"]


class TextAligner:
    name = "fake-text"

    def __init__(self):
        self.calls = []

    def align(self, pcm16k, language):
        self.calls.append(None)
        return AlignmentResult(language, "", [])

    def align_text(self, pcm16k, language, text):
        self.calls.append(text)
        return AlignmentResult(language, text, [AlignedUnit("t", 0, 0.1, 0.9, word="tea")])


def write_wav(path):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * 1600)


def test_reference_uses_lesson_text_beside_the_wav(tmp_path):
    write_wav(tmp_path / "lesson1.wav")
    (tmp_path / "lesson1.txt").write_text("tea\n", encoding="utf-8")
    aligner = TextAligner()
    result = ReferenceStore().align_from_wav("lesson1", tmp_path / "lesson1.wav", Language.ENGLISH, aligner)
    assert aligner.calls == ["tea"] and result.units[0].word == "tea"


def test_reference_without_lesson_text_decodes_the_recording(tmp_path):
    write_wav(tmp_path / "lesson2.wav")
    aligner = TextAligner()
    ReferenceStore().align_from_wav("lesson2", tmp_path / "lesson2.wav", Language.ENGLISH, aligner)
    assert aligner.calls == [None]


def test_align_text_falls_back_when_espeak_is_missing(monkeypatch):
    aligner = Wav2Vec2PhonemeAligner()
    log_probs = frames(0, 1, 1, 0, 2, 2, 0)
    monkeypatch.setattr(aligner, "_posteriors", lambda samples: (log_probs, 0, VOCAB))

    def missing(word, language):
        raise FileNotFoundError("espeak-ng")

    monkeypatch.setattr(g2p, "espeak_ipa", missing)
    result = aligner.align_text(b"\x00\x00" * 800, Language.ENGLISH, "tea")
    assert [u.label for u in result.units] == ["t", "iː"]
    assert all(u.word is None for u in result.units)
