import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import main  # noqa: E402
from app.services.alignment import (  # noqa: E402
    Language,
    Wav2Vec2PhonemeAligner,
    units_from_ctc,
)

BLANK = 0
VOCAB = {0: "<pad>", 1: "t", 2: "iː", 3: "aɪ"}


def frames(*best, p=0.9):
    rows = np.full((len(best), len(VOCAB)), (1 - p) / (len(VOCAB) - 1))
    rows[np.arange(len(best)), best] = p
    return np.log(rows)


def test_one_unit_per_emitted_token_with_frame_times():
    units = units_from_ctc(frames(0, 1, 1, 0, 2, 2, 2, 0), 0.02, BLANK, VOCAB)
    assert [u.label for u in units] == ["t", "iː"]
    assert [u.token_ids for u in units] == [[1], [2]]
    # A unit owns the blanks after it, up to the next unit's first frame.
    assert units[0].start_s == pytest.approx(0.02) and units[0].end_s == pytest.approx(0.08)
    assert units[1].start_s == pytest.approx(0.08) and units[1].end_s == pytest.approx(0.14)
    assert units[0].ctc_confidence == pytest.approx(0.9)


def test_repeated_phoneme_split_by_blank_is_two_units():
    units = units_from_ctc(frames(3, 0, 3), 0.02, BLANK, VOCAB)
    assert [u.label for u in units] == ["aɪ", "aɪ"]


def test_silence_decodes_to_no_units():
    assert units_from_ctc(frames(0, 0, 0), 0.02, BLANK, VOCAB) == []


def test_environment_selects_phoneme_aligner(monkeypatch):
    monkeypatch.setenv("SHADOWTALK_ALIGNER", "phoneme")
    monkeypatch.setattr(Wav2Vec2PhonemeAligner, "available", staticmethod(lambda: True))
    aligner = main.default_aligner()
    assert isinstance(aligner, Wav2Vec2PhonemeAligner)
    assert aligner.model_name == "facebook/wav2vec2-xlsr-53-espeak-cv-ft"


@pytest.mark.parametrize("pcm", [b"", b"\x00\x00" * 399])
def test_too_short_audio_aligns_to_nothing_without_loading_the_model(pcm, monkeypatch):
    aligner = Wav2Vec2PhonemeAligner()
    monkeypatch.setattr(aligner, "_load", lambda: pytest.fail("model loaded"))
    result = aligner.align(pcm, Language.ENGLISH)
    assert result.units == [] and result.transcript == ""
