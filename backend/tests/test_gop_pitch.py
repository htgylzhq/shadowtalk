"""Day 5 tests: GOP from CTC posteriors, pitch contour DTW, payload v1.

Run from backend/:  .venv/bin/python -m pytest tests -q
Synthetic posteriors and F0 contours — no torch, no GPU, no model
download. torchcrepe and the wav2vec2 posterior path are exercised for
real only on a worker with torch installed (see ROADMAP Day 5).
"""

import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.main import app
from app.services.alignment import (
    AlignedUnit,
    AlignmentResult,
    Language,
    score_utterance,
    speaking_rate_normalise,
)
from app.services.gop import CTCPosteriors, ctc_viterbi_log_likelihood, gop_score
from app.services.pitch import (
    CrepePitchExtractor,
    PitchContour,
    PitchUnavailableError,
    compare_contours,
    dtw_distance,
    normalise_per_speaker,
)
from app.services.reference import ReferenceStore

client = TestClient(app)

BLANK, A, B, C = 0, 1, 2, 3
FRAME_S = 0.02


def posteriors(frames, vocab=4):
    """frames: list of {token: prob}; the rest of the mass is spread evenly."""
    rows = []
    for spec in frames:
        rest = (1.0 - sum(spec.values())) / (vocab - len(spec))
        rows.append([spec.get(k, rest) for k in range(vocab)])
    return CTCPosteriors(np.log(np.array(rows)), FRAME_S, BLANK)


def pcm_frame(amplitude: float = 0.3, samples: int = 320) -> bytes:
    out = bytearray()
    for i in range(samples):
        value = int(amplitude * 32767 * math.sin(2 * math.pi * 440.0 * i / 16000))
        out += struct.pack("<h", value)
    return bytes(out)


# --- GOP ------------------------------------------------------------------


def test_viterbi_needs_blank_between_repeated_tokens():
    lp = posteriors([{A: 0.9}, {A: 0.9}]).log_probs
    assert ctc_viterbi_log_likelihood(lp, [A]) is not None
    # "A A" needs a blank between the repeats: 2 frames cannot fit it.
    assert ctc_viterbi_log_likelihood(lp, [A, A]) is None
    lp3 = posteriors([{A: 0.9}, {BLANK: 0.9}, {A: 0.9}]).log_probs
    assert ctc_viterbi_log_likelihood(lp3, [A, A]) == pytest.approx(3 * math.log(0.9))


def test_gop_is_one_when_target_is_what_the_model_hears():
    post = posteriors([{BLANK: 0.8}, {A: 0.9}, {A: 0.9}, {B: 0.9}, {BLANK: 0.8}])
    assert gop_score(post, [A, B], 0.0, 0.10) == pytest.approx(1.0)


def test_gop_drops_when_a_competing_token_wins_frames():
    # Learner was meant to say A B but the model hears A C.
    post = posteriors([{A: 0.9}, {A: 0.9}, {C: 0.9}, {C: 0.9}])
    good = gop_score(posteriors([{A: 0.9}, {A: 0.9}, {B: 0.9}, {B: 0.9}]), [A, B], 0, 0.08)
    bad = gop_score(post, [A, B], 0.0, 0.08)
    assert good == pytest.approx(1.0)
    assert bad is not None and bad < 0.5


def test_gop_uses_only_the_units_own_frames():
    # Frames 0-1 are a clean A, frames 2-3 are a clean C.
    post = posteriors([{A: 0.9}, {A: 0.9}, {C: 0.9}, {C: 0.9}])
    assert gop_score(post, [A], 0.0, 0.04) == pytest.approx(1.0)
    assert gop_score(post, [C], 0.04, 0.08) == pytest.approx(1.0)
    assert gop_score(post, [A], 0.04, 0.08) < 0.5


def test_gop_none_when_span_is_empty_or_tokens_missing():
    post = posteriors([{A: 0.9}])
    assert gop_score(post, [A], 0.5, 0.6) is None  # beyond the last frame
    assert gop_score(post, [], 0.0, 0.02) is None


def test_speaking_rate_normalise_keeps_token_ids():
    units = [AlignedUnit("a", 0.0, 0.2, 0.9, [A])]
    ref = [AlignedUnit("a", 0.0, 0.1, 0.9)]
    assert speaking_rate_normalise(units, ref)[0].token_ids == [A]


def _learner_with_posteriors():
    # Unit "a" (frames 0-2) is pronounced cleanly; unit "b" (frames 3-5)
    # is heard as C instead of B.
    post = posteriors(
        [{A: 0.9}, {A: 0.9}, {BLANK: 0.9}, {C: 0.9}, {C: 0.9}, {BLANK: 0.9}]
    )
    return AlignmentResult(
        language=Language.ENGLISH,
        transcript="ab",
        units=[
            AlignedUnit("a", 0.0, 0.06, 0.9, [A]),
            AlignedUnit("b", 0.06, 0.12, 0.9, [B]),
        ],
        posteriors=post,
    )


def test_score_utterance_uses_ctc_posteriors_for_gop():
    result = score_utterance(
        _learner_with_posteriors(),
        AlignmentResult(language=Language.ENGLISH, transcript="", units=[]),
    )
    assert result.gop_source == "ctc_posteriors"
    assert result.units[0].gop == pytest.approx(1.0)
    assert result.units[0].gop_source == "ctc_posteriors"
    assert result.units[1].gop < 0.5
    assert result.units[1].flagged is False
    assert result.overall_gop == pytest.approx(
        (result.units[0].gop + result.units[1].gop) / 2
    )


def test_confidence_gating_still_wins_over_posteriors():
    learner = _learner_with_posteriors()
    learner.units[1].ctc_confidence = 0.3
    result = score_utterance(
        learner, AlignmentResult(language=Language.ENGLISH, transcript="", units=[])
    )
    assert result.units[1].gop is None
    assert result.units[1].flagged is True
    assert result.overall_gop == pytest.approx(1.0)


def test_unit_whose_frames_cannot_fit_its_tokens_is_gated():
    learner = _learner_with_posteriors()
    learner.units[0].token_ids = [A, A]  # needs 3 frames incl. blank: 3 ok
    learner.units[1] = AlignedUnit("bb", 0.06, 0.08, 0.9, [B, B])  # 1 frame
    result = score_utterance(
        learner, AlignmentResult(language=Language.ENGLISH, transcript="", units=[])
    )
    assert result.units[1].gop is None
    assert result.units[1].flagged is True


def test_aligner_without_posteriors_falls_back_and_says_so():
    learner = AlignmentResult(
        language=Language.JAPANESE,
        transcript="x",
        units=[AlignedUnit("a", 0.0, 0.3, 0.8)],
    )
    result = score_utterance(
        learner, AlignmentResult(language=Language.JAPANESE, transcript="", units=[])
    )
    assert result.units[0].gop == pytest.approx(0.8)
    assert result.gop_source == "ctc_confidence"


# --- Pitch ------------------------------------------------------------------


def contour(hz, frame_s=0.01):
    return PitchContour(np.array(hz, dtype=float), frame_s)


def rising(base_hz, n=40, semitones=6.0):
    return [base_hz * 2 ** (semitones * i / (n - 1) / 12) for i in range(n)]


def test_normalise_per_speaker_cancels_register():
    low = normalise_per_speaker(contour(rising(110.0)))
    high = normalise_per_speaker(contour(rising(220.0)))
    assert low is not None and high is not None
    np.testing.assert_allclose(low, high, atol=1e-9)
    assert normalise_per_speaker(contour([np.nan, np.nan])) is None


def test_same_intonation_an_octave_apart_scores_zero():
    cmp = compare_contours(contour(rising(110.0)), contour(rising(220.0)))
    assert cmp is not None
    assert cmp.distance == pytest.approx(0.0, abs=1e-9)
    assert cmp.unit_delta((0.0, 0.4), (0.0, 0.4)) == pytest.approx(0.0, abs=1e-9)


def test_dtw_absorbs_tempo_difference():
    ref = np.array([0.0, 1.0, 2.0, 3.0])
    slow = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    assert dtw_distance(slow, ref) == pytest.approx(0.0)
    assert dtw_distance(slow[::-1].copy(), ref) > 1.0


def test_flat_learner_against_rising_reference_has_signed_deltas():
    ref = contour(rising(200.0, n=40, semitones=8.0))
    flat = contour([120.0] * 40)
    cmp = compare_contours(flat, ref)
    assert cmp is not None
    assert cmp.distance > 1.0
    first = cmp.unit_delta((0.0, 0.1), (0.0, 0.1))
    last = cmp.unit_delta((0.3, 0.4), (0.3, 0.4))
    # The reference spans -4..+4 st around its median; flat sits at 0.
    assert first == pytest.approx(3.0, abs=0.3)  # higher than the low start
    assert last == pytest.approx(-3.0, abs=0.3)  # lower than the high end


def test_unvoiced_unit_has_no_pitch_delta():
    learner = contour([np.nan] * 10 + [150.0] * 10)
    cmp = compare_contours(learner, contour([200.0] * 20))
    assert cmp is not None
    assert cmp.unit_delta((0.0, 0.1), (0.0, 0.1)) is None
    assert cmp.unit_delta((0.1, 0.2), (0.1, 0.2)) == pytest.approx(0.0)


def test_score_utterance_reports_pitch_per_unit_and_skips_gated():
    learner = AlignmentResult(
        language=Language.JAPANESE,
        transcript="x",
        units=[AlignedUnit("a", 0.0, 0.2, 0.9), AlignedUnit("b", 0.2, 0.4, 0.3)],
    )
    reference = AlignmentResult(
        language=Language.JAPANESE,
        transcript="x",
        units=[AlignedUnit("a", 0.0, 0.2, 0.9), AlignedUnit("b", 0.2, 0.4, 0.9)],
    )
    result = score_utterance(
        learner,
        reference,
        learner_pitch=contour([120.0] * 40),
        reference_pitch=contour(rising(200.0, n=40, semitones=8.0)),
    )
    assert result.pitch_dtw_distance is not None and result.pitch_dtw_distance > 1.0
    assert result.units[0].pitch_delta is not None
    assert result.units[1].flagged is True
    assert result.units[1].pitch_delta is None
    # Day 4 relative scoring is unchanged by the pitch inputs.
    assert result.units[0].duration_ratio == pytest.approx(1.0)
    assert result.duration_dtw_distance == pytest.approx(0.0)


def test_crepe_extractor_unavailable_without_torch():
    if CrepePitchExtractor.available():
        pytest.skip("torchcrepe installed in this environment")
    with pytest.raises(PitchUnavailableError):
        CrepePitchExtractor().extract(pcm_frame())


class CountingPitchExtractor:
    name = "fake-crepe"
    calls = 0

    def extract(self, pcm16k: bytes) -> PitchContour:
        type(self).calls += 1
        return contour(rising(180.0, n=60))


def test_reference_store_extracts_reference_pitch_once():
    CountingPitchExtractor.calls = 0
    store = ReferenceStore()
    store.register(
        "s1", AlignmentResult(Language.JAPANESE, "x", []), pcm16k=pcm_frame()
    )
    first = store.pitch_contour("s1", CountingPitchExtractor())
    second = store.pitch_contour("s1", CountingPitchExtractor())
    assert first is second
    assert CountingPitchExtractor.calls == 1
    assert store.pitch_contour("no-audio", CountingPitchExtractor()) is None


# --- WS payload v1 ------------------------------------------------------------


class PosteriorFakeAligner:
    name = "fake-whisperx"

    def align(self, pcm16k: bytes, language: Language) -> AlignmentResult:
        return AlignmentResult(
            language=language,
            transcript="こんにちは",
            units=[
                AlignedUnit("こん", 0.0, 0.30, 0.9, [A]),
                AlignedUnit("にち", 0.30, 0.60, 0.3, [B]),  # below the gate
            ],
            posteriors=posteriors([{A: 0.9}] * 15 + [{B: 0.9}] * 15),
        )


def _run_ws(reference_id="s1"):
    with client.websocket_connect("/ws/align") as ws:
        ws.send_json({"type": "start", "language": "ja", "reference_id": reference_id})
        assert ws.receive_json()["type"] == "ready"
        for _ in range(10):
            ws.send_bytes(pcm_frame())
        ws.send_json({"type": "end"})
        assert ws.receive_json()["type"] == "endpoint"
        return ws.receive_json()


def test_ws_score_payload_v1_with_gop_and_pitch(monkeypatch):
    monkeypatch.setattr(main, "ALIGNER", PosteriorFakeAligner())
    monkeypatch.setattr(main, "PITCH_EXTRACTOR", CountingPitchExtractor())
    main.REFERENCE_STORE.register(
        "s1",
        AlignmentResult(
            Language.JAPANESE,
            "こんにちは",
            [AlignedUnit("こん", 0.0, 0.30, 0.9), AlignedUnit("にち", 0.30, 0.60, 0.9)],
        ),
        pcm16k=pcm_frame(),
    )
    try:
        score = _run_ws()
    finally:
        main.REFERENCE_STORE.forget("s1")
    assert score["score_version"] == 1
    assert score["gop_source"] == "ctc_posteriors"
    assert score["pitch"] == "ok"
    assert score["pitch_backend"] == "fake-crepe"
    assert score["pitch_dtw_distance"] == pytest.approx(0.0, abs=1e-9)
    assert score["scores"] == [
        {"phoneme": "こん", "gop": pytest.approx(1.0), "pitch_delta": pytest.approx(0.0, abs=1e-9), "flagged": False},
        {"phoneme": "にち", "gop": None, "pitch_delta": None, "flagged": True},
    ]
    # Day 4 fields are still there and unchanged in shape.
    assert score["reference_alignment"] == "ok"
    assert score["units"][0]["duration_ratio"] == pytest.approx(1.0)


def test_ws_pitch_missing_without_reference_audio(monkeypatch):
    monkeypatch.setattr(main, "ALIGNER", PosteriorFakeAligner())
    monkeypatch.setattr(main, "PITCH_EXTRACTOR", CountingPitchExtractor())
    score = _run_ws(reference_id="no-such-lesson")
    assert score["pitch"] == "missing"
    assert score["pitch_dtw_distance"] is None
    assert score["scores"][0]["pitch_delta"] is None
    assert score["scores"][0]["gop"] == pytest.approx(1.0)


def test_ws_pitch_unavailable_by_default(monkeypatch):
    monkeypatch.setattr(main, "ALIGNER", None)
    monkeypatch.setattr(main, "PITCH_EXTRACTOR", None)
    score = _run_ws()
    assert score["pitch"] == "unavailable"
    assert score["pitch_backend"] is None
    assert score["score_version"] == 1
    assert score["scores"] == []


@pytest.mark.parametrize("reference_id", ["../sessions/abc", "a/b", "", 7])
def test_ws_rejects_reference_id_that_is_not_a_slug(reference_id):
    with client.websocket_connect("/ws/align") as ws:
        ws.send_json({"type": "start", "language": "ja", "reference_id": reference_id})
        assert ws.receive_json() == {"type": "error", "error": "invalid_reference_id"}
        ws.send_json({"type": "start", "language": "ja", "reference_id": "lesson-1_a"})
        assert ws.receive_json()["type"] == "ready"


def test_silence_around_speech_does_not_count_as_pitch():
    speech = rising(180.0)
    padded = [np.nan] * 30 + speech + [np.nan] * 50
    cmp = compare_contours(contour(padded), contour(rising(120.0)))
    assert cmp.distance == pytest.approx(0.0, abs=1e-9)


def test_long_recordings_are_bounded_and_still_compared(monkeypatch):
    import time as _time
    long_rise = rising(150.0, n=6000)
    start = _time.monotonic()
    same = compare_contours(contour(long_rise), contour(rising(200.0, n=5000)))
    flat = compare_contours(contour([150.0] * 6000), contour(rising(200.0, n=5000)))
    assert _time.monotonic() - start < 5.0
    assert same.distance == pytest.approx(0.0, abs=0.05)
    assert flat.distance > 1.0


def test_gop_scores_the_reference_unit_not_a_substituted_word():
    # The learner said B where the reference has A; the model hears B clearly.
    learner = AlignmentResult(
        language=Language.ENGLISH,
        transcript="b",
        units=[AlignedUnit("b", 0.0, 0.10, 0.9, [B])],
        posteriors=posteriors([{B: 0.9}] * 5),
    )
    reference = AlignmentResult(
        language=Language.ENGLISH,
        transcript="a",
        units=[AlignedUnit("a", 0.0, 0.10, 0.9, [A])],
    )
    assert score_utterance(learner, reference).units[0].gop < 0.1


def test_pitch_extractor_follows_environment(monkeypatch):
    monkeypatch.delenv("SHADOWTALK_PITCH", raising=False)
    assert main.default_pitch_extractor() is None
    monkeypatch.setenv("SHADOWTALK_PITCH", "torchcrepe")
    monkeypatch.setattr(main.CrepePitchExtractor, "available", staticmethod(lambda: True))
    assert isinstance(main.default_pitch_extractor(), main.CrepePitchExtractor)


def test_registering_new_reference_audio_drops_cached_pitch():
    CountingPitchExtractor.calls = 0
    store = ReferenceStore()
    empty = AlignmentResult(Language.JAPANESE, "x", [])
    store.register("s1", empty, pcm16k=pcm_frame())
    first = store.pitch_contour("s1", CountingPitchExtractor())
    store.register("s1", empty, pcm16k=pcm_frame(amplitude=0.5))
    assert store.pitch_contour("s1", CountingPitchExtractor()) is not first
    assert CountingPitchExtractor.calls == 2
