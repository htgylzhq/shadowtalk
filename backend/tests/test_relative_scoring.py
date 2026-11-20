"""Day 4 tests: relative scoring vs a canonical reference alignment.

Run from backend/:  .venv/bin/python -m pytest tests -q
Uses fake aligners — no torch, no GPU, no model download. The real
WhisperXAligner is still only exercised for its "unavailable here"
behaviour; a real reference recording has not been aligned yet.
"""

import math
import struct
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.main import app
from app.services.alignment import (
    AlignedUnit,
    AlignmentResult,
    AlignmentUnavailableError,
    Language,
    dtw_duration_distance,
    score_utterance,
)
from app.services.reference import ReferenceStore, load_reference_pcm

client = TestClient(app)


def pcm_frame(amplitude: float = 0.3, samples: int = 320) -> bytes:
    out = bytearray()
    for i in range(samples):
        value = int(amplitude * 32767 * math.sin(2 * math.pi * 440.0 * i / 16000))
        out += struct.pack("<h", value)
    return bytes(out)


def units(spec):
    """spec: list of (label, start, end, confidence)."""
    return [AlignedUnit(label, s, e, c) for label, s, e, c in spec]


def result_for(spec, language=Language.JAPANESE):
    return AlignmentResult(language=language, transcript="x", units=units(spec))


class CountingFakeAligner:
    """Fake aligner that counts calls, to prove reference caching."""

    name = "fake-whisperx"
    calls = 0

    def align(self, pcm16k: bytes, language: Language) -> AlignmentResult:
        type(self).calls += 1
        return result_for([("こん", 0.0, 0.30, 0.9), ("にち", 0.30, 0.60, 0.9)])


class LearnerFakeAligner:
    name = "fake-whisperx"

    def align(self, pcm16k: bytes, language: Language) -> AlignmentResult:
        return result_for([("こん", 0.0, 0.30, 0.9), ("にち", 0.30, 0.60, 0.9)])


def test_dtw_duration_distance_identical_is_zero():
    assert dtw_duration_distance([0.3, 0.3], [0.3, 0.3]) == pytest.approx(0.0)
    assert dtw_duration_distance([], [0.3]) is None
    assert dtw_duration_distance([0.3], []) is None


def test_dtw_duration_distance_grows_with_rhythm_mismatch():
    assert dtw_duration_distance([0.1, 0.5], [0.3, 0.3]) == pytest.approx(0.2)


def test_relative_scoring_same_rhythm_after_rate_normalisation():
    # Learner speaks everything 2x slower; after normalisation the
    # durations match the reference exactly, so rhythm is perfect.
    learner = result_for([("a", 0.0, 0.6, 0.9), ("b", 0.6, 1.2, 0.9)])
    reference = result_for([("a", 0.0, 0.3, 0.9), ("b", 0.3, 0.6, 0.9)])
    scored = score_utterance(learner, reference)
    assert [u.duration_ratio for u in scored.units] == pytest.approx([1.0, 1.0])
    assert scored.duration_dtw_distance == pytest.approx(0.0)
    assert scored.overall_gop == pytest.approx(0.9)


def test_relative_scoring_flags_stretched_unit():
    # Same overall speed, but the learner stretches the first unit
    # and rushes the second: rhythm mismatch, not a GOP error.
    learner = result_for([("a", 0.0, 0.5, 0.9), ("b", 0.5, 0.6, 0.9)])
    reference = result_for([("a", 0.0, 0.3, 0.9), ("b", 0.3, 0.6, 0.9)])
    scored = score_utterance(learner, reference)
    assert scored.units[0].duration_ratio == pytest.approx(0.5 / 0.3)
    assert scored.units[1].duration_ratio == pytest.approx(0.1 / 0.3)
    assert scored.duration_dtw_distance is not None
    assert scored.duration_dtw_distance > 0.1
    # Pronunciation itself is still scored — rhythm is reported apart.
    assert scored.units[0].gop == pytest.approx(0.9)
    assert scored.units[0].flagged is False


def test_reference_low_confidence_gates_learner_unit():
    # The model cannot place the boundary reliably on the *reference*
    # either, so a confident-looking learner unit must not be scored
    # as an error (or as a pass) on that boundary.
    learner = result_for([("a", 0.0, 0.3, 0.95), ("b", 0.3, 0.6, 0.9)])
    reference = result_for([("a", 0.0, 0.3, 0.2), ("b", 0.3, 0.6, 0.9)])
    scored = score_utterance(learner, reference)
    assert scored.units[0].gop is None
    assert scored.units[0].flagged is True
    assert scored.units[1].gop == pytest.approx(0.9)
    assert scored.overall_gop == pytest.approx(0.9)


def test_reference_store_aligns_once_and_caches():
    CountingFakeAligner.calls = 0
    store = ReferenceStore()
    first = store.align_reference("s1", pcm_frame(), Language.JAPANESE, CountingFakeAligner())
    second = store.align_reference("s1", pcm_frame(), Language.JAPANESE, CountingFakeAligner())
    assert first is second
    assert CountingFakeAligner.calls == 1
    assert store.get("s1") is first
    assert store.get("unknown") is None


def test_reference_store_raises_without_aligner():
    store = ReferenceStore()
    with pytest.raises(AlignmentUnavailableError):
        store.align_reference("s1", pcm_frame(), Language.JAPANESE, None)


def test_load_reference_pcm_rejects_non_16k(tmp_path):
    path = tmp_path / "ref.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00" * 100)
    with pytest.raises(ValueError, match="16 kHz"):
        load_reference_pcm(path)


def test_ws_score_with_cached_reference_reports_relative_scores(monkeypatch):
    monkeypatch.setattr(main, "ALIGNER", LearnerFakeAligner())
    main.REFERENCE_STORE.register(
        "s1", result_for([("こん", 0.0, 0.30, 0.9), ("にち", 0.30, 0.60, 0.9)])
    )
    try:
        with client.websocket_connect("/ws/align") as ws:
            ws.send_json({"type": "start", "language": "ja", "reference_id": "s1"})
            assert ws.receive_json()["type"] == "ready"
            for _ in range(10):
                ws.send_bytes(pcm_frame())
            ws.send_json({"type": "end"})
            assert ws.receive_json()["type"] == "endpoint"
            score = ws.receive_json()
            assert score["type"] == "score"
            assert score["reference_alignment"] == "ok"
            assert score["duration_dtw_distance"] == pytest.approx(0.0)
            assert score["units"] == [
                {"label": "こん", "gop": 0.9, "flagged": False, "duration_ratio": 1.0},
                {"label": "にち", "gop": 0.9, "flagged": False, "duration_ratio": 1.0},
            ]
    finally:
        # Do not leak the registered reference into other tests.
        main.REFERENCE_STORE._cache.pop("s1", None)


def test_ws_score_without_reference_says_missing(monkeypatch):
    monkeypatch.setattr(main, "ALIGNER", LearnerFakeAligner())
    with client.websocket_connect("/ws/align") as ws:
        ws.send_json({"type": "start", "language": "ja", "reference_id": "no-such-lesson"})
        assert ws.receive_json()["type"] == "ready"
        for _ in range(10):
            ws.send_bytes(pcm_frame())
        ws.send_json({"type": "end"})
        assert ws.receive_json()["type"] == "endpoint"
        score = ws.receive_json()
        assert score["reference_alignment"] == "missing"
        assert score["duration_dtw_distance"] is None
        assert score["units"][0]["duration_ratio"] is None
        assert score["overall_gop"] == pytest.approx(0.9)
