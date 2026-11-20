"""Day 3 tests: WhisperX alignment wiring into the WS score payload.

Run from backend/:  .venv/bin/python -m pytest tests -q
Uses a fake aligner injected via main.ALIGNER — no torch, no GPU,
no model download. The real WhisperXAligner is only exercised for its
"unavailable here" behaviour, which is the honest local-dev state.
"""

import sys
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
    WhisperXAligner,
    detect_language,
    profile_for,
    score_utterance,
    speaking_rate_normalise,
)

client = TestClient(app)


class FakeAligner:
    """Deterministic stand-in for WhisperX: two Japanese mora units."""

    name = "fake-whisperx"

    def align(self, pcm16k: bytes, language: Language) -> AlignmentResult:
        return AlignmentResult(
            language=language,
            transcript="こんにちは",
            units=[
                AlignedUnit("こん", 0.0, 0.30, 0.9),
                AlignedUnit("にち", 0.30, 0.60, 0.3),  # below the 0.5 gate
            ],
        )


def pcm_frame(amplitude: float = 0.3, samples: int = 320) -> bytes:
    import math
    import struct

    out = bytearray()
    for i in range(samples):
        value = int(amplitude * 32767 * math.sin(2 * math.pi * 440.0 * i / 16000))
        out += struct.pack("<h", value)
    return bytes(out)


def test_whisperx_aligner_unavailable_without_stack():
    assert not WhisperXAligner.available()
    with pytest.raises(AlignmentUnavailableError):
        WhisperXAligner().align(pcm_frame(), Language.JAPANESE)


def test_japanese_routes_to_mora_units():
    assert detect_language(b"", hint="ja") == Language.JAPANESE
    assert profile_for(Language.JAPANESE).unit.value == "mora"


def test_speaking_rate_normalise_skips_empty_reference():
    units = [AlignedUnit("a", 0.0, 0.5, 0.9), AlignedUnit("b", 0.5, 1.0, 0.9)]
    assert speaking_rate_normalise(units, []) == units


def test_score_gates_low_confidence_units():
    learner = AlignmentResult(
        language=Language.JAPANESE,
        transcript="x",
        units=[
            AlignedUnit("a", 0.0, 0.5, 0.9),
            AlignedUnit("b", 0.5, 1.0, 0.3),
        ],
    )
    reference = AlignmentResult(language=Language.JAPANESE, transcript="x", units=[])
    result = score_utterance(learner, reference)
    assert result.units[0].gop == pytest.approx(0.9)
    assert result.units[0].flagged is False
    assert result.units[1].gop is None  # gated, never scored as an error
    assert result.units[1].flagged is True
    assert result.overall_gop == pytest.approx(0.9)


def test_ws_score_reports_alignment_unavailable_by_default(monkeypatch):
    monkeypatch.setattr(main, "ALIGNER", None)
    with client.websocket_connect("/ws/align") as ws:
        ws.send_json({"type": "start", "language": "ja", "reference_id": "s1"})
        assert ws.receive_json()["type"] == "ready"
        for _ in range(10):
            ws.send_bytes(pcm_frame())
        ws.send_json({"type": "end"})
        assert ws.receive_json()["type"] == "endpoint"
        score = ws.receive_json()
        assert score["type"] == "score"
        assert score["alignment"] == "unavailable"
        assert score["alignment_backend"] is None
        assert score["transcript"] == ""
        assert score["units"] == []
        assert score["overall_gop"] is None


def test_ws_score_returns_fake_aligner_units(monkeypatch):
    monkeypatch.setattr(main, "ALIGNER", FakeAligner())
    with client.websocket_connect("/ws/align") as ws:
        ws.send_json({"type": "start", "language": "ja", "reference_id": "s1"})
        assert ws.receive_json()["type"] == "ready"
        for _ in range(10):
            ws.send_bytes(pcm_frame())
        ws.send_json({"type": "end"})
        assert ws.receive_json()["type"] == "endpoint"
        score = ws.receive_json()
        assert score["type"] == "score"
        assert score["alignment"] == "ok"
        assert score["alignment_backend"] == "fake-whisperx"
        assert score["alignment_unit"] == "mora"
        assert score["transcript"] == "こんにちは"
        assert score["units"] == [
            {"label": "こん", "gop": 0.9, "flagged": False, "duration_ratio": None},
            {"label": "にち", "gop": None, "flagged": True, "duration_ratio": None},
        ]
        assert score["overall_gop"] == pytest.approx(0.9)
