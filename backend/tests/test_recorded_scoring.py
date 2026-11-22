"""Opt-in integration test for the model-backed recording path."""

import math
import os
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main
from app.services.alignment import Wav2Vec2PhonemeAligner
from app.services.pitch import CrepePitchExtractor
from app.services.reference import ReferenceStore, load_reference_pcm


@pytest.mark.skipif(
    os.getenv("SHADOWTALK_RUN_MODEL_TESTS") != "1",
    reason="requires the wav2vec2 and torchcrepe models; see README",
)
def test_recorded_wav_returns_relative_gop_and_pitch(monkeypatch, tmp_path):
    reference_id = "librispeech"
    reference = tmp_path / f"{reference_id}.wav"
    shutil.copyfile(Path(__file__).with_name("librispeech.wav"), reference)
    # Replay the same speaker as a baseline; a final 5 ms of silence also
    # exercises the partial-frame path without changing the spoken content.
    pcm = load_reference_pcm(reference) + bytes(main.PCM_FRAME_BYTES // 4)

    monkeypatch.setenv("SHADOWTALK_ALIGNER", "phoneme")
    monkeypatch.setenv("SHADOWTALK_PITCH", "torchcrepe")
    aligner = main.default_aligner()
    pitch = main.default_pitch_extractor()
    assert isinstance(aligner, Wav2Vec2PhonemeAligner)
    assert isinstance(pitch, CrepePitchExtractor)
    monkeypatch.setattr(main, "ALIGNER", aligner)
    monkeypatch.setattr(main, "PITCH_EXTRACTOR", pitch)
    monkeypatch.setattr(main, "REFERENCES_DIR", tmp_path)
    monkeypatch.setattr(main, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(main, "REFERENCE_STORE", ReferenceStore())
    # The transport is client-ended here so Silero availability does not
    # change which samples reach the real scoring models.
    monkeypatch.setattr(main.VoiceActivityDetector, "_try_load_silero", staticmethod(lambda: None))

    with TestClient(main.app) as client, client.websocket_connect("/ws/align") as ws:
        ws.send_json({"type": "start", "language": "en", "reference_id": reference_id})
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        for offset in range(0, len(pcm), main.PCM_FRAME_BYTES):
            ws.send_bytes(pcm[offset:offset + main.PCM_FRAME_BYTES])
        ws.send_json({"type": "end"})
        endpoint = ws.receive_json()
        score = ws.receive_json()

    assert endpoint["type"] == "endpoint"
    assert endpoint["reason"] == "client_end"
    assert endpoint["speech_ms"] > 0
    assert load_reference_pcm(tmp_path / endpoint["utterance_wav"]) == pcm
    assert score["type"] == "score"
    assert score["session_id"] == ready["session_id"]
    assert score["score_version"] == 1
    assert score["reference_id"] == reference_id
    assert score["alignment"] == score["reference_alignment"] == score["pitch"] == "ok"
    assert score["alignment_backend"] == "wav2vec2-phoneme"
    assert score["gop_source"] == "ctc_posteriors"
    assert score["pitch_backend"] == "torchcrepe"
    assert 0 < score["overall_gop"] <= 1
    scored = [row for row in score["scores"] if not row["flagged"]]
    assert len(scored) >= 3
    assert all(row["phoneme"] and 0 <= row["gop"] <= 1 for row in scored)
    assert all(
        row["gop"] is None and row["pitch_delta"] is None
        for row in score["scores"] if row["flagged"]
    )
    deltas = [row["pitch_delta"] for row in scored if row["pitch_delta"] is not None]
    assert deltas and all(math.isfinite(delta) and abs(delta) < 0.2 for delta in deltas)
    contour = score["pitch_contour"]
    assert contour["frame_s"] == pytest.approx(0.01)
    for trace in (contour["learner"], contour["reference"]):
        assert sum(sample is not None and math.isfinite(sample) for sample in trace) >= 10
    assert score["pitch_dtw_distance"] == pytest.approx(0, abs=0.2)
