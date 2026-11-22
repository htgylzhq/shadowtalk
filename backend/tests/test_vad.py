"""Day 2 tests: VAD endpointing and the streaming WebSocket.

Run from backend/:  .venv/bin/python -m pytest tests -q
Uses synthetic PCM only — no microphone, no torch, no GPU.
"""

import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.vad import VoiceActivityDetector, frame_rms, save_utterance_wav


def pcm_frame(amplitude: float, freq_hz: float = 440.0, samples: int = 320) -> bytes:
    """One 20 ms frame: a sine tone at `amplitude`, or silence at 0."""
    out = bytearray()
    for i in range(samples):
        value = int(amplitude * 32767 * math.sin(2 * math.pi * freq_hz * i / 16000))
        out += struct.pack("<h", value)
    return bytes(out)


def test_silence_never_endpoints():
    vad = VoiceActivityDetector()
    assert vad.backend == "energy"  # torch/Silero not installed in dev venv
    for _ in range(100):  # 2 s of silence
        result = vad.process_frame(pcm_frame(0.0))
        assert not result.is_speech
        assert not result.endpoint_reached


def test_speech_then_silence_endpoints():
    vad = VoiceActivityDetector()
    for _ in range(25):  # 500 ms of speech
        result = vad.process_frame(pcm_frame(0.3))
        assert result.is_speech
        assert not result.endpoint_reached
    endpointed = False
    for _ in range(40):  # 800 ms of silence — endpoint is at 600 ms
        result = vad.process_frame(pcm_frame(0.0))
        if result.endpoint_reached:
            endpointed = True
            break
    assert endpointed
    assert vad.speech_ms == 500


def test_short_blip_does_not_endpoint():
    vad = VoiceActivityDetector()
    for _ in range(5):  # 100 ms — below MIN_SPEECH_MS (200 ms)
        vad.process_frame(pcm_frame(0.3))
    for _ in range(40):
        result = vad.process_frame(pcm_frame(0.0))
        assert not result.endpoint_reached


def test_frame_rms_silence_vs_speech():
    assert frame_rms(pcm_frame(0.0)) == 0.0
    assert frame_rms(pcm_frame(0.3)) > 0.1


def test_save_utterance_wav(tmp_path):
    path = save_utterance_wav(pcm_frame(0.3) * 10, tmp_path / "u.wav")
    assert path.exists()
    assert path.stat().st_size > 44  # WAV header + samples


# --- WebSocket end-to-end -------------------------------------------------

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_ws_vad_endpoint_and_saved_utterance(tmp_path, monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "SESSIONS_DIR", tmp_path)
    with client.websocket_connect("/ws/align") as ws:
        ws.send_json({"type": "start", "language": "ja", "reference_id": "s1"})
        assert ws.receive_json()["type"] == "ready"
        for _ in range(25):
            ws.send_bytes(pcm_frame(0.3))
        for _ in range(40):
            ws.send_bytes(pcm_frame(0.0))
        endpoint = ws.receive_json()
        assert endpoint["type"] == "endpoint"
        assert endpoint["reason"] == "vad"
        assert endpoint["speech_ms"] == 500
        assert endpoint["utterance_wav"] is not None
        assert (tmp_path / endpoint["utterance_wav"]).exists()
        score = ws.receive_json()
        assert score["type"] == "score"
        assert score["endpoint_reason"] == "vad"
        assert score["vad_backend"] == "energy"
        assert score["frames_received"] >= 25


def test_ws_explicit_end():
    with client.websocket_connect("/ws/align") as ws:
        ws.send_json({"type": "start", "language": "ja"})
        ws.receive_json()
        for _ in range(10):
            ws.send_bytes(pcm_frame(0.3))
        ws.send_json({"type": "end"})
        endpoint = ws.receive_json()
        assert endpoint["type"] == "endpoint"
        assert endpoint["reason"] == "client_end"
        score = ws.receive_json()
        assert score["type"] == "score"
        assert score["frames_received"] == 10


def test_empty_recording_does_not_load_the_models(monkeypatch, tmp_path):
    import app.main as main

    class UnusedModel:
        def align(self, *args):
            raise AssertionError("empty sessions must not run alignment")

        def extract(self, *args):
            raise AssertionError("empty sessions must not run pitch")

    monkeypatch.setattr(main, "ALIGNER", UnusedModel())
    monkeypatch.setattr(main, "PITCH_EXTRACTOR", UnusedModel())
    monkeypatch.setattr(main, "SESSIONS_DIR", tmp_path)
    with client.websocket_connect("/ws/align") as ws:
        ws.send_json({"type": "start", "language": "en"})
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(b"")
        ws.send_json({"type": "end"})
        endpoint = ws.receive_json()
        score = ws.receive_json()

    assert endpoint["utterance_wav"] is None
    assert score["frames_received"] == 0
    assert score["scores"] == []
    assert score["alignment"] != "error"
    assert score["pitch"] != "error"
    assert not list(tmp_path.iterdir())


def test_ws_ignores_stop_after_vad_until_the_next_start(monkeypatch, tmp_path):
    import app.main as main

    monkeypatch.setattr(main, "SESSIONS_DIR", tmp_path)
    with client.websocket_connect("/ws/align") as ws:
        ws.send_json({"type": "start"})
        assert ws.receive_json()["type"] == "ready"
        for _ in range(25):
            ws.send_bytes(pcm_frame(0.3))
        for _ in range(30):
            ws.send_bytes(pcm_frame(0.0))
        assert ws.receive_json()["reason"] == "vad"
        assert ws.receive_json()["type"] == "score"
        ws.send_json({"type": "end"})
        ws.send_json({"type": "start"})
        assert ws.receive_json()["type"] == "ready"


def test_health_responds_while_websocket_scoring_waits(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    import app.main as main
    from app.services.alignment import AlignmentResult

    entered, release, finished = Event(), Event(), Event()

    class SlowAligner:
        def align(self, pcm, language):
            entered.set()
            release.wait(5)
            finished.set()
            return AlignmentResult(language, "", [])

    monkeypatch.setattr(main, "ALIGNER", SlowAligner())
    monkeypatch.setattr(main, "SESSIONS_DIR", tmp_path)
    with (
        ThreadPoolExecutor(max_workers=1) as executor,
        TestClient(app) as shared_client,
        shared_client.websocket_connect("/ws/align") as ws,
    ):
        ws.send_json({"type": "start"})
        assert ws.receive_json()["type"] == "ready"

        def request_health_during_alignment():
            assert entered.wait(2)
            response = shared_client.get("/health")
            assert not finished.is_set()
            return response

        request = executor.submit(request_health_during_alignment)
        ws.send_bytes(pcm_frame(0.3))
        ws.send_json({"type": "end"})
        try:
            assert request.result(timeout=2).json()["status"] == "ok"
        finally:
            release.set()
        assert ws.receive_json()["type"] == "endpoint"
        assert ws.receive_json()["type"] == "score"
