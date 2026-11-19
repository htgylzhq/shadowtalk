"""ShadowTalk API.

Endpoints:
    GET  /health     liveness probe
    WS   /ws/align   streaming shadowing session

Streaming protocol (client -> server):
    - binary messages: 20 ms PCM16 frames, 16 kHz mono (640 bytes each)
    - text message {"type": "start", "language": "ja",
                    "reference_id": "<lesson sentence id>"}
    - text message {"type": "end"} when the utterance is finished

Server -> client:
    - {"type": "ready"} on start
    - {"type": "endpoint", ...} when Silero/energy VAD decides the
      utterance ended (silence after speech), even without an "end"
    - {"type": "score", ...} with per-unit GOP scores once alignment of
      the utterance completes (target: < 300 ms post-utterance, because
      alignment runs incrementally on the GPU worker during streaming).

Day 2: VAD endpointing runs over the live stream and each finished
utterance is saved as a WAV under backend/data/sessions/<session>.wav
(never committed — see .gitignore). Silero VAD is used when torch and
the model are available; otherwise the energy VAD in services/vad.py
decides the endpoint.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.services.alignment import (
    AlignmentResult,
    detect_language,
    profile_for,
    score_utterance,
)
from app.services.vad import (
    FRAME_MS,
    VoiceActivityDetector,
    save_utterance_wav,
)

app = FastAPI(title="ShadowTalk API", version="0.1.0")

SAMPLE_RATE_HZ = 16_000
# 20 ms of 16 kHz mono PCM16 = 320 samples = 640 bytes.
PCM_FRAME_BYTES = SAMPLE_RATE_HZ * FRAME_MS // 1000 * 2

SESSIONS_DIR = Path(__file__).resolve().parents[1] / "data" / "sessions"


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "shadowtalk", "version": "0.1.0"}


class UtteranceSession:
    """Buffers one utterance's frames until endpointing."""

    def __init__(self) -> None:
        self.session_id: str = uuid.uuid4().hex[:12]
        self.frames: list[bytes] = []
        self.language_hint: str | None = None
        self.reference_id: str | None = None
        self.started_at: float = time.monotonic()
        self.vad = VoiceActivityDetector()
        self.endpoint_reason: str | None = None
        self.endpoint_sent: bool = False
        self._byte_buffer = b""

    @property
    def pcm(self) -> bytes:
        return b"".join(self.frames)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def add_frame(self, data: bytes) -> bool:
        """Append a frame, run VAD over complete 20 ms chunks.

        Returns True when VAD endpointing just fired for this utterance.
        """
        self.frames.append(data)
        self._byte_buffer += data
        endpoint = False
        while len(self._byte_buffer) >= PCM_FRAME_BYTES:
            chunk, self._byte_buffer = (
                self._byte_buffer[:PCM_FRAME_BYTES],
                self._byte_buffer[PCM_FRAME_BYTES:],
            )
            result = self.vad.process_frame(chunk)
            if result.endpoint_reached:
                endpoint = True
        return endpoint

    def save_utterance(self) -> str | None:
        """Persist the utterance WAV; returns the file name, or None if empty."""
        if not self.frames:
            return None
        path = save_utterance_wav(self.pcm, SESSIONS_DIR / f"{self.session_id}.wav")
        return path.name


@app.websocket("/ws/align")
async def align_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    session = UtteranceSession()
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"] is not None:
                if session.add_frame(message["bytes"]) and not session.endpoint_sent:
                    session.endpoint_sent = True
                    session.endpoint_reason = "vad"
                    await _finish_session(websocket, session)
                    session = UtteranceSession()
                continue

            text = message.get("text")
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                continue

            if event.get("type") == "start":
                session = UtteranceSession()
                session.language_hint = event.get("language")
                session.reference_id = event.get("reference_id")
                await websocket.send_json({"type": "ready", "session_id": session.session_id})

            elif event.get("type") == "end":
                if session.endpoint_sent:
                    continue
                session.endpoint_sent = True
                session.endpoint_reason = "client_end"
                await _finish_session(websocket, session)
                session = UtteranceSession()
    except WebSocketDisconnect:
        return


async def _finish_session(websocket: WebSocket, session: UtteranceSession) -> None:
    """Save the utterance, tell the client it ended, then score it."""
    wav_name = session.save_utterance()
    await websocket.send_json(
        {
            "type": "endpoint",
            "session_id": session.session_id,
            "reason": session.endpoint_reason,
            "speech_ms": session.vad.speech_ms,
            "utterance_wav": wav_name,
        }
    )
    await websocket.send_json(await _score_session(session, wav_name))


async def _score_session(session: UtteranceSession, wav_name: str | None = None) -> dict:
    """Run alignment + scoring for a finished utterance.

    Stub wiring: the GPU worker returns WhisperX alignments for both
    the learner audio and the canonical reference; scoring is relative
    to the reference alignment (see services/alignment.py).
    """
    language = detect_language(session.pcm, hint=session.language_hint)
    profile = profile_for(language)

    # TODO(worker): replace empty alignments with WhisperXAligner output
    # for learner audio and for the reference_id's canonical recording.
    learner = AlignmentResult(language=language, transcript="", units=[])
    reference = AlignmentResult(language=language, transcript="", units=[])
    result = score_utterance(learner, reference)

    latency_ms = int((time.monotonic() - session.started_at) * 1000)
    return {
        "type": "score",
        "session_id": session.session_id,
        "language": language.value,
        "alignment_unit": profile.unit.value,
        "reference_id": session.reference_id,
        "frames_received": session.frame_count,
        "utterance_wav": wav_name,
        "endpoint_reason": session.endpoint_reason,
        "vad_backend": session.vad.backend,
        "speech_ms": session.vad.speech_ms,
        "overall_gop": result.overall_gop,
        "pitch_dtw_distance": result.pitch_dtw_distance,
        "units": [
            {"label": u.label, "gop": u.gop, "flagged": u.flagged}
            for u in result.units
        ],
        "session_ms": latency_ms,
    }
