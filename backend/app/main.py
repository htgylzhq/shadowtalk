"""ShadowTalk API.

Endpoints:
    GET  /health     liveness probe
    WS   /ws/align   streaming shadowing session

Streaming protocol (client -> server):
    - binary messages: 20 ms Opus audio frames, captured at 16 kHz mono
    - text message {"type": "start", "language": "ja",
                    "reference_id": "<lesson sentence id>"}
    - text message {"type": "end"} when the utterance is finished

Server -> client:
    - {"type": "score", ...} with per-unit GOP scores once alignment of
      the utterance completes (target: < 300 ms post-utterance, because
      alignment runs incrementally on the GPU worker during streaming).

Silero VAD endpointing runs on the worker side to detect the end of an
utterance when the client does not send an explicit "end".
"""

from __future__ import annotations

import json
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.services.alignment import (
    AlignmentResult,
    detect_language,
    profile_for,
    score_utterance,
)

app = FastAPI(title="ShadowTalk API", version="0.1.0")

SAMPLE_RATE_HZ = 16_000
FRAME_MS = 20
# 20 ms of 16 kHz mono PCM16 = 320 samples = 640 bytes (Opus frames vary).
PCM_FRAME_BYTES = SAMPLE_RATE_HZ * FRAME_MS // 1000 * 2


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "shadowtalk", "version": "0.1.0"}


class UtteranceSession:
    """Buffers one utterance's frames until endpointing."""

    def __init__(self) -> None:
        self.frames: list[bytes] = []
        self.language_hint: str | None = None
        self.reference_id: str | None = None
        self.started_at: float = time.monotonic()

    @property
    def pcm(self) -> bytes:
        return b"".join(self.frames)

    @property
    def frame_count(self) -> int:
        return len(self.frames)


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
                # TODO(worker): forward each frame to the GPU alignment
                # worker (Redis queue) so alignment runs incrementally,
                # and run Silero VAD endpointing over the stream.
                session.frames.append(message["bytes"])
                continue

            text = message.get("text")
            if not text:
                continue
            event = json.loads(text)

            if event.get("type") == "start":
                session = UtteranceSession()
                session.language_hint = event.get("language")
                session.reference_id = event.get("reference_id")
                await websocket.send_json({"type": "ready"})

            elif event.get("type") == "end":
                score = await _score_session(session)
                await websocket.send_json(score)
                session = UtteranceSession()
    except WebSocketDisconnect:
        return


async def _score_session(session: UtteranceSession) -> dict:
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
        "language": language.value,
        "alignment_unit": profile.unit.value,
        "reference_id": session.reference_id,
        "frames_received": session.frame_count,
        "overall_gop": result.overall_gop,
        "pitch_dtw_distance": result.pitch_dtw_distance,
        "units": [
            {"label": u.label, "gop": u.gop, "flagged": u.flagged}
            for u in result.units
        ],
        "session_ms": latency_ms,
    }
