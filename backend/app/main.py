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
import re
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.services.alignment import (
    PHONEME_MODEL,
    Wav2Vec2PhonemeAligner,
    Aligner,
    AlignmentResult,
    AlignmentUnavailableError,
    WhisperXAligner,
    detect_language,
    profile_for,
    score_utterance,
)
from app.services.pitch import (
    CrepePitchExtractor,
    PitchContour,
    PitchExtractor,
    PitchUnavailableError,
)
from app.services.reference import ReferenceStore, load_reference_pcm
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
REFERENCES_DIR = Path(__file__).resolve().parents[1] / "data" / "references"
# reference_id names a file under REFERENCES_DIR, so it is a bare slug.
REFERENCE_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")

# Day 4: cached reference alignments, keyed by reference_id. A
# reference recording is aligned once (same aligner as the learner)
# and reused for every attempt; see services/reference.py.
REFERENCE_STORE = ReferenceStore()

# Day 3: the aligner used to score finished utterances. None in local
# dev (no GPU / WhisperX stack here) — the score payload then says so
# explicitly ("alignment": "unavailable") instead of returning empty
# units as if alignment had run. Set it in production / on a GPU worker,
# e.g. `main.ALIGNER = WhisperXAligner(model_size="base", device="cuda")`,
# or via SHADOWTALK_ALIGNER=whisperx (see default_aligner below).


def default_aligner() -> Aligner | None:
    """Build the aligner from the environment, or None in local dev."""
    import os

    kind = os.environ.get("SHADOWTALK_ALIGNER")
    device = os.environ.get("SHADOWTALK_DEVICE", "cpu")
    if kind == "phoneme" and Wav2Vec2PhonemeAligner.available():
        return Wav2Vec2PhonemeAligner(
            model_name=os.environ.get("SHADOWTALK_PHONEME_MODEL", PHONEME_MODEL),
            device=device,
        )
    if kind != "whisperx":
        return None
    if not WhisperXAligner.available():
        return None
    return WhisperXAligner(
        model_size=os.environ.get("SHADOWTALK_WHISPER_MODEL", "tiny"),
        device=os.environ.get("SHADOWTALK_DEVICE", "cpu"),
    )


# Day 5: F0 extractor for pitch-contour scoring. None in local dev (no
# torch / torchcrepe here) — the payload then says "pitch": "unavailable".
# Set it on a GPU worker, e.g. `main.PITCH_EXTRACTOR =
# CrepePitchExtractor(device="cuda")`, or via SHADOWTALK_PITCH=torchcrepe.
SCORE_PAYLOAD_VERSION = 1


def default_pitch_extractor() -> PitchExtractor | None:
    """Build the pitch extractor from the environment, or None in local dev."""
    import os

    if os.environ.get("SHADOWTALK_PITCH") != "torchcrepe":
        return None
    if not CrepePitchExtractor.available():
        return None
    return CrepePitchExtractor(
        model=os.environ.get("SHADOWTALK_CREPE_MODEL", "tiny"),
        device=os.environ.get("SHADOWTALK_DEVICE", "cpu"),
    )



ALIGNER: Aligner | None = default_aligner()
PITCH_EXTRACTOR: PitchExtractor | None = default_pitch_extractor()


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
                reference_id = event.get("reference_id")
                if reference_id is not None and not (
                    isinstance(reference_id, str) and REFERENCE_ID_RE.fullmatch(reference_id)
                ):
                    await websocket.send_json(
                        {"type": "error", "error": "invalid_reference_id"}
                    )
                    continue
                session = UtteranceSession()
                session.language_hint = event.get("language")
                session.reference_id = reference_id
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


async def _score_session(
    session: UtteranceSession, wav_name: str | None = None
) -> dict:
    """Run alignment + scoring for a finished utterance.

    Day 4: the learner is scored *relative* to the canonical reference
    alignment for session.reference_id when one is available — from the
    in-memory ReferenceStore, or by aligning
    data/references/<reference_id>.wav once and caching it. When no
    reference exists the payload says so ("missing") and scoring falls
    back to the learner units alone, as in Day 3.
    """
    language = detect_language(session.pcm, hint=session.language_hint)
    profile = profile_for(language)

    aligner = ALIGNER
    alignment_status = "unavailable"
    learner = AlignmentResult(language=language, transcript="", units=[])
    if aligner is not None:
        try:
            learner = aligner.align(session.pcm, language)
            alignment_status = "ok"
        except AlignmentUnavailableError:
            alignment_status = "unavailable"
        except Exception:
            alignment_status = "error"

    reference = AlignmentResult(language=language, transcript="", units=[])
    reference_status = "missing"
    if session.reference_id:
        cached = REFERENCE_STORE.get(session.reference_id)
        if cached is not None:
            reference = cached
            reference_status = "ok"
        else:
            wav_path = REFERENCES_DIR / f"{session.reference_id}.wav"
            if wav_path.exists():
                if aligner is None:
                    reference_status = "unavailable"
                else:
                    try:
                        reference = REFERENCE_STORE.align_from_wav(
                            session.reference_id, wav_path, language, aligner
                        )
                        reference_status = "ok"
                    except AlignmentUnavailableError:
                        reference_status = "unavailable"
                    except Exception:
                        reference_status = "error"
    pitch_status, learner_pitch, reference_pitch = _pitch_contours(session)
    result = score_utterance(
        learner,
        reference,
        learner_pitch=learner_pitch,
        reference_pitch=reference_pitch,
    )
    if pitch_status == "ok" and result.pitch_dtw_distance is None:
        pitch_status = "unvoiced"

    latency_ms = int((time.monotonic() - session.started_at) * 1000)
    return {
        "type": "score",
        "session_id": session.session_id,
        "language": language.value,
        "alignment_unit": profile.unit.value,
        "alignment": alignment_status,
        "alignment_backend": getattr(aligner, "name", None) if aligner else None,
        "transcript": learner.transcript,
        "reference_id": session.reference_id,
        "reference_alignment": reference_status,
        "skipped": result.skipped,
        "duration_dtw_distance": result.duration_dtw_distance,
        "frames_received": session.frame_count,
        "utterance_wav": wav_name,
        "endpoint_reason": session.endpoint_reason,
        "vad_backend": session.vad.backend,
        "speech_ms": session.vad.speech_ms,
        "overall_gop": result.overall_gop,
        "gop_source": result.gop_source,
        "pitch": pitch_status,
        "pitch_backend": getattr(PITCH_EXTRACTOR, "name", None),
        "pitch_dtw_distance": result.pitch_dtw_distance,
        **(
            {"pitch_contour": result.pitch_comparison.contour_payload()}
            if result.pitch_comparison is not None
            else {}
        ),
        "units": [
            {
                "label": u.label,
                "gop": u.gop,
                "flagged": u.flagged,
                "duration_ratio": u.duration_ratio,
            }
            for u in result.units
        ],
        # Score payload v1 (Day 5): one entry per aligned unit. "phoneme"
        # is the unit label in the language profile's unit set (phoneme,
        # mora, or initial/final); "flagged" means confidence-gated, so
        # gop and pitch_delta are None and it is not a learner error.
        "score_version": SCORE_PAYLOAD_VERSION,
        "scores": [
            {
                # The unit label: an IPA phoneme with the phoneme aligner
                # (a Mandarin initial or toned final), a word with WhisperX.
                "phoneme": u.label,
                # Lesson word of the unit; None without <id>.txt lesson text.
                "word": u.word,
                "gop": u.gop,
                "pitch_delta": u.pitch_delta,
                # Mandarin finals only: lexical tone (1-4, 5 = neutral) and
                # the RMS semitone gap of the pitch shape vs. the reference.
                "tone": u.tone,
                "tone_distance": u.tone_distance,
                "flagged": u.flagged,
            }
            for u in result.units
        ],
        "session_ms": latency_ms,
    }


def _pitch_contours(
    session: UtteranceSession,
) -> tuple[str, PitchContour | None, PitchContour | None]:
    """F0 contours for the learner and the session's reference.

    Status: "ok" (both extracted), "missing" (no reference audio),
    "unavailable" (no extractor / torchcrepe), or "error".
    """
    extractor = PITCH_EXTRACTOR
    if extractor is None:
        return "unavailable", None, None
    try:
        learner_pitch = extractor.extract(session.pcm)
        reference_pitch = None
        if session.reference_id:
            reference_pitch = REFERENCE_STORE.pitch_contour(
                session.reference_id,
                extractor,
                REFERENCES_DIR / f"{session.reference_id}.wav",
            )
    except PitchUnavailableError:
        return "unavailable", None, None
    except Exception:
        return "error", None, None
    if reference_pitch is None:
        return "missing", learner_pitch, None
    return "ok", learner_pitch, reference_pitch
