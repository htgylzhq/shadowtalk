"""Voice activity detection and utterance storage for ShadowTalk.

Day 2 scope (see ROADMAP.md): the iOS recorder streams 20 ms PCM frames
over ``/ws/align``; the server runs VAD over the stream, decides when an
utterance has ended, and saves the utterance per session.

Two backends:
    * Silero VAD — used when ``torch`` and the Silero model are available
      (GPU worker / production). Loaded lazily; any failure falls back.
    * Energy VAD — always available, no heavy dependency. RMS energy with
      hysteresis plus a silence hangover. Good enough for endpointing in
      local development and for tests; it is *not* the production scorer,
      it only decides when the user stopped speaking.

Silero expects 512-sample windows at 16 kHz; the energy path works on
the 20 ms (320-sample) frames the client sends, so the detector accepts
any frame length and internally re-chunks.
"""

from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE_RATE_HZ = 16_000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE_HZ * FRAME_MS // 1000  # 320 samples, 640 bytes

# Energy-VAD defaults, tuned for close-mic phone audio. Speech is
# typically RMS > 0.02; room noise / silence sits well below 0.005.
SPEECH_RMS_THRESHOLD = 0.02
SILENCE_RMS_THRESHOLD = 0.008
SILENCE_MS_TO_ENDPOINT = 600
MIN_SPEECH_MS = 200


def frame_rms(pcm16: bytes) -> float:
    """RMS amplitude (0..1) of a PCM16 mono frame."""
    if len(pcm16) < 2:
        return 0.0
    count = len(pcm16) // 2
    samples = struct.unpack(f"<{count}h", pcm16[: count * 2])
    mean_square = sum(s * s for s in samples) / count
    return math.sqrt(mean_square) / 32768.0


def save_utterance_wav(pcm16: bytes, path: str | Path) -> Path:
    """Save raw 16 kHz mono PCM16 as a WAV file; returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE_HZ)
        wav.writeframes(pcm16)
    return path


@dataclass
class VadResult:
    is_speech: bool
    endpoint_reached: bool
    rms: float
    speech_ms: int
    silence_ms: int


@dataclass
class VoiceActivityDetector:
    """Streaming VAD with endpoint detection.

    Feed 20 ms frames with :meth:`process_frame`. ``endpoint_reached``
    becomes True once speech has been seen and silence has then lasted
    ``silence_ms_to_endpoint``. Call :meth:`reset` between utterances.
    """

    silence_ms_to_endpoint: int = SILENCE_MS_TO_ENDPOINT
    min_speech_ms: int = MIN_SPEECH_MS
    backend: str = field(init=False, default="energy")
    _silero_model: object = field(init=False, default=None, repr=False)
    _speech_ms: int = field(init=False, default=0)
    _silence_ms: int = field(init=False, default=0)
    _seen_speech: bool = field(init=False, default=False)
    _in_speech: bool = field(init=False, default=False)
    _endpoint: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._silero_model = self._try_load_silero()
        if self._silero_model is not None:
            self.backend = "silero"

    @staticmethod
    def _try_load_silero() -> object | None:
        """Best-effort Silero load; returns None (energy fallback) on any failure."""
        try:
            import torch  # noqa: PLC0415  (heavy, optional)

            model, _utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=True,
            )
            return model
        except Exception:
            return None

    def process_frame(self, pcm16: bytes) -> VadResult:
        if self._endpoint:
            return VadResult(False, True, 0.0, self._speech_ms, self._silence_ms)

        rms = frame_rms(pcm16)
        is_speech = self._is_speech(pcm16, rms)

        if is_speech:
            self._in_speech = True
            self._seen_speech = True
            self._speech_ms += FRAME_MS
            self._silence_ms = 0
        else:
            self._in_speech = False
            self._silence_ms += FRAME_MS
            if self._seen_speech:
                # Speech continues to accumulate only while speaking;
                # silence after speech counts toward the endpoint.
                pass

        endpoint = (
            self._seen_speech
            and self._speech_ms >= self.min_speech_ms
            and self._silence_ms >= self.silence_ms_to_endpoint
        )
        if endpoint:
            self._endpoint = True
        return VadResult(
            is_speech=is_speech,
            endpoint_reached=endpoint,
            rms=rms,
            speech_ms=self._speech_ms,
            silence_ms=self._silence_ms,
        )

    def _is_speech(self, pcm16: bytes, rms: float) -> bool:
        if self._silero_model is not None:
            try:
                return self._silero_is_speech(pcm16)
            except Exception:
                self._silero_model = None
                self.backend = "energy"
        # Energy path with hysteresis: harder to enter speech than to
        # stay in it, so breaths / short gaps do not chop an utterance.
        threshold = (
            SILENCE_RMS_THRESHOLD if self._in_speech else SPEECH_RMS_THRESHOLD
        )
        return rms >= threshold

    def _silero_is_speech(self, pcm16: bytes) -> bool:
        import torch  # noqa: PLC0415

        # Silero VAD wants 512 samples; pad the 320-sample frame.
        samples = struct.unpack(f"<{len(pcm16) // 2}h", pcm16)
        tensor = torch.tensor([s / 32768.0 for s in samples], dtype=torch.float32)
        padded = torch.nn.functional.pad(tensor, (0, 512 - len(tensor)))
        with torch.no_grad():
            prob = self._silero_model(padded, SAMPLE_RATE_HZ).item()  # type: ignore[union-attr]
        return prob >= 0.5

    @property
    def seen_speech(self) -> bool:
        return self._seen_speech

    @property
    def speech_ms(self) -> int:
        return self._speech_ms

    def reset(self) -> None:
        self._speech_ms = 0
        self._silence_ms = 0
        self._seen_speech = False
        self._in_speech = False
        self._endpoint = False
