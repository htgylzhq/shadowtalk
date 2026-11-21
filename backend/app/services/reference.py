"""Canonical reference recordings for ShadowTalk (Day 4).

A lesson sentence has one canonical reference recording — the native
speaker's version. It is forced-aligned *once* through the same
aligner as the learner, and the cached alignment is reused for every
learner attempt. Scoring the learner *relative* to that reference
alignment (same model, same language head) is what cancels model bias:
a boundary the model gets wrong on the reference is wrong the same
way on the learner, so it must not be scored as a learner error.

Storage: reference WAVs live under ``backend/data/references/<id>.wav``
(16 kHz mono PCM16, never committed — see .gitignore). In-process,
``ReferenceStore`` holds the aligned result so the GPU worker is not
re-run per attempt.
"""

from __future__ import annotations

import wave
from pathlib import Path

from app.services.alignment import (
    Aligner,
    AlignmentResult,
    AlignmentUnavailableError,
    Language,
)
from app.services.pitch import PitchContour, PitchExtractor


def load_reference_pcm(wav_path: str | Path) -> bytes:
    """Read a 16 kHz mono PCM16 WAV into raw bytes for the aligner."""
    path = Path(wav_path)
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError(
                f"reference WAV must be mono PCM16, got "
                f"{wav.getnchannels()}ch / {wav.getsampwidth() * 8}-bit: {path}"
            )
        if wav.getframerate() != 16_000:
            raise ValueError(
                f"reference WAV must be 16 kHz, got {wav.getframerate()} Hz: {path}"
            )
        return wav.readframes(wav.getnframes())


class ReferenceStore:
    """Cache of reference alignments, keyed by reference_id."""

    def __init__(self) -> None:
        self._cache: dict[str, AlignmentResult] = {}
        # Day 5: reference audio and its F0 contour, for pitch scoring.
        self._pcm: dict[str, bytes] = {}
        self._pitch: dict[str, PitchContour] = {}

    def get(self, reference_id: str) -> AlignmentResult | None:
        return self._cache.get(reference_id)

    def register(
        self,
        reference_id: str,
        result: AlignmentResult,
        pcm16k: bytes | None = None,
    ) -> AlignmentResult:
        self._cache[reference_id] = result
        if pcm16k is not None:
            self._pcm[reference_id] = pcm16k
            self._pitch.pop(reference_id, None)
        return result

    def forget(self, reference_id: str) -> None:
        self._cache.pop(reference_id, None)
        self._pcm.pop(reference_id, None)
        self._pitch.pop(reference_id, None)

    def pitch_contour(
        self,
        reference_id: str,
        extractor: PitchExtractor,
        wav_path: str | Path | None = None,
    ) -> PitchContour | None:
        """Extract the reference F0 contour once; later calls reuse it.

        Uses the reference audio kept from alignment / registration, or
        ``wav_path`` when given and present. None when there is no
        reference audio at all.
        """
        cached = self._pitch.get(reference_id)
        if cached is not None:
            return cached
        pcm = self._pcm.get(reference_id)
        if pcm is None and wav_path is not None and Path(wav_path).exists():
            pcm = load_reference_pcm(wav_path)
            self._pcm[reference_id] = pcm
        if pcm is None:
            return None
        contour = extractor.extract(pcm)
        self._pitch[reference_id] = contour
        return contour

    def align_reference(
        self,
        reference_id: str,
        pcm16k: bytes,
        language: Language,
        aligner: Aligner | None,
        text: str | None = None,
    ) -> AlignmentResult:
        """Align a reference recording once; later calls reuse the cache.

        With lesson ``text`` and an aligner that supports it, the units
        are the text's intended phonemes force-aligned to the recording.

        Raises AlignmentUnavailableError when no aligner is configured
        (local dev without the WhisperX stack) — callers surface that
        as "unavailable", never as an empty reference.
        """
        cached = self._cache.get(reference_id)
        if cached is not None:
            return cached
        if aligner is None:
            raise AlignmentUnavailableError(
                "no aligner configured: cannot align reference recording"
            )
        align_text = getattr(aligner, "align_text", None)
        if text and align_text is not None:
            result = align_text(pcm16k, language, text)
        else:
            result = aligner.align(pcm16k, language)
        self._cache[reference_id] = result
        self._pcm[reference_id] = pcm16k
        return result

    def align_from_wav(
        self,
        reference_id: str,
        wav_path: str | Path,
        language: Language,
        aligner: Aligner | None,
    ) -> AlignmentResult:
        """Align ``<id>.wav``, using ``<id>.txt`` beside it as lesson text."""
        text_path = Path(wav_path).with_suffix(".txt")
        text = text_path.read_text(encoding="utf-8").strip() if text_path.exists() else None
        return self.align_reference(
            reference_id, load_reference_pcm(wav_path), language, aligner, text
        )
