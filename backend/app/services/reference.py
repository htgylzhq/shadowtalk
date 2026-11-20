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

    def get(self, reference_id: str) -> AlignmentResult | None:
        return self._cache.get(reference_id)

    def register(self, reference_id: str, result: AlignmentResult) -> AlignmentResult:
        self._cache[reference_id] = result
        return result

    def align_reference(
        self,
        reference_id: str,
        pcm16k: bytes,
        language: Language,
        aligner: Aligner | None,
    ) -> AlignmentResult:
        """Align a reference recording once; later calls reuse the cache.

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
        result = aligner.align(pcm16k, language)
        self._cache[reference_id] = result
        return result

    def align_from_wav(
        self,
        reference_id: str,
        wav_path: str | Path,
        language: Language,
        aligner: Aligner | None,
    ) -> AlignmentResult:
        return self.align_reference(
            reference_id, load_reference_pcm(wav_path), language, aligner
        )
