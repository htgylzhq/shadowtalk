"""Alignment and scoring services for ShadowTalk.

Pipeline (production, on GPU workers):
    audio frames -> Silero VAD endpointing -> faster-whisper ASR
    -> wav2vec2 forced alignment (WhisperX) -> GOP scoring from CTC
    posteriors -> torchcrepe F0 pitch + DTW vs. reference contour.

This module defines the interfaces and the language-routing / scoring
logic. Heavy ML models are imported lazily inside the worker methods so
the API process can run without them in local development.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Language(str, Enum):
    ENGLISH = "en"
    JAPANESE = "ja"
    MANDARIN = "zh"


class AlignmentUnit(str, Enum):
    PHONEME = "phoneme"          # English (IPA)
    MORA = "mora"                # Japanese: mora-timed, phoneme is wrong unit
    INITIAL_FINAL = "initial_final"  # Mandarin: initial/final + tone


@dataclass(frozen=True)
class LanguageProfile:
    language: Language
    unit: AlignmentUnit
    phoneme_set: str
    xlsr_model: str  # language-specific wav2vec2 XLS-R alignment head


LANGUAGE_PROFILES: dict[Language, LanguageProfile] = {
    Language.ENGLISH: LanguageProfile(
        language=Language.ENGLISH,
        unit=AlignmentUnit.PHONEME,
        phoneme_set="IPA",
        xlsr_model="wav2vec2-xls-r-en",
    ),
    Language.JAPANESE: LanguageProfile(
        language=Language.JAPANESE,
        unit=AlignmentUnit.MORA,
        phoneme_set="mora",
        xlsr_model="wav2vec2-xls-r-ja",
    ),
    Language.MANDARIN: LanguageProfile(
        language=Language.MANDARIN,
        unit=AlignmentUnit.INITIAL_FINAL,
        phoneme_set="pinyin initial/final + tone",
        xlsr_model="wav2vec2-xls-r-zh",
    ),
}


@dataclass
class AlignedUnit:
    """One aligned unit (phoneme / mora / initial-final) of an utterance."""

    label: str
    start_s: float
    end_s: float
    ctc_confidence: float  # 0..1 posterior confidence for this boundary


@dataclass
class AlignmentResult:
    language: Language
    transcript: str
    units: list[AlignedUnit] = field(default_factory=list)


@dataclass
class UnitScore:
    label: str
    gop: float | None  # None when confidence-gated: never score a guess
    flagged: bool = False


@dataclass
class ScoreResult:
    units: list[UnitScore]
    overall_gop: float | None
    pitch_dtw_distance: float | None


def detect_language(audio: bytes, hint: str | None = None) -> Language:
    """Route an utterance to a language profile.

    Production: run a lightweight language-ID model over the first
    second of audio. A client-supplied hint (the lesson's language)
    wins when present, which is the normal in-app case.
    """
    if hint:
        return Language(hint)
    # TODO(worker): language-ID model on GPU worker.
    return Language.ENGLISH


def profile_for(language: Language) -> LanguageProfile:
    return LANGUAGE_PROFILES[language]


class WhisperXAligner:
    """Forced alignment via the WhisperX pipeline on a GPU worker.

    Steps: faster-whisper ASR -> wav2vec2 forced alignment using the
    language-specific XLS-R head from LANGUAGE_PROFILES.
    """

    def align(self, pcm16k: bytes, language: Language) -> AlignmentResult:
        raise NotImplementedError(
            "WhisperXAligner runs on the GPU worker; "
            "wire up faster-whisper + wav2vec2 here."
        )


class PitchScorer:
    """F0 pitch comparison: torchcrepe extraction, DTW vs. reference."""

    def contour_distance(
        self, learner_pcm16k: bytes, reference_pcm16k: bytes
    ) -> float:
        raise NotImplementedError(
            "Extract F0 with torchcrepe for both clips, normalise per "
            "speaker (z-score), and return the DTW distance."
        )


# --- Scoring -------------------------------------------------------------
# Drift handling, in order:
#   1. Score RELATIVE to a forced alignment of the canonical reference
#      recording run through the same model, so model bias cancels out.
#   2. Speaking-rate normalisation before any duration comparison.
#   3. Per-phoneme CTC confidence gating: low-confidence boundaries are
#      interpolated and flagged, never scored as learner errors.

CONFIDENCE_GATE = 0.5


def gop_from_ctc_posteriors(unit: AlignedUnit) -> float:
    """Goodness of Pronunciation for one unit from its CTC posteriors.

    Stub: production computes the log posterior ratio of the target
    unit against all competing units over the unit's frames.
    """
    return unit.ctc_confidence  # placeholder until posteriors are wired


def speaking_rate_normalise(
    units: list[AlignedUnit], reference_units: list[AlignedUnit]
) -> list[AlignedUnit]:
    """Scale unit durations by the learner/reference total-duration ratio."""
    learner_total = sum(u.end_s - u.start_s for u in units) or 1.0
    reference_total = sum(u.end_s - u.start_s for u in reference_units) or 1.0
    ratio = reference_total / learner_total
    return [
        AlignedUnit(u.label, u.start_s * ratio, u.end_s * ratio, u.ctc_confidence)
        for u in units
    ]


def score_utterance(
    learner: AlignmentResult,
    reference: AlignmentResult,
    pitch_distance: float | None = None,
) -> ScoreResult:
    """Score a learner utterance against its canonical reference alignment."""
    normalised = speaking_rate_normalise(learner.units, reference.units)
    unit_scores: list[UnitScore] = []
    scored: list[float] = []
    for unit in normalised:
        if unit.ctc_confidence < CONFIDENCE_GATE:
            unit_scores.append(UnitScore(label=unit.label, gop=None, flagged=True))
            continue
        gop = gop_from_ctc_posteriors(unit)
        scored.append(gop)
        unit_scores.append(UnitScore(label=unit.label, gop=gop))
    overall = sum(scored) / len(scored) if scored else None
    return ScoreResult(
        units=unit_scores, overall_gop=overall, pitch_dtw_distance=pitch_distance
    )
