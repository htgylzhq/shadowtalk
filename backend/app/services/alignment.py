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

import tempfile
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol


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


class AlignmentUnavailableError(RuntimeError):
    """Raised when the WhisperX stack is not installed / not loadable.

    The API process runs without the heavy ML stack in local dev (see
    requirements.txt: faster-whisper / whisperx are GPU-worker deps).
    Callers must surface this as an explicit "unavailable" status,
    never as silently empty alignments.
    """


class Aligner(Protocol):
    """Anything that can turn 16 kHz mono PCM into aligned units."""

    name: str

    def align(self, pcm16k: bytes, language: Language) -> AlignmentResult: ...


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
    # Learner duration / reference duration after speaking-rate
    # normalisation (1.0 = same rhythm as the reference). None when
    # there is no paired reference unit (Day 4: no reference yet).
    duration_ratio: float | None = None
    reference_confidence: float | None = None


@dataclass
class ScoreResult:
    units: list[UnitScore]
    overall_gop: float | None
    pitch_dtw_distance: float | None
    # DTW distance between the rate-normalised learner unit durations
    # and the reference unit durations (seconds per unit, averaged).
    # None when there is no reference to compare against.
    duration_dtw_distance: float | None = None


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

    Steps: faster-whisper ASR (word timestamps on) -> whisperx
    ``load_align_model`` with the language-specific XLS-R head from
    LANGUAGE_PROFILES -> ``whisperx.align`` -> word-level segments.

    Honest scope note (Day 3, 2026-11-20): WhisperX returns
    **word-level** timestamps. Splitting words into per-phoneme /
    per-mora units needs the wav2vec2 phoneme CTC head, which is Day 4
    work — so ``align()`` labels units with words for now, and the
    roadmap says so. Do not present word timestamps as phoneme
    timestamps.

    Heavy imports live inside :meth:`_load` so the API process starts
    without torch / faster-whisper / whisperx installed. When they are
    missing, :meth:`align` raises :class:`AlignmentUnavailableError`.
    """

    name = "whisperx"

    def __init__(
        self,
        model_size: str = "tiny",
        device: str | None = None,
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.device = device or "cpu"
        self.compute_type = compute_type
        self._asr_model: object | None = None
        self._align_model: object | None = None
        self._align_metadata: object | None = None

    @staticmethod
    def available() -> bool:
        """True when the WhisperX stack imports in this process."""
        try:
            import faster_whisper  # noqa: F401, PLC0415
            import whisperx  # noqa: F401, PLC0415
        except Exception:
            return False
        return True

    def _load(self) -> None:
        if self._asr_model is not None:
            return
        try:
            from faster_whisper import WhisperModel  # noqa: PLC0415
            import whisperx  # noqa: PLC0415
        except Exception as exc:  # missing deps, no GPU, bad install, ...
            raise AlignmentUnavailableError(
                "WhisperX stack not available in this process: "
                "install faster-whisper + whisperx (+ torch) on a GPU "
                f"worker. Underlying error: {exc}"
            ) from exc
        self._asr_model = WhisperModel(
            self.model_size, device=self.device, compute_type=self.compute_type
        )
        self._whisperx = whisperx  # type: ignore[attr-defined]

    def align(self, pcm16k: bytes, language: Language) -> AlignmentResult:
        if not pcm16k:
            return AlignmentResult(language=language, transcript="", units=[])
        self._load()
        assert self._asr_model is not None
        whisperx = self._whisperx  # type: ignore[attr-defined]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)
        try:
            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16_000)
                wav.writeframes(pcm16k)

            segments, _info = self._asr_model.transcribe(  # type: ignore[union-attr]
                str(wav_path), language=language.value, word_timestamps=True
            )
            segments = list(segments)
            transcript = " ".join(s.text.strip() for s in segments).strip()

            if self._align_model is None:
                self._align_model, self._align_metadata = whisperx.load_align_model(
                    language_code=language.value, device=self.device
                )
            aligned = whisperx.align(
                [
                    {"text": s.text, "start": s.start, "end": s.end}
                    for s in segments
                ],
                self._align_model,
                self._align_metadata,
                str(wav_path),
                self.device,
            )
            units: list[AlignedUnit] = []
            for segment in aligned.get("segments", []):
                for word in segment.get("words", []):
                    if "start" not in word or "end" not in word:
                        continue
                    units.append(
                        AlignedUnit(
                            label=str(word["word"]).strip(),
                            start_s=float(word["start"]),
                            end_s=float(word["end"]),
                            ctc_confidence=float(word.get("score", 1.0)),
                        )
                    )
            return AlignmentResult(language=language, transcript=transcript, units=units)
        finally:
            wav_path.unlink(missing_ok=True)


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
    """Scale unit durations by the learner/reference total-duration ratio.

    Day 4: the reference is now wired in (see services/reference.py
    and main._score_session). With no reference, return the learner
    units unchanged instead of scaling by a bogus ratio.
    """
    if not units or not reference_units:
        return list(units)
    learner_total = sum(u.end_s - u.start_s for u in units) or 1.0
    reference_total = sum(u.end_s - u.start_s for u in reference_units) or 1.0
    ratio = reference_total / learner_total
    return [
        AlignedUnit(u.label, u.start_s * ratio, u.end_s * ratio, u.ctc_confidence)
        for u in units
    ]


def dtw_duration_distance(
    learner_durations: list[float], reference_durations: list[float]
) -> float | None:
    """DTW distance between two unit-duration sequences.

    Classic dynamic time warping with |a - b| as the per-step cost,
    normalised by the longer sequence length so the result is in
    seconds-per-unit and comparable across sentences. Returns None
    when either sequence is empty. Used for rhythm: after speaking-
    rate normalisation the totals already match, so a large distance
    means the learner stretched some units and rushed others relative
    to the reference, even at the same overall speed.
    """
    if not learner_durations or not reference_durations:
        return None
    n, m = len(learner_durations), len(reference_durations)
    inf = float("inf")
    prev = [inf] * (m + 1)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = [inf] * (m + 1)
        for j in range(1, m + 1):
            cost = abs(learner_durations[i - 1] - reference_durations[j - 1])
            cur[j] = cost + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return prev[m] / max(n, m)


def score_utterance(
    learner: AlignmentResult,
    reference: AlignmentResult,
    pitch_distance: float | None = None,
) -> ScoreResult:
    """Score a learner utterance against its canonical reference alignment.

    Relative scoring (Day 4): units are paired by position with the
    reference units, after speaking-rate normalisation. A unit is
    gated — flagged, gop None, never scored as a learner error — when
    *its own* CTC confidence is below the gate, *or* when the paired
    reference unit is below the gate: a boundary the model cannot
    place reliably on the reference cannot be blamed on the learner
    (model bias cancels). ``duration_ratio`` reports the rhythm
    comparison separately from pronunciation, so a slow-but-correct
    unit is not misread as mispronounced.
    """
    normalised = speaking_rate_normalise(learner.units, reference.units)
    unit_scores: list[UnitScore] = []
    scored: list[float] = []
    for i, unit in enumerate(normalised):
        ref = reference.units[i] if i < len(reference.units) else None
        ref_duration = (ref.end_s - ref.start_s) if ref else None
        duration_ratio = None
        if ref_duration:
            duration_ratio = (unit.end_s - unit.start_s) / ref_duration
        ref_conf = ref.ctc_confidence if ref else None
        if unit.ctc_confidence < CONFIDENCE_GATE or (
            ref_conf is not None and ref_conf < CONFIDENCE_GATE
        ):
            unit_scores.append(
                UnitScore(
                    label=unit.label,
                    gop=None,
                    flagged=True,
                    duration_ratio=duration_ratio,
                    reference_confidence=ref_conf,
                )
            )
            continue
        gop = gop_from_ctc_posteriors(unit)
        scored.append(gop)
        unit_scores.append(
            UnitScore(
                label=unit.label,
                gop=gop,
                duration_ratio=duration_ratio,
                reference_confidence=ref_conf,
            )
        )
    overall = sum(scored) / len(scored) if scored else None
    duration_dtw = None
    if reference.units:
        duration_dtw = dtw_duration_distance(
            [u.end_s - u.start_s for u in normalised],
            [u.end_s - u.start_s for u in reference.units],
        )
    return ScoreResult(
        units=unit_scores,
        overall_gop=overall,
        pitch_dtw_distance=pitch_distance,
        duration_dtw_distance=duration_dtw,
    )
