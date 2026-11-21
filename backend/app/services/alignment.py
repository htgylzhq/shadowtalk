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
import unicodedata
import wave
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Protocol

from app.services.gop import CTCPosteriors, gop_score
from app.services.pitch import PitchContour, compare_contours


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
    # Target token ids of this unit in the CTC head's vocabulary, used
    # for GOP. None when the aligner does not provide them.
    token_ids: list[int] | None = None


@dataclass
class AlignmentResult:
    language: Language
    transcript: str
    units: list[AlignedUnit] = field(default_factory=list)
    # Frame-level CTC log-posteriors of the alignment head, for GOP.
    # None when the aligner does not expose them.
    posteriors: CTCPosteriors | None = None


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
    # Signed mean semitone difference vs. the reference pitch contour,
    # each side relative to its own speaker's median F0. None when
    # there is no pitch comparison, the unit is gated, or it is unvoiced.
    pitch_delta: float | None = None
    # "ctc_posteriors" (real GOP) or "ctc_confidence" (fallback when the
    # aligner exposes no posteriors). None when the unit is gated.
    gop_source: str | None = None


@dataclass
class ScoreResult:
    units: list[UnitScore]
    overall_gop: float | None
    pitch_dtw_distance: float | None
    # DTW distance between the rate-normalised learner unit durations
    # and the reference unit durations (seconds per unit, averaged).
    # None when there is no reference to compare against.
    duration_dtw_distance: float | None = None
    # "ctc_posteriors", "ctc_confidence", "mixed", or None (nothing scored).
    gop_source: str | None = None
    # Reference unit labels the learner never said (no learner unit paired).
    skipped: list[str] = field(default_factory=list)


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
            posteriors = self._ctc_posteriors(pcm16k)
            dictionary = self._align_metadata["dictionary"]  # type: ignore[index]
            units: list[AlignedUnit] = []
            for segment in aligned.get("segments", []):
                for word in segment.get("words", []):
                    if "start" not in word or "end" not in word:
                        continue
                    label = str(word["word"]).strip()
                    units.append(
                        AlignedUnit(
                            label=label,
                            start_s=float(word["start"]),
                            end_s=float(word["end"]),
                            ctc_confidence=float(word.get("score", 1.0)),
                            token_ids=_token_ids(label, dictionary) or None,
                        )
                    )
            return AlignmentResult(
                language=language,
                transcript=transcript,
                units=units,
                posteriors=posteriors,
            )
        finally:
            wav_path.unlink(missing_ok=True)

    def _ctc_posteriors(self, pcm16k: bytes) -> CTCPosteriors:
        """Run the alignment head once more to keep its frame posteriors.

        ``whisperx.align`` computes these internally but does not return
        them, so GOP re-runs the same wav2vec2 model (one forward pass).
        Not yet exercised against a real model — see ROADMAP Day 5.
        """
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        samples = np.frombuffer(pcm16k, dtype="<i2").astype(np.float32) / 32768.0
        waveform = torch.from_numpy(samples)[None].to(self.device)
        metadata = self._align_metadata
        with torch.inference_mode():
            if metadata["type"] == "torchaudio":  # type: ignore[index]
                emissions, _ = self._align_model(waveform)  # type: ignore[misc]
            else:
                emissions = self._align_model(waveform).logits  # type: ignore[misc, union-attr]
            log_probs = torch.log_softmax(emissions, dim=-1)[0].cpu().numpy()
        dictionary = metadata["dictionary"]  # type: ignore[index]
        blank_id = next(
            (code for char, code in dictionary.items() if char in ("[pad]", "<pad>")), 0
        )
        duration_s = samples.size / 16_000
        return CTCPosteriors(
            log_probs=log_probs,
            frame_s=duration_s / max(log_probs.shape[0], 1),
            blank_id=blank_id,
        )


def _token_ids(label: str, dictionary: dict[str, int]) -> list[int]:
    """Map a unit label to CTC vocabulary ids, as whisperx does (chars, lowercased)."""
    ids: list[int] = []
    for char in label:
        if char in dictionary:
            ids.append(dictionary[char])
        elif char.lower() in dictionary:
            ids.append(dictionary[char.lower()])
    return ids


# --- Scoring -------------------------------------------------------------
# Drift handling, in order:
#   1. Score RELATIVE to a forced alignment of the canonical reference
#      recording run through the same model, so model bias cancels out.
#   2. Speaking-rate normalisation before any duration comparison.
#   3. Per-phoneme CTC confidence gating: low-confidence boundaries are
#      interpolated and flagged, never scored as learner errors.

CONFIDENCE_GATE = 0.5


def gop_from_ctc_posteriors(
    unit: AlignedUnit, posteriors: CTCPosteriors | None = None
) -> tuple[float | None, str]:
    """Goodness of Pronunciation for one unit, and where it came from.

    With frame posteriors and target token ids (the real aligner), GOP
    is the CTC posterior ratio of the target tokens against the best
    competing path over the unit's frames (see services/gop.py); it is
    None when the unit's frames cannot fit its own tokens, which the
    caller gates as a boundary problem. Without posteriors (an aligner
    that does not expose them), it falls back to the unit's CTC
    confidence and says so via the "ctc_confidence" source.
    """
    if posteriors is not None and unit.token_ids:
        return (
            gop_score(posteriors, unit.token_ids, unit.start_s, unit.end_s),
            "ctc_posteriors",
        )
    return unit.ctc_confidence, "ctc_confidence"


def speaking_rate_normalise(
    units: list[AlignedUnit],
    reference_units: list[AlignedUnit],
    pairs: list[int | None] | None = None,
) -> list[AlignedUnit]:
    """Scale unit durations by the learner/reference total-duration ratio.

    Day 4: the reference is now wired in (see services/reference.py
    and main._score_session). With no reference, return the learner
    units unchanged instead of scaling by a bogus ratio.
    """
    if not units or not reference_units:
        return list(units)
    # With pairs, only paired units set the rate: a skipped or extra
    # word is not tempo.
    if pairs is not None:
        matched = [(u, reference_units[j]) for u, j in zip(units, pairs) if j is not None]
        units_used, refs_used = [u for u, _ in matched], [r for _, r in matched]
    else:
        units_used, refs_used = units, reference_units
    learner_total = sum(u.end_s - u.start_s for u in units_used) or 1.0
    reference_total = sum(u.end_s - u.start_s for u in refs_used) or 1.0
    ratio = reference_total / learner_total
    return [
        replace(u, start_s=u.start_s * ratio, end_s=u.end_s * ratio) for u in units
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


def _unit_key(label: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKC", label).casefold()
        if unicodedata.category(c)[0] in "LN"
    )


def pair_units(
    learner: list[AlignedUnit], reference: list[AlignedUnit]
) -> list[int | None]:
    """For each learner unit, the index of its reference unit, or None.

    Labels are sequence-aligned, so a skipped or inserted unit does not
    shift every later pair, and a substituted unit still meets the
    reference unit it replaced.
    """
    a = [_unit_key(u.label) for u in learner]
    b = [_unit_key(u.label) for u in reference]
    # Edit-distance alignment: match 0, substitution 1, skip/insert 1.
    # A substitution beats a skip + insert, so "no no" vs "oh no" pairs
    # the wrong first word with "oh" instead of reporting "oh" skipped.
    n, m = len(a), len(b)
    cost = [[i + j if i == 0 or j == 0 else 0 for j in range(m + 1)] for i in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(
                cost[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
            )
    pairs: list[int | None] = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        if cost[i][j] == cost[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            pairs[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif cost[i][j] == cost[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return pairs


def score_utterance(
    learner: AlignmentResult,
    reference: AlignmentResult,
    pitch_distance: float | None = None,
    learner_pitch: PitchContour | None = None,
    reference_pitch: PitchContour | None = None,
) -> ScoreResult:
    """Score a learner utterance against its canonical reference alignment.

    Relative scoring (Day 4): units are paired with the reference units
    by label sequence alignment (see pair_units), after speaking-rate normalisation. A unit is
    gated — flagged, gop None, never scored as a learner error — when
    *its own* CTC confidence is below the gate, *or* when the paired
    reference unit is below the gate: a boundary the model cannot
    place reliably on the reference cannot be blamed on the learner
    (model bias cancels). ``duration_ratio`` reports the rhythm
    comparison separately from pronunciation, so a slow-but-correct
    unit is not misread as mispronounced.

    Day 5: GOP comes from the learner's CTC posteriors when the aligner
    provides them (``gop_source`` "ctc_posteriors"), and a unit whose
    frames cannot fit its own target tokens is gated like a
    low-confidence one. With both pitch contours, the contours are
    normalised per speaker and DTW-aligned: ``pitch_dtw_distance`` is
    the mean semitone gap along the warp and each scored unit gets a
    signed ``pitch_delta``. Gated units never get a GOP or pitch_delta.
    """
    pairs = pair_units(learner.units, reference.units)
    normalised = speaking_rate_normalise(learner.units, reference.units, pairs)
    pitch = None
    if learner_pitch is not None and reference_pitch is not None:
        pitch = compare_contours(learner_pitch, reference_pitch)
        if pitch is not None:
            pitch_distance = pitch.distance
    unit_scores: list[UnitScore] = []
    scored: list[float] = []
    sources: set[str] = set()
    for i, unit in enumerate(normalised):
        # GOP and pitch read the learner's own frames, so they use the
        # original (not rate-normalised) unit boundaries.
        original = learner.units[i]
        ref = reference.units[pairs[i]] if pairs[i] is not None else None
        ref_duration = (ref.end_s - ref.start_s) if ref else None
        duration_ratio = None
        if ref_duration:
            duration_ratio = (unit.end_s - unit.start_s) / ref_duration
        ref_conf = ref.ctc_confidence if ref else None
        gated = unit.ctc_confidence < CONFIDENCE_GATE or (
            ref_conf is not None and ref_conf < CONFIDENCE_GATE
        )
        # Score what the learner should have said: the reference unit's
        # tokens over the learner's frames, so a confident substitution
        # does not grade itself.
        target = original
        if ref is not None and ref.token_ids:
            target = replace(original, token_ids=ref.token_ids)
        gop, source = (None, None) if gated else gop_from_ctc_posteriors(
            target, learner.posteriors
        )
        if gop is None:
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
        scored.append(gop)
        sources.add(source)
        unit_scores.append(
            UnitScore(
                label=unit.label,
                gop=gop,
                duration_ratio=duration_ratio,
                reference_confidence=ref_conf,
                pitch_delta=(
                    pitch.unit_delta(
                        (original.start_s, original.end_s), (ref.start_s, ref.end_s)
                    )
                    if pitch and ref
                    else None
                ),
                gop_source=source,
            )
        )
    overall = sum(scored) / len(scored) if scored else None
    duration_dtw = None
    if reference.units:
        duration_dtw = dtw_duration_distance(
            [u.end_s - u.start_s for u in normalised],
            [u.end_s - u.start_s for u in reference.units],
        )
    gop_source = None
    if sources:
        gop_source = sources.pop() if len(sources) == 1 else "mixed"
    paired = {j for j in pairs if j is not None}
    return ScoreResult(
        units=unit_scores,
        overall_gop=overall,
        skipped=[u.label for j, u in enumerate(reference.units) if j not in paired],
        pitch_dtw_distance=pitch_distance,
        duration_dtw_distance=duration_dtw,
        gop_source=gop_source,
    )
