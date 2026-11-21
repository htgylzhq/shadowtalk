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

import subprocess
import tempfile
import unicodedata
import wave
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Protocol

import numpy as np

from app.services.gop import CTCPosteriors, ctc_forced_align, ctc_forced_align_words, gop_score
from app.services.pitch import PitchComparison, PitchContour, compare_contours


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
    # Lesson word the unit belongs to (lesson-text G2P); None otherwise.
    word: str | None = None
    # Position of that word in the lesson text; None without lesson text.
    word_index: int | None = None
    # Mandarin lexical tone (1-4, 5 = neutral) on a final; None otherwise.
    tone: int | None = None


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
    # Lesson word of the paired reference unit (from lesson-text G2P).
    word: str | None = None
    # Signed mean semitone difference vs. the reference pitch contour,
    # each side relative to its own speaker's median F0. None when
    # there is no pitch comparison, the unit is gated, or it is unvoiced.
    pitch_delta: float | None = None
    # "ctc_posteriors" (real GOP) or "ctc_confidence" (fallback when the
    # aligner exposes no posteriors). None when the unit is gated.
    gop_source: str | None = None
    # Mandarin finals: the reference's lexical tone (None when unknown)
    # and the RMS semitone gap between the learner's and reference's
    # pitch shapes over the final, each mean-centred so only the contour
    # counts. None off Mandarin finals or without usable pitch.
    tone: int | None = None
    tone_distance: float | None = None


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


PHONEME_MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
WAV2VEC2_FRAME_S = 320 / 16_000  # conv feature-extractor stride
WAV2VEC2_MIN_SAMPLES = 400  # receptive field of one output frame


def units_from_ctc(
    log_probs: np.ndarray,
    frame_s: float,
    blank_id: int,
    id_to_token: dict[int, str],
) -> list[AlignedUnit]:
    """Greedy CTC decode into one unit per emitted token.

    A unit runs from its first frame to the next unit's first frame (the
    blanks after a phoneme belong to it); the last unit ends after its
    own frames. ``ctc_confidence`` is the mean top posterior over the
    token's own frames.
    """
    best = log_probs.argmax(axis=1)
    top = np.exp(log_probs.max(axis=1))
    runs: list[list[int]] = []  # [token, first frame, last frame]
    prev = blank_id
    for t, k in enumerate(best.tolist()):
        if k != blank_id:
            if k == prev:
                runs[-1][2] = t
            else:
                runs.append([k, t, t])
        prev = k
    units = []
    for n, (k, first, last) in enumerate(runs):
        end = runs[n + 1][1] if n + 1 < len(runs) else last + 1
        units.append(
            AlignedUnit(
                label=id_to_token[k],
                start_s=first * frame_s,
                end_s=end * frame_s,
                ctc_confidence=float(top[first : last + 1].mean()),
                token_ids=[k],
            )
        )
    return units


class Wav2Vec2PhonemeAligner:
    """Phoneme units from a multilingual wav2vec2 IPA CTC head.

    Decodes the phonemes the speaker actually produced, with frame
    times, straight from the CTC posteriors, so no transcript or G2P is
    needed: the reference recording is decoded the same way and
    ``pair_units`` matches the two sequences. Verified on CPU with
    English. Mandarin decodes are grouped into initials and toned finals
    (app.services.mandarin), run on CPU with synthetic espeak speech
    only; Japanese still needs mora grouping.
    """

    name = "wav2vec2-phoneme"

    def __init__(self, model_name: str = PHONEME_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or "cpu"
        self._model: object | None = None
        self._processor: object | None = None

    @staticmethod
    def available() -> bool:
        try:
            import torch  # noqa: F401, PLC0415
            import transformers  # noqa: F401, PLC0415
        except Exception:
            return False
        return True

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import (  # noqa: PLC0415
                AutoModelForCTC,
                AutoProcessor,
            )
        except Exception as exc:
            raise AlignmentUnavailableError(
                f"transformers + torch are needed for phoneme alignment: {exc}"
            ) from exc
        # Audio-only use: never phonemize text, so phonemizer / espeak-ng
        # are not needed when the tokenizer loads.
        self._processor = AutoProcessor.from_pretrained(self.model_name, do_phonemize=False)
        self._model = AutoModelForCTC.from_pretrained(self.model_name).to(self.device).eval()

    def align(self, pcm16k: bytes, language: Language) -> AlignmentResult:
        samples = np.frombuffer(pcm16k, dtype="<i2").astype(np.float32) / 32768.0
        if samples.size < WAV2VEC2_MIN_SAMPLES:
            return AlignmentResult(language=language, transcript="", units=[])
        return self._decoded(*self._posteriors(samples), language)

    def _decoded(
        self, log_probs: np.ndarray, blank_id: int, vocab: dict[str, int], language: Language
    ) -> AlignmentResult:
        id_to_token = {i: t for t, i in vocab.items()}
        units = units_from_ctc(log_probs, WAV2VEC2_FRAME_S, blank_id, id_to_token)
        if language is Language.MANDARIN:
            from app.services.mandarin import group_initial_final  # noqa: PLC0415

            units = group_initial_final(units)
        return AlignmentResult(
            language=language,
            transcript=" ".join(u.label for u in units),
            units=units,
            posteriors=CTCPosteriors(log_probs, WAV2VEC2_FRAME_S, blank_id),
        )

    def _posteriors(self, samples: np.ndarray) -> tuple[np.ndarray, int, dict[str, int]]:
        self._load()
        import torch  # noqa: PLC0415

        processor, model = self._processor, self._model
        inputs = processor(samples, sampling_rate=16_000, return_tensors="pt")  # type: ignore[operator]
        with torch.inference_mode():
            logits = model(inputs.input_values.to(self.device)).logits[0]  # type: ignore[operator]
            log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
        tokenizer = processor.tokenizer  # type: ignore[attr-defined]
        return log_probs, tokenizer.pad_token_id, tokenizer.get_vocab()

    def align_text(self, pcm16k: bytes, language: Language, text: str) -> AlignmentResult:
        """Force-align the lesson text's phonemes (espeak-ng G2P) to a recording.

        Each unit is one intended phoneme tagged with its lesson word, so
        the reference targets come from the text instead of a decode of
        the recording. Falls back to ``align`` when G2P yields nothing or
        the phonemes do not fit the audio.
        """
        from app.services.g2p import lesson_units  # noqa: PLC0415

        samples = np.frombuffer(pcm16k, dtype="<i2").astype(np.float32) / 32768.0
        if samples.size < WAV2VEC2_MIN_SAMPLES:
            return AlignmentResult(language=language, transcript="", units=[])
        log_probs, blank_id, vocab = self._posteriors(samples)
        try:
            targets = lesson_units(text, language, vocab)
        except (OSError, subprocess.SubprocessError):
            targets = []  # espeak-ng missing or failed: decode the recording
        flat = [t for target in targets for t in target.token_ids]
        spans = ctc_forced_align(log_probs, flat, blank_id) if flat else None
        if spans is None:
            return self._decoded(log_probs, blank_id, vocab, language)
        probs = np.exp(log_probs)
        units = []
        k = 0
        for target in targets:
            n = len(target.token_ids)
            first, last = spans[k][0], spans[k + n - 1][1]
            k += n
            end = spans[k][0] if k < len(spans) else last + 1
            confidence = np.mean(
                [probs[a : b + 1, t].mean() for t, (a, b) in zip(target.token_ids, spans[k - n : k])]
            )
            units.append(
                AlignedUnit(
                    label=target.label,
                    start_s=first * WAV2VEC2_FRAME_S,
                    end_s=end * WAV2VEC2_FRAME_S,
                    ctc_confidence=float(confidence),
                    token_ids=list(target.token_ids),
                    word=target.word,
                    word_index=target.word_index,
                    tone=target.tone,
                )
            )
        return AlignmentResult(
            language=language,
            transcript=text,
            units=units,
            posteriors=CTCPosteriors(log_probs, WAV2VEC2_FRAME_S, blank_id),
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
        if unicodedata.category(c)[0] in "LMN"  # keep IPA diacritics (n̩ ≠ n)
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
    # Equal-cost alignments are broken by the most exact matches, so
    # "i um like tea" vs "i like hot tea" keeps "like" with "like".
    n, m = len(a), len(b)
    # cost[i][j] = (edits, -exact matches), compared lexicographically.
    cost = [[(i + j, 0) for j in range(m + 1)] for i in range(n + 1)]

    def diagonal(i: int, j: int) -> tuple[int, int]:
        edits, neg_matches = cost[i - 1][j - 1]
        same = a[i - 1] == b[j - 1]
        return (edits + (not same), neg_matches - same)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            up, left = cost[i - 1][j], cost[i][j - 1]
            cost[i][j] = min(diagonal(i, j), (up[0] + 1, up[1]), (left[0] + 1, left[1]))
    pairs: list[int | None] = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        if cost[i][j] == diagonal(i, j):
            pairs[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif cost[i][j] == (cost[i - 1][j][0] + 1, cost[i - 1][j][1]):
            i -= 1
        else:
            j -= 1
    return pairs


def force_to_reference(
    learner: AlignmentResult, reference: AlignmentResult
) -> tuple[AlignmentResult, list[int | None]] | None:
    """Force-align the reference's lesson phonemes onto the learner's posteriors.

    Only for a reference built from lesson text (every unit has a
    ``word`` and ``token_ids``) and a learner aligner that keeps its CTC
    posteriors. Each learner unit is then the intended phoneme over the
    learner's own frames, so a phoneme the free decode would miss is
    still scored. Whole lesson words may be left out of the alignment
    (ctc_forced_align_words), so a word the learner skipped is reported
    as skipped instead of being stretched over the next word's audio; a
    kept word whose frames the model hears only as blank is dropped too.

    Returns the learner result and its pairs into ``reference.units``,
    or None to fall back to free-decoded units and ``pair_units``.
    """
    post = learner.posteriors
    refs = reference.units
    if post is None or not refs or any(u.word is None or not u.token_ids for u in refs):
        return None
    # A word is a run of consecutive units with the same lesson word
    # (and word position, when the reference records it).
    groups: list[list[int]] = []
    for j, u in enumerate(refs):
        prev = refs[groups[-1][-1]] if groups else None
        if prev is not None and (prev.word, prev.word_index) == (u.word, u.word_index):
            groups[-1].append(j)
        else:
            groups.append([j])
    aligned = ctc_forced_align_words(
        post.log_probs,
        [[t for j in g for t in refs[j].token_ids or []] for g in groups],
        post.blank_id,
    )
    if aligned is None:
        return None
    top = post.log_probs.argmax(axis=-1)
    top_p = np.exp(post.log_probs.max(axis=-1))
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for g, spans in zip(groups, aligned):
        if spans is None:
            continue  # the lattice skipped the whole word
        k = 0
        for j in g:
            n = len(refs[j].token_ids or [])
            first[j], last[j] = spans[k][0], spans[k + n - 1][1]
            k += n
        # Forced onto frames the model hears as blank only: not said either.
        if not any((top[first[j] : last[j] + 1] != post.blank_id).any() for j in g):
            for j in g:
                del first[j], last[j]
    kept = sorted(first)
    units: list[AlignedUnit] = []
    pairs: list[int | None] = []
    for n, j in enumerate(kept):
        u = refs[j]
        end = first[kept[n + 1]] if n + 1 < len(kept) else last[j] + 1
        units.append(
            AlignedUnit(
                label=u.label,
                start_s=first[j] * post.frame_s,
                end_s=end * post.frame_s,
                ctc_confidence=float(top_p[first[j] : last[j] + 1].mean()),
                token_ids=u.token_ids,
                word=u.word,
                word_index=u.word_index,
                tone=u.tone,
            )
        )
        pairs.append(j)
    forced = AlignmentResult(
        language=learner.language,
        transcript=reference.transcript,
        units=units,
        posteriors=post,
    )
    return forced, pairs


def _tone_fields(
    language: Language,
    unit: AlignedUnit,
    ref: AlignedUnit | None,
    pitch: PitchComparison | None,
) -> dict[str, float | int | None]:
    """``tone`` / ``tone_distance`` for a scored Mandarin final."""
    if language is not Language.MANDARIN or ref is None:
        return {}
    from app.services.mandarin import is_final  # noqa: PLC0415

    if not is_final(ref.label):
        return {}
    distance = None
    if pitch is not None:
        distance = pitch.unit_shape_distance(
            (unit.start_s, unit.end_s), (ref.start_s, ref.end_s)
        )
    return {"tone": ref.tone, "tone_distance": distance}


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

    With a lesson-text reference, the learner units are the lesson
    phonemes force-aligned to the learner's frames (see
    force_to_reference) instead of a free decode.

    Day 5: GOP comes from the learner's CTC posteriors when the aligner
    provides them (``gop_source`` "ctc_posteriors"), and a unit whose
    frames cannot fit its own target tokens is gated like a
    low-confidence one. With both pitch contours, the contours are
    normalised per speaker and DTW-aligned: ``pitch_dtw_distance`` is
    the mean semitone gap along the warp and each scored unit gets a
    signed ``pitch_delta``. Gated units never get a GOP or pitch_delta.
    """
    forced = force_to_reference(learner, reference)
    if forced is not None:
        learner, pairs = forced
    else:
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
                    word=ref.word if ref is not None and ref.word else unit.word,
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
                word=ref.word if ref is not None and ref.word else unit.word,
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
                **_tone_fields(learner.language, original, ref, pitch),
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
