"""Pitch contour extraction and comparison vs. the reference (Day 5).

F0 comes from torchcrepe on the GPU worker (lazy import, like WhisperX).
Comparison runs in the API process with numpy only:

1. **Per-speaker normalisation.** Each clip's voiced F0 is converted to
   semitones relative to *that speaker's own* median F0. This cancels
   register (a low male learner shadowing a high female reference is
   not "wrong") while keeping pitch range and movement, which is what
   intonation feedback is about. A z-score would also rescale range and
   hide a learner who speaks flat, so it is deliberately not used.
2. **DTW.** The normalised contours (unvoiced gaps linearly
   interpolated so the warp is continuous) are aligned with dynamic
   time warping. ``distance`` is the accumulated |semitone difference|
   along the warp path over the longer contour's length (as Day 4's
   duration DTW): one number for the whole sentence's intonation.
3. **Per-unit pitch_delta.** The learner unit's mean voiced semitone
   level minus the paired reference unit's. Units are paired by the
   forced alignment, not by the pitch warp: a warp driven by pitch
   values maps a flat contour onto the reference's mid-range frames
   and hides exactly the error it should report (seen on a real
   torchcrepe run). Positive = higher than the reference, relative to
   each speaker's own register. None when either unit has no voiced
   frames (e.g. a voiceless consonant).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

SAMPLE_RATE_HZ = 16_000
CREPE_HOP_S = 0.01  # 10 ms hop, torchcrepe's usual setting


class PitchUnavailableError(RuntimeError):
    """Raised when torchcrepe / torch is not installed in this process."""


@dataclass
class PitchContour:
    """Per-frame F0 in Hz; NaN on unvoiced frames."""

    f0_hz: np.ndarray
    frame_s: float = CREPE_HOP_S

    @property
    def voiced(self) -> np.ndarray:
        return np.isfinite(self.f0_hz) & (self.f0_hz > 0)


class PitchExtractor(Protocol):
    name: str

    def extract(self, pcm16k: bytes) -> PitchContour: ...


class CrepePitchExtractor:
    """F0 extraction with torchcrepe (Viterbi decoding, periodicity gate)."""

    name = "torchcrepe"

    def __init__(
        self,
        model: str = "tiny",
        device: str | None = None,
        fmin_hz: float = 50.0,
        fmax_hz: float = 550.0,
        periodicity_threshold: float = 0.21,
        silence_below_peak_db: float = 40.0,
    ) -> None:
        self.model = model
        self.device = device or "cpu"
        self.fmin_hz = fmin_hz
        self.fmax_hz = fmax_hz
        self.periodicity_threshold = periodicity_threshold
        # Relative to the clip's own loudest frame: an absolute dB floor
        # depends on mic gain and dropped a whole voiced clip in testing.
        self.silence_below_peak_db = silence_below_peak_db

    @staticmethod
    def available() -> bool:
        try:
            import torch  # noqa: F401, PLC0415
            import torchcrepe  # noqa: F401, PLC0415
        except Exception:
            return False
        return True

    def extract(self, pcm16k: bytes) -> PitchContour:
        samples = np.frombuffer(pcm16k, dtype="<i2").astype(np.float32) / 32768.0
        if samples.size == 0:
            return PitchContour(np.array([], dtype=np.float64), CREPE_HOP_S)
        try:
            import torch  # noqa: PLC0415
            import torchcrepe  # noqa: PLC0415
        except Exception as exc:
            raise PitchUnavailableError(
                "torchcrepe not available in this process: install torch + "
                f"torchcrepe on a GPU worker. Underlying error: {exc}"
            ) from exc

        hop = int(round(CREPE_HOP_S * SAMPLE_RATE_HZ))
        audio = torch.from_numpy(samples)[None]
        with torch.inference_mode():
            pitch, periodicity = torchcrepe.predict(
                audio,
                SAMPLE_RATE_HZ,
                hop,
                self.fmin_hz,
                self.fmax_hz,
                self.model,
                batch_size=512,
                device=self.device,
                return_periodicity=True,
            )
            periodicity = torchcrepe.filter.median(periodicity, 3)
            loudness = torchcrepe.loudness.a_weighted(audio, SAMPLE_RATE_HZ, hop)
            silent = loudness < loudness.max() - self.silence_below_peak_db
            periodicity = periodicity.masked_fill(silent, 0.0)
            pitch = torchcrepe.threshold.At(self.periodicity_threshold)(pitch, periodicity)
        f0 = pitch[0].detach().cpu().numpy().astype(np.float64)
        return PitchContour(f0, CREPE_HOP_S)


MAX_DTW_FRAMES = 400  # 4 s at the 10 ms crepe hop
TONE_SHAPE_POINTS = 10
TONE_MIN_VOICED_FRAMES = 3  # 30 ms at the 10 ms crepe hop


def normalise_per_speaker(contour: PitchContour) -> np.ndarray | None:
    """Semitones relative to the speaker's median voiced F0 (NaN when unvoiced).

    None when the clip has no voiced frames.
    """
    voiced = contour.voiced
    if not voiced.any():
        return None
    semitones = np.full(contour.f0_hz.shape, np.nan)
    semitones[voiced] = 12.0 * np.log2(contour.f0_hz[voiced])
    return semitones - np.median(semitones[voiced])


def _interpolate_unvoiced(semitones: np.ndarray) -> np.ndarray:
    voiced = np.isfinite(semitones)
    idx = np.arange(semitones.size)
    return np.interp(idx, idx[voiced], semitones[voiced])


def _dtw_input(semitones: np.ndarray) -> np.ndarray:
    """The voiced span, gaps interpolated, at most MAX_DTW_FRAMES long.

    Leading/trailing silence is cut so it is not scored as held pitch,
    and long clips are decimated so DTW stays O(MAX_DTW_FRAMES**2).
    """
    voiced = np.flatnonzero(np.isfinite(semitones))
    span = _interpolate_unvoiced(semitones[voiced[0] : voiced[-1] + 1])
    if span.size <= MAX_DTW_FRAMES:
        return span
    # Evenly spaced and inclusive of both ends, so the final pitch stays.
    return span[np.linspace(0, span.size - 1, MAX_DTW_FRAMES).round().astype(int)]


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """DTW with |a - b| cost, normalised by the longer sequence length."""
    n, m = a.size, b.size
    cost = np.abs(a[:, None] - b[None, :])
    prev = np.full(m + 1, np.inf)
    prev[0] = 0.0
    for i in range(n):
        ci = cost[i]
        # Diagonal / vertical moves vectorise; the horizontal move is a
        # running min along the row.
        base = np.minimum(prev[1:], prev[:-1]) + ci
        row = np.full(m + 1, np.inf)
        running = np.inf
        for j in range(m):
            running = min(base[j], running + ci[j])
            row[j + 1] = running
        prev = row
    return float(prev[m] / max(n, m))


@dataclass
class PitchComparison:
    distance: float  # DTW distance in semitones, see dtw_distance
    learner_semitones: np.ndarray  # NaN on unvoiced frames
    reference_semitones: np.ndarray
    frame_s: float

    def contour_payload(self) -> dict[str, float | list[float | None]]:
        def samples(values: np.ndarray) -> list[float | None]:
            return [round(float(value), 3) if np.isfinite(value) else None for value in values]

        return {
            "frame_s": self.frame_s,
            "learner": samples(self.learner_semitones),
            "reference": samples(self.reference_semitones),
        }

    def unit_delta(
        self, learner_span: tuple[float, float], reference_span: tuple[float, float]
    ) -> float | None:
        """Signed semitone gap (learner − reference) for one paired unit."""
        learner = _voiced_mean(self.learner_semitones, learner_span, self.frame_s)
        reference = _voiced_mean(self.reference_semitones, reference_span, self.frame_s)
        if learner is None or reference is None:
            return None
        return learner - reference

    def unit_shape_distance(
        self,
        learner_span: tuple[float, float],
        reference_span: tuple[float, float],
        points: int = TONE_SHAPE_POINTS,
    ) -> float | None:
        """RMS semitone gap between the two pitch shapes over one paired unit.

        Each span's contour is resampled to ``points`` and mean-centred,
        so a learner who says the right tone higher or lower than the
        reference is not penalised; only the shape (level, rise, dip,
        fall) counts. This is what scores a Mandarin tone on its final.
        None when either span has too few voiced frames.
        """
        learner = _shape(self.learner_semitones, learner_span, self.frame_s, points)
        reference = _shape(self.reference_semitones, reference_span, self.frame_s, points)
        if learner is None or reference is None:
            return None
        return float(np.sqrt(np.mean((learner - reference) ** 2)))


def _shape(
    semitones: np.ndarray, span: tuple[float, float], frame_s: float, points: int
) -> np.ndarray | None:
    a = max(int(np.floor(span[0] / frame_s + 1e-9)), 0)
    b = int(np.ceil(span[1] / frame_s - 1e-9))
    window = semitones[a:b]
    voiced = np.flatnonzero(np.isfinite(window))
    if voiced.size < TONE_MIN_VOICED_FRAMES:
        return None
    # Only the voiced stretch: a final's leading/trailing unvoiced
    # frames carry no tone.
    t = np.linspace(voiced[0], voiced[-1], points)
    shape = np.interp(t, voiced, window[voiced])
    return shape - shape.mean()


def _voiced_mean(
    semitones: np.ndarray, span: tuple[float, float], frame_s: float
) -> float | None:
    a = max(int(np.floor(span[0] / frame_s + 1e-9)), 0)
    b = int(np.ceil(span[1] / frame_s - 1e-9))
    window = semitones[a:b]
    window = window[np.isfinite(window)]
    if window.size == 0:
        return None
    return float(window.mean())


def compare_contours(
    learner: PitchContour, reference: PitchContour
) -> PitchComparison | None:
    """Normalise both contours per speaker and DTW them.

    None when either clip has no voiced frames (nothing to compare).
    """
    learner_st = normalise_per_speaker(learner)
    reference_st = normalise_per_speaker(reference)
    if learner_st is None or reference_st is None:
        return None
    return PitchComparison(
        distance=dtw_distance(_dtw_input(learner_st), _dtw_input(reference_st)),
        learner_semitones=learner_st,
        reference_semitones=reference_st,
        frame_s=learner.frame_s,
    )
