"""Goodness of Pronunciation (GOP) from CTC posteriors (Day 5).

The alignment head (wav2vec2 XLS-R, CTC) emits a log-posterior over its
token vocabulary for every ~20 ms frame. For one aligned unit we take
the frames inside its boundaries and compare two paths through them:

* the best path *constrained* to the unit's target tokens, using the
  standard CTC topology (optional blanks around and between tokens,
  repeats allowed) — what the learner was supposed to say;
* the best *unconstrained* path (the arg-max token on every frame) —
  what the model actually hears.

GOP is the per-frame average log ratio of the two, mapped to 0..1 with
``exp``: 1.0 means the target tokens are exactly what the model hears
on every frame, and lower values mean competing tokens win frames.
This is the CTC analogue of the classic Witt & Young GOP (log posterior
of the target over the best competitor, normalised by duration).

Pure numpy so it runs in the API process; the posteriors themselves are
produced on the GPU worker by the aligner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class CTCPosteriors:
    """Frame-level CTC log-posteriors for one utterance."""

    log_probs: np.ndarray  # (frames, vocab), log-softmax output of the CTC head
    frame_s: float  # seconds per frame (wav2vec2: ~0.02)
    blank_id: int = 0

    @property
    def num_frames(self) -> int:
        return int(self.log_probs.shape[0])

    def frame_span(self, start_s: float, end_s: float) -> tuple[int, int]:
        """Frame index range [a, b) covering the time span."""
        n = self.num_frames
        a = int(math.floor(start_s / self.frame_s + 1e-9))
        b = int(math.ceil(end_s / self.frame_s - 1e-9))
        a = min(max(a, 0), n)
        b = min(max(b, a), n)
        return a, b


def ctc_viterbi_log_likelihood(
    log_probs: np.ndarray, targets: Sequence[int], blank_id: int = 0
) -> float | None:
    """Log-probability of the best CTC path that emits exactly ``targets``.

    Returns None when the frames cannot fit the target sequence (too
    few frames for the tokens plus the blanks required between repeats).
    """
    frames = int(log_probs.shape[0])
    if frames == 0:
        return None
    if not targets:
        return float(log_probs[:, blank_id].sum())

    ext = [blank_id]
    for token in targets:
        ext += [int(token), blank_id]
    states = len(ext)
    ext_arr = np.asarray(ext)
    # A state may skip the preceding blank only when it is a real token
    # that differs from the token two states back.
    can_skip = np.zeros(states, dtype=bool)
    for s in range(2, states):
        can_skip[s] = ext[s] != blank_id and ext[s] != ext[s - 2]

    neg_inf = -np.inf
    alpha = np.full(states, neg_inf)
    alpha[0] = log_probs[0, blank_id]
    alpha[1] = log_probs[0, ext[1]]
    for t in range(1, frames):
        stay = alpha
        step = np.concatenate(([neg_inf], alpha[:-1]))
        skip = np.where(can_skip, np.concatenate(([neg_inf, neg_inf], alpha[:-2])), neg_inf)
        alpha = np.maximum(np.maximum(stay, step), skip) + log_probs[t, ext_arr]

    best = max(alpha[-1], alpha[-2])
    if not np.isfinite(best):
        return None
    return float(best)


def gop_score(
    posteriors: CTCPosteriors,
    token_ids: Sequence[int],
    start_s: float,
    end_s: float,
) -> float | None:
    """GOP in 0..1 for one unit, or None when it cannot be computed.

    None means the unit's boundaries do not leave room for its own
    target tokens — a boundary problem, which callers gate (flag)
    rather than score as a learner error.
    """
    if not token_ids:
        return None
    a, b = posteriors.frame_span(start_s, end_s)
    if b <= a:
        return None
    window = posteriors.log_probs[a:b]
    constrained = ctc_viterbi_log_likelihood(window, token_ids, posteriors.blank_id)
    if constrained is None:
        return None
    free = float(window.max(axis=1).sum())
    log_ratio = min(0.0, (constrained - free) / (b - a))
    return float(math.exp(log_ratio))
