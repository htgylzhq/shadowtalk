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


def ctc_forced_align(
    log_probs: np.ndarray, targets: Sequence[int], blank_id: int = 0
) -> list[tuple[int, int]] | None:
    """Frame span ``(first, last)`` of each target token on the best CTC path.

    Same lattice as ``ctc_viterbi_log_likelihood``, with backpointers.
    Returns None when the frames cannot fit the targets.
    """
    frames = int(log_probs.shape[0])
    if frames == 0 or not targets:
        return None
    ext = [blank_id]
    for token in targets:
        ext += [int(token), blank_id]
    states = len(ext)
    ext_arr = np.asarray(ext)
    can_skip = np.zeros(states, dtype=bool)
    for s in range(2, states):
        can_skip[s] = ext[s] != blank_id and ext[s] != ext[s - 2]

    neg_inf = -np.inf
    alpha = np.full(states, neg_inf)
    alpha[0] = log_probs[0, blank_id]
    alpha[1] = log_probs[0, ext[1]]
    back = np.zeros((frames, states), dtype=np.int8)  # 0 stay, 1 step, 2 skip
    for t in range(1, frames):
        step = np.concatenate(([neg_inf], alpha[:-1]))
        skip = np.where(can_skip, np.concatenate(([neg_inf, neg_inf], alpha[:-2])), neg_inf)
        moves = np.stack([alpha, step, skip])
        back[t] = moves.argmax(axis=0)
        alpha = moves.max(axis=0) + log_probs[t, ext_arr]

    state = states - 1 if alpha[-1] >= alpha[-2] else states - 2
    if not np.isfinite(alpha[state]):
        return None
    path = np.empty(frames, dtype=int)
    for t in range(frames - 1, -1, -1):
        path[t] = state
        state -= int(back[t, state])
    spans: list[tuple[int, int]] = []
    for k in range(len(targets)):
        hits = np.flatnonzero(path == 2 * k + 1)
        spans.append((int(hits[0]), int(hits[-1])))
    return spans


SKIP_WORD_LOG_PENALTY = -3.0


def ctc_forced_align_words(
    log_probs: np.ndarray,
    words: Sequence[Sequence[int]],
    blank_id: int = 0,
    skip_log_penalty: float = SKIP_WORD_LOG_PENALTY,
) -> list[list[tuple[int, int]] | None] | None:
    """Forced alignment where whole words may be left out.

    The lattice of ``ctc_forced_align`` plus jumps that leave out one or
    more words, each costing ``skip_log_penalty``, so a word the speaker
    never said is skipped instead of being stretched over a neighbouring
    word's audio. Words may also be left out at the start or end.
    Returns, per word, the ``(first, last)`` frame span of each target
    token, or None for a skipped word; None overall when nothing fits.
    """
    frames = int(log_probs.shape[0])
    if frames == 0 or not words or not any(words):
        return None
    ext = [blank_id]
    blank_before: list[int] = []
    token_state: list[list[int]] = []
    for word in words:
        blank_before.append(len(ext) - 1)
        states_w = []
        for token in word:
            states_w.append(len(ext))
            ext += [int(token), blank_id]
        token_state.append(states_w)
    states = len(ext)
    blank_before.append(states - 1)
    ext_arr = np.asarray(ext)
    can_skip = np.zeros(states, dtype=bool)
    for s in range(2, states):
        can_skip[s] = ext[s] != blank_id and ext[s] != ext[s - 2]
    n_words = len(words)

    def entry(w: int) -> list[int]:
        return [blank_before[w]] + token_state[w][:1]

    def exit_(u: int) -> list[int]:
        return [blank_before[u + 1]] + token_state[u][-1:] if u >= 0 else [0]

    # Leaving out words u+1 .. v-1 jumps from the end of word u (its last
    # token or the blank after it) to the start of word v (the blank before
    # it or its first token), so no extra blank frame is needed.
    jumps = [
        (src, dst, (v - u - 1) * skip_log_penalty)
        for u in range(-1, n_words)
        for v in range(u + 2, n_words)
        for src in exit_(u)
        for dst in entry(v)
        if src != dst and not (ext[src] == ext[dst] != blank_id)
    ]
    j_src = np.array([j[0] for j in jumps], dtype=int)
    j_dst = np.array([j[1] for j in jumps], dtype=int)
    j_pen = np.array([j[2] for j in jumps], dtype=float)

    neg_inf = -np.inf
    idx = np.arange(states)
    alpha = np.full(states, neg_inf)
    for v in range(n_words):
        for s in entry(v):
            alpha[s] = max(alpha[s], log_probs[0, ext[s]] + v * skip_log_penalty)
    back = np.zeros((frames, states), dtype=np.int32)
    for t in range(1, frames):
        step = np.concatenate(([neg_inf], alpha[:-1]))
        skip = np.where(
            can_skip,
            np.concatenate(([neg_inf, neg_inf], alpha[:-2])),
            neg_inf,
        )
        moves = np.stack([alpha, step, skip])
        choice = moves.argmax(axis=0)
        best = moves.max(axis=0)
        pred = idx - choice
        if jumps:
            cand = alpha[j_src] + j_pen
            order = np.argsort(cand, kind="stable")
            jump_best = np.full(states, neg_inf)
            jump_src = np.zeros(states, dtype=int)
            jump_best[j_dst[order]] = cand[order]
            jump_src[j_dst[order]] = j_src[order]
            better = jump_best > best
            best[better] = jump_best[better]
            pred[better] = jump_src[better]
        back[t] = pred
        alpha = best + log_probs[t, ext_arr]

    # Words left out at the end: stop at the end of word u.
    end_score, state = neg_inf, states - 1
    for u in range(-1, n_words):
        for s in exit_(u):
            score = alpha[s] + (n_words - 1 - u) * skip_log_penalty
            if score > end_score:
                end_score, state = score, s
    if not np.isfinite(end_score):
        return None
    path = np.empty(frames, dtype=int)
    for t in range(frames - 1, -1, -1):
        path[t] = state
        state = int(back[t, state]) if t else state
    out: list[list[tuple[int, int]] | None] = []
    for states_w in token_state:
        spans = []
        for st in states_w:
            hits = np.flatnonzero(path == st)
            if hits.size == 0:
                break
            spans.append((int(hits[0]), int(hits[-1])))
        out.append(spans if len(spans) == len(states_w) and spans else None)
    return out


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
