# ShadowTalk — Week 1 Commit Roadmap

One sentence, end to end: record → align → score, with sub-300 ms
post-utterance feedback and no audio-thread crashes. At least one
commit lands every day; each day below is the commit (or commit set)
that proves it.

## Day 1 — Foundation (done)
- Initial commit: repo scaffold, architecture README
- `backend/`: FastAPI app with `GET /health` and `WS /ws/align`
- `ios/`: SwiftUI recorder — AVAudioSession `.playAndRecord`,
  AVAudioEngine tap, 16 kHz mono PCM, 20 ms frames via `onFrame`

## Day 2 — Audio pipeline end to end (done, 2026-11-20)
- iOS recorder streams frames over WebSocket to the backend session
  (`ios/ShadowTalk/AlignSocket.swift`, wired in `RecorderViewModel`)
- Silero VAD endpointing on the server, with an energy-VAD fallback
  when torch/Silero is unavailable (`backend/app/services/vad.py`);
  utterance saved per session as WAV under `backend/data/sessions/`
- Commit: `feat(audio): stream recorder frames, VAD endpointing`
- Verified: 8 tests pass (`backend/tests/test_vad.py`), including a
  WebSocket end-to-end test; `/health` returns ok on uvicorn.
  Note: Swift was not compiled here (no Xcode on Linux) — build in
  Xcode before running on device. Scoring itself is still the Day 1
  stub (empty alignments); real WhisperX scoring is Day 3–5.

## Day 3 — Alignment for one sentence (wiring done, 2026-11-20)
- `WhisperXAligner` implemented with lazy imports: faster-whisper ASR
  (word timestamps) + `whisperx.load_align_model` / `whisperx.align`
  using the language-specific XLS-R head; raises
  `AlignmentUnavailableError` when the stack is not installed.
- Wired into `WS /ws/align`: `_score_session` uses `main.ALIGNER`
  when set (or `SHADOWTALK_ALIGNER=whisperx` via `default_aligner`);
  the score payload now carries `alignment` (`ok` / `unavailable` /
  `error`), `alignment_backend` and `transcript`, and returns real
  per-unit GOP scores when an aligner is present. With no aligner
  (local dev) it says `unavailable` explicitly instead of returning
  empty units as if alignment had run.
- Fixed: `speaking_rate_normalise` no longer scales learner units by
  a bogus ratio when the reference alignment is still empty (Day 4).
- Commit: `feat(align): whisperx pipeline, phoneme timestamps endpoint`
- Verified: 14 tests pass (8 Day 2 + 6 new in `tests/test_alignment.py`),
  including a fake-aligner WS end-to-end test (Japanese mora units,
  low-confidence unit gated, never scored as an error).
- **Not yet done — do not call Day 3 finished:** WhisperX itself has
  NOT been run here (no torch / faster-whisper / whisperx installed,
  no GPU, no real spoken sentence to transcribe). WhisperX returns
  word-level timestamps; per-phoneme / per-mora timestamps need the
  wav2vec2 phoneme CTC head, which moves to Day 4. The remaining Day 3
  step is: run `WhisperXAligner` on a GPU worker (or `SHADOWTALK_ALIGNER`
  =whisperx) against one real recorded sentence and confirm the word
  timestamps land in the score payload.

## Day 4 — Relative scoring vs reference (done, 2026-11-20)
- `ReferenceStore` (`backend/app/services/reference.py`): a reference
  recording is aligned once through the same aligner as the learner
  and cached by `reference_id`; later attempts reuse it (proven by a
  counting-aligner test: 1 align call for 2 lookups). Reference WAVs
  load from `backend/data/references/<id>.wav` (16 kHz mono PCM16,
  validated, never committed) or are registered in memory.
- Learner scored relative to the reference (`score_utterance`):
  speaking-rate normalisation first, then per-unit `duration_ratio`
  (learner / reference duration, 1.0 = same rhythm) and a DTW duration
  distance over the unit-duration sequences — rhythm is reported
  separately from pronunciation, so a slow-but-correct unit is not
  misread as mispronounced.
- Confidence gating now cancels model bias both ways: a unit is
  flagged (gop None, never scored as an error) when its own CTC
  confidence is below 0.5 *or* when the paired reference unit is
  below 0.5 — a boundary the model cannot place on the reference
  cannot be blamed on the learner.
- Wired into `WS /ws/align`: score payload now carries
  `reference_alignment` (`ok` / `missing` / `unavailable` / `error`),
  `duration_dtw_distance`, and per-unit `duration_ratio`; with no
  reference it says `missing` and scores the learner alone, as in
  Day 3.
- Commit: `feat(score): relative alignment scoring with confidence gating`
- Verified: 24 tests pass (14 Day 2–3 + 10 new in
  `tests/test_relative_scoring.py`), including a WS end-to-end test
  with a cached reference (DTW 0.0, duration ratios 1.0) and one with
  a missing reference.
- **Not yet done — do not call Day 4 finished:** no real reference
  recording has been aligned here (no WhisperX stack / GPU, no real
  spoken sentence), and units are still word-level from WhisperX —
  per-phoneme / per-mora splitting needs the wav2vec2 phoneme CTC
  head. GOP is still the CTC confidence placeholder, and pitch DTW
  is Day 5.

## Day 5 — GOP + pitch
- Goodness-of-pronunciation score per phoneme from CTC posteriors
- torchcrepe F0 extraction; DTW pitch contour vs reference,
  normalised per speaker
- Score payload v1: `{phoneme, gop, pitch_delta, flagged}`
- Commit: `feat(score): GOP and pitch contour scoring`

## Day 6 — Score screen on iOS
- Waveform + per-phoneme pass/fail rendering from the score payload
- Pitch contour overlay against the reference
- One-tap retry loop back into the recorder
- Commit: `feat(ios): score screen with phoneme and pitch rendering`

## Day 7 — Hardening + demo
- Post-utterance latency measured and logged (target: < 300 ms)
- Audio-session interruption / route-change crash tests
- Recorded end-to-end demo; TestFlight build
- Commit: `chore(release): week-1 demo, latency report, testflight`

## Language handling (from day 3 onward)
Language ID routes first, then a language-specific XLS-R alignment
head: English in IPA, Japanese aligned by mora (mora-timed language —
phoneme boundaries are the wrong unit), Mandarin by initial/final
plus tone contour.
