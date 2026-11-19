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

## Day 2 — Audio pipeline end to end
- iOS recorder streams frames over WebSocket to the backend session
- Silero VAD endpointing on the server; utterance saved per session
- Commit: `feat(audio): stream recorder frames, VAD endpointing`

## Day 3 — Alignment for one sentence
- faster-whisper transcription of a single recorded sentence
- wav2vec2 forced alignment → per-phoneme timestamps as JSON
- Commit: `feat(align): whisperx pipeline, phoneme timestamps endpoint`

## Day 4 — Relative scoring vs reference
- Forced alignment of the canonical reference recording
- Learner scored relative to reference (model bias cancels)
- Speaking-rate normalisation + DTW duration comparison
- Per-phoneme CTC confidence gating: low-confidence units flagged,
  never scored as errors
- Commit: `feat(score): relative alignment scoring with confidence gating`

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
