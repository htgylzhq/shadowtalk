# ShadowTalk

A language-learning app built entirely around **shadowing**: hear a native
sentence, record yourself speaking it in sync, and get instant, phoneme-level
feedback on pronunciation, rhythm, and intonation.

Most language apps teach vocabulary. ShadowTalk trains your mouth. Shadowing
is the technique interpreters train with, and it is the standard method in
Japanese language schools — yet learners still practise with raw audio files
and zero feedback. ShadowTalk closes that loop: bring any sentence, shadow it,
and be scored on your own voice against the canonical reference recording.

## How it works

1. Tap a sentence and hear the native (canonical reference) audio.
2. Record yourself shadowing it. Audio streams to the backend *while you
   speak*, in 20 ms frames.
3. The backend aligns your recording against the reference, scores each
   phoneme (or mora, for Japanese), compares your pitch contour, and returns
   a score — targeted at **< 300 ms after you finish speaking**, because
   alignment runs incrementally during the utterance, not after it.

## Architecture

### iOS client (`ios/`)

- Native **Swift / SwiftUI** — no cross-platform audio layer; that is where
  latency and audio-session crashes come from.
- `AVAudioSession` in `.playAndRecord` mode, `AVAudioEngine` input tap,
  capture converted to **16 kHz mono PCM**, chunked into 20 ms frames and
  streamed over a WebSocket.

### Backend (`backend/`)

- **FastAPI + Postgres + Redis**, with GPU workers for the alignment
  pipeline.
- WebSocket ingestion of 20 ms Opus frames; **Silero VAD** for endpointing
  (deciding when an utterance has ended).
- Alignment pipeline (**WhisperX**):
  - `faster-whisper` for ASR,
  - **wav2vec2 forced alignment** for phoneme boundaries,
  - **GOP (goodness of pronunciation)** scoring from CTC posteriors,
  - **torchcrepe** F0 pitch extraction, DTW-warped against the reference
    pitch contour and normalised per speaker.

### Cross-language phoneme-boundary drift

One universal alignment model drifts across languages, so ShadowTalk does
not use one:

- **Language ID routes first**, then a language-specific **wav2vec2 XLS-R**
  alignment head and unit set:
  - English — phonemes (IPA)
  - Japanese — **mora**, not phonemes (Japanese is mora-timed; phoneme
    boundaries are the wrong unit)
  - Mandarin — initial / final units plus tone contour
- Scoring is **relative, never absolute**: the learner's alignment is
  compared against a forced alignment of the canonical reference recording
  run through the *same* model, so model bias cancels out on both sides.
- **Speaking-rate normalisation + DTW** before duration comparison.
- **Per-phoneme CTC confidence gating**: low-confidence boundaries are
  interpolated and flagged — never scored as learner errors.

## Repo layout

```
shadowtalk/
├── README.md
├── backend/
│   ├── requirements.txt
│   └── app/
│       ├── main.py            # FastAPI app: /health + /ws/align WebSocket
│       └── services/
│           └── alignment.py   # Language routing, WhisperX alignment, GOP
└── ios/
    └── ShadowTalk/
        ├── ShadowTalkApp.swift
        ├── ContentView.swift
        └── RecorderViewModel.swift   # AVAudioEngine capture, 16 kHz mono
```

## Week-1 milestone

One sentence, end to end:

- [ ] iOS: record one sentence (16 kHz mono, 20 ms frames, streamed live)
- [ ] Backend: receive stream, VAD endpointing, forced alignment completes
      with **sub-300 ms post-utterance feedback**
- [ ] Scoring: per-unit GOP score + pitch-contour comparison vs. reference
- [ ] No audio-thread crashes across iOS audio-session interruptions
      (calls, route changes, headphone unplug)

After week 1: lessons, spaced-repetition review, and the content pipeline
that turns arbitrary text/audio into graded shadowing lessons.

## Running the backend locally

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
# GET  http://localhost:8000/health
# WS   ws://localhost:8000/ws/align
```

The heavy ML dependencies (WhisperX, faster-whisper, torchcrepe) run on GPU
workers in production; the API process imports them lazily, so the app
starts without them for local development of the streaming layer.
