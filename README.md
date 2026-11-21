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
  streamed over a WebSocket (`AlignSocket.swift` → `/ws/align`).

### Backend (`backend/`)

- **FastAPI + Postgres + Redis**, with GPU workers for the alignment
  pipeline.
- WebSocket ingestion of 20 ms Opus frames; **Silero VAD** for endpointing
  (deciding when an utterance has ended), with an energy-VAD fallback
  when Silero/torch is unavailable. Each finished utterance is saved
  as a WAV under `backend/data/sessions/`.
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

- [x] iOS: record one sentence (16 kHz mono, 20 ms frames, streamed live)
      — Day 2, 2026-11-20; VAD endpointing + utterance WAV saving included
- [~] Backend: receive stream, VAD endpointing, forced alignment completes
      with **sub-300 ms post-utterance feedback** — Day 3 wiring done
      2026-11-20 (`WhisperXAligner` + score payload returns real units
      when an aligner is configured); the model itself has not run yet
      (no GPU / WhisperX stack on this machine, no real spoken sentence),
      and per-phoneme timestamps still need the wav2vec2 phoneme head
      (Day 4)
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

## Run the iOS recorder

1. Open `ios/ShadowTalk.xcodeproj` in Xcode and run the **ShadowTalk** scheme
   on an iOS 17 or later simulator or device.
2. Choose **View sample score** to open the score screen without the microphone
   or backend. To score a recording, tap **Record**, speak, and tap **Stop**
   if the backend hasn't detected the end of speech. If the app can't reach the
   server, it shows a labeled sample instead.

The app uses `http://localhost:8000` by default. To use another server, set
`SHADOWTALK_BACKEND_URL` to its HTTP or HTTPS base URL in the **ShadowTalk**
scheme's Run environment or the app target's build settings. Plain HTTP is
allowed only on loopback. For a physical device, use a reachable HTTPS URL
rather than `localhost`.

To compare against a reference, set `SHADOWTALK_REFERENCE_ID` to the basename
of a WAV in `backend/data/references/` and `SHADOWTALK_LANGUAGE` to its
language code (default `ja`). Run the backend with a configured aligner and
`SHADOWTALK_PITCH=torchcrepe`. The chart uses the optional `pitch_contour`
object in the v1 score response. Its `learner` and `reference` arrays are
semitones relative to each speaker's median, with `null` for unvoiced frames
and `frame_s` seconds between samples. The chart shows elapsed time for each
recording, not the dynamic time warping path.

Run the model and recorder flow tests with **Product > Test** on an iPhone
simulator. The offline recording test needs simulator microphone access.

For the opt-in live WebSocket and recorder tests, start the fixture backend
from `backend/` with `PYTHONPATH=. .venv/bin/python tests/score_screen_fixture.py`.
Run `xcodebuild test` with `TEST_RUNNER_SHADOWTALK_FIXTURE_URL=http://127.0.0.1:8766`
in its environment. The fixture exercises the real API with synthetic alignment
and pitch data. The recorder test also needs a working simulator audio input.
