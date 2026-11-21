"""Local FastAPI fixture for the opt-in iOS recording tests."""

import uvicorn

import app.main as main
from test_gop_pitch import (
    AlignedUnit,
    AlignmentResult,
    Language,
    PosteriorFakeAligner,
    contour,
    pcm_frame,
    rising,
)


class FixtureAligner(PosteriorFakeAligner):
    def align(self, pcm16k, language):
        if len(pcm16k) < 2 * main.PCM_FRAME_BYTES or len(pcm16k) % main.PCM_FRAME_BYTES:
            return AlignmentResult(language, "", [])
        return super().align(pcm16k, language)


class FixturePitchExtractor:
    name = "fixture-pitch"

    def extract(self, pcm16k):
        if len(pcm16k) > main.PCM_FRAME_BYTES:
            return contour([120.0] * 60)
        return contour(rising(180.0, n=60))


if __name__ == "__main__":
    main.ALIGNER = FixtureAligner()
    main.PITCH_EXTRACTOR = FixturePitchExtractor()
    main.REFERENCE_STORE.register(
        "score-fixture",
        AlignmentResult(
            Language.JAPANESE, "こんにちは",
            [AlignedUnit("こん", 0, 0.30, 0.9), AlignedUnit("にち", 0.30, 0.60, 0.9)],
        ),
        pcm16k=pcm_frame(),
    )
    uvicorn.run(main.app, host="127.0.0.1", port=8766)
