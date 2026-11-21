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


class FixturePitchExtractor:
    name = "fixture-pitch"

    def extract(self, pcm16k):
        if len(pcm16k) > len(pcm_frame()):
            return contour([120.0] * 60)
        return contour(rising(180.0, n=60))


if __name__ == "__main__":
    main.ALIGNER = PosteriorFakeAligner()
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
