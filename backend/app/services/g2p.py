"""Lesson text to reference phoneme targets.

espeak-ng (GPL-3) runs as a separate ``espeak-ng`` process and is never
imported or linked. Its IPA is split into the phoneme CTC model's own
tokens (the model was fine-tuned on espeak phonemes), so each lesson
word becomes a list of vocabulary ids.
"""

from __future__ import annotations

import shutil
import subprocess
import unicodedata
from collections.abc import Callable

from app.services.alignment import Language

ESPEAK_VOICES = {
    Language.ENGLISH: "en-us",
    Language.JAPANESE: "ja",
    Language.MANDARIN: "cmn",
}
_DROP = {"ˈ", "ˌ", " ", "\u200d"}  # stress marks and joiners carry no phoneme


def espeak_available() -> bool:
    return shutil.which("espeak-ng") is not None


def espeak_ipa(word: str, language: Language) -> str:
    """IPA for one word, from the espeak-ng CLI."""
    out = subprocess.run(
        ["espeak-ng", "-q", "--ipa", "-v", ESPEAK_VOICES[language], word],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return out.stdout.strip()


def lesson_words(text: str) -> list[str]:
    """Split lesson text into words, dropping surrounding punctuation."""
    words = []
    text = text.replace("\u2019", "'").replace("\u02bc", "'")  # curly apostrophes
    for raw in text.split():
        word = "".join(
            c if unicodedata.category(c)[0] in "LMN" or c == "'" else " "
            for c in raw
        ).strip(" '")
        words += word.split()
    return words


def tokenize_ipa(ipa: str, vocab: dict[str, int]) -> list[int]:
    """Greedy longest match of IPA against the CTC vocabulary.

    Symbols the vocabulary lacks are dropped rather than guessed.
    """
    ipa = unicodedata.normalize("NFC", "".join(c for c in ipa if c not in _DROP))
    longest = max((len(t) for t in vocab), default=1)
    ids: list[int] = []
    i = 0
    while i < len(ipa):
        for size in range(min(longest, len(ipa) - i), 0, -1):
            token = vocab.get(ipa[i : i + size])
            if token is not None:
                ids.append(token)
                i += size
                break
        else:
            i += 1
    return ids


def lesson_targets(
    text: str,
    language: Language,
    vocab: dict[str, int],
    ipa: Callable[[str, Language], str] | None = None,
) -> list[tuple[str, list[int]]]:
    """``(word, token_ids)`` for each lesson word that yields phonemes."""
    ipa = ipa or espeak_ipa
    targets = []
    for word in lesson_words(text):
        ids = tokenize_ipa(ipa(word, language), vocab)
        if ids:
            targets.append((word, ids))
    return targets
