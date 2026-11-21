"""Mandarin lesson text to initial/final units with tone.

espeak-ng's ``cmn`` voice gives one space-separated IPA syllable per
character, written the way the phoneme CTC head was trained (its
vocabulary holds whole toned finals such as ``onɡ5`` or ``iou2``). The
IPA tone marks are lossy: tone 1 and tone 4 are both written ``5``. The
lexical tone (after espeak's tone sandhi, e.g. 你好 -> ní hǎo) is read
from the ``-x`` phoneme mnemonics instead, which keep the Chao contour
digits (55, 35, 21/214, 51; anything else is the neutral tone).

Each syllable becomes an *initial* unit (the onset consonant, or glide
``j`` / ``w``, when there is one) and a *final* unit (the toned vowel
token plus any coda), so GOP is scored per initial/final and tone is
scored on the final from the pitch contour.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from dataclasses import replace

from app.services.alignment import AlignedUnit, Language
from app.services.g2p import ESPEAK_VOICES, LessonUnit, lesson_words, tokenize_ipa

# Characters espeak's cmn IPA appends to a final for its tone.
TONE_MARKS = frozenset("12345ɜ")
CODAS = frozenset({"n", "ŋ", "ɻ"})
_CHAO_TONES = {"55": 1, "35": 2, "21": 3, "214": 3, "51": 4}

PhonemeFn = Callable[[str], tuple[str, str]]


def espeak_cmn(text: str) -> tuple[str, str]:
    """``(ipa, mnemonics)`` for a chunk of Chinese text, from the espeak-ng CLI."""
    voice = ESPEAK_VOICES[Language.MANDARIN]

    def run(flag: str) -> str:
        out = subprocess.run(
            ["espeak-ng", "-q", flag, "-v", voice, text],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()

    return run("--ipa"), run("-x")


def is_final(label: str) -> bool:
    """True for a final unit label (it carries an espeak tone mark)."""
    return any(c in TONE_MARKS for c in label)


def tone_from_mnemonic(syllable: str) -> int:
    """Lexical tone 1-4, or 5 for neutral, from an espeak ``-x`` syllable."""
    digits = "".join(re.findall(r"\d", syllable))
    return _CHAO_TONES.get(digits, 5)


def split_initial_final(tokens: list[str]) -> tuple[int, int] | None:
    """``(initial_len, final_len)`` for one syllable's tokens, or None.

    The initial is every token before the first toned token; the final
    is the toned token and everything after it. None when the syllable
    has no toned token (nothing to anchor a final on).
    """
    for k, token in enumerate(tokens):
        if is_final(token):
            return k, len(tokens) - k
    return None


def lesson_units(
    text: str, vocab: dict[str, int], phonemes: PhonemeFn | None = None
) -> list[LessonUnit]:
    """Initial/final units for each syllable of the lesson text.

    Each Chinese character is its own lesson word when espeak gives one
    syllable per character; otherwise (digits, Latin letters) the whole
    chunk is the word for all its syllables.
    """
    phonemes = phonemes or espeak_cmn
    id_to_token = {i: t for t, i in vocab.items()}
    units: list[LessonUnit] = []
    word_index = 0
    for chunk in lesson_words(text):
        ipa, mnemonics = phonemes(chunk)
        syllables = ipa.split()
        tones = [tone_from_mnemonic(s) for s in mnemonics.replace("_|", " ").split()]
        if len(tones) != len(syllables):
            tones = [None] * len(syllables)
        # One syllable per character (Han characters and single digits).
        per_char = len(chunk) == len(syllables)
        for n, (syllable, tone) in enumerate(zip(syllables, tones)):
            ids = tokenize_ipa(syllable.replace("-", ""), vocab)
            split = split_initial_final([id_to_token[i] for i in ids])
            if split is None:
                continue
            k, _ = split
            word = chunk[n] if per_char else chunk
            index = word_index + n if per_char else word_index
            if k:
                units.append(
                    LessonUnit(word, index, "".join(id_to_token[i] for i in ids[:k]), ids[:k])
                )
            units.append(
                LessonUnit(word, index, "".join(id_to_token[i] for i in ids[k:]), ids[k:], tone)
            )
        word_index += len(syllables) if per_char else 1
    return units


def group_initial_final(units: list[AlignedUnit]) -> list[AlignedUnit]:
    """Attach decoded codas to their finals.

    The CTC head already emits a Mandarin initial or a toned final as one
    token; only a coda (n, ŋ, ɻ) comes separately. It joins the final
    before it unless a toned token follows directly, in which case it is
    the next syllable's initial. Any other token stays its own unit.
    Decoded finals carry no lexical tone: the IPA mark cannot tell tone
    1 from tone 4.
    """
    groups: list[list[AlignedUnit]] = []
    for n, unit in enumerate(units):
        nxt = units[n + 1].label if n + 1 < len(units) else ""
        after_final = bool(groups) and is_final(groups[-1][0].label)
        if after_final and unit.label in CODAS and not is_final(nxt):
            groups[-1].append(unit)
        else:
            groups.append([unit])
    return [_merge(g) for g in groups]


def _merge(group: list[AlignedUnit]) -> AlignedUnit:
    if len(group) == 1:
        return group[0]
    return replace(
        group[0],
        label="".join(u.label for u in group),
        end_s=group[-1].end_s,
        ctc_confidence=sum(u.ctc_confidence for u in group) / len(group),
        token_ids=[t for u in group for t in u.token_ids or []],
    )
