"""Japanese lesson text to mora units.

The reading comes from fugashi with unidic-lite (MeCab + UniDic), using
each morpheme's pronunciation field (katakana, long vowels written as
``ー``, the particle は read as ワ). The reading is split into morae and
each mora becomes one unit of the phoneme CTC model's tokens:

    緑茶 -> リョクチャ -> リョ [ɾ j o] | ク [k] | チャ [tʃ a]
    コーヒー -> コー [k o] | ヒー [x i]
    学校 -> ガッコー -> ガ [ɡ a] | ッコー [k o]

The symbols were picked by CTC likelihood of native-sounding (neural TTS)
Japanese on the model, not taken from espeak-ng's Japanese voice: plain
a/i/u/e/o, ɾ for the r row, ʃ/tʃ/dʒ, x for ヒ. A long vowel stays inside
its mora with the short vowel token, since native long vowels fit oː
far worse than o; vowel length is judged by the duration ratio. ッ is
folded into the next mora, ン takes the place of the next consonant (m
before p/b/m, ŋ before k/g, n otherwise), and devoiced i/u are dropped.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.g2p import LessonUnit

_ROWS = {
    "": "アイウエオ",
    "k": "カキクケコ",
    "ɡ": "ガギグゲゴ",
    "s": "サシスセソ",
    "z": "ザジズゼゾ",
    "t": "タチツテト",
    "d": "ダヂヅデド",
    "n": "ナニヌネノ",
    "h": "ハヒフヘホ",
    "b": "バビブベボ",
    "p": "パピプペポ",
    "m": "マミムメモ",
    "ɾ": "ラリルレロ",
}
_VOWELS = "aiueo"
_KANA: dict[str, tuple[str, str]] = {
    kana: (cons, vowel)
    for cons, row in _ROWS.items()
    for kana, vowel in zip(row, _VOWELS)
}
_KANA.update(
    {
        "シ": ("ʃ", "i"),
        "ジ": ("dʒ", "i"),
        "チ": ("tʃ", "i"),
        "ツ": ("ts", "u"),
        "ヂ": ("dʒ", "i"),
        "ヅ": ("z", "u"),
        "ヒ": ("x", "i"),
        "フ": ("ɸ", "u"),
        "ヤ": ("j", "a"),
        "ユ": ("j", "u"),
        "ヨ": ("j", "o"),
        "ワ": ("w", "a"),
        "ヲ": ("", "o"),
        "ヴ": ("v", "u"),
    }
)
# Small ャュョ after an i-column kana: palatalised onset + vowel.
_SMALL_Y = {"ャ": "a", "ュ": "u", "ョ": "o"}
_PALATAL = {
    "": ["j"],
    "k": ["kʲ"],
    "ɡ": ["ɡʲ"],
    "n": ["nʲ"],
    "x": ["x"],
    "b": ["bʲ"],
    "p": ["pʲ"],
    "m": ["mʲ"],
    "ɾ": ["ɾ", "j"],
}
# Small ァィゥェォ: the preceding kana's consonant with a new vowel.
_SMALL_V = {"ァ": "a", "ィ": "i", "ゥ": "u", "ェ": "e", "ォ": "o"}
_LABIAL = {"p", "b", "m", "pʲ", "bʲ", "mʲ"}
_VELAR = {"k", "ɡ", "kʲ", "ɡʲ"}
_VOICELESS = {"k", "kʲ", "s", "ʃ", "t", "tʃ", "ts", "h", "x", "ɸ", "p", "pʲ"}


def to_katakana(text: str) -> str:
    return "".join(
        chr(ord(c) + 0x60) if "\u3041" <= c <= "\u3096" else c for c in text
    )


def morae(kana: str) -> list[tuple[str, list[str]]]:
    """Split a kana reading into ``(mora, phones)``.

    Characters that are not kana are skipped. ッ is kept as a mora with
    no phones; ``devoice`` / ``lesson_units`` fold it into the next mora.
    """
    out: list[tuple[str, list[str]]] = []
    for c in to_katakana(kana):
        if c in _SMALL_Y and out and out[-1][1][-1:] == ["i"]:
            label, phones = out[-1]
            onset = _PALATAL.get(phones[0] if len(phones) == 2 else "", phones[:1])
            out[-1] = (label + c, [*onset, _SMALL_Y[c]])
        elif c in _SMALL_V and out and out[-1][1][-1:] and out[-1][1][-1] in _VOWELS:
            label, phones = out[-1]
            cons = phones[0] if len(phones) == 2 else "w" if label == "ウ" else ""
            out[-1] = (label + c, [cons, _SMALL_V[c]] if cons else [_SMALL_V[c]])
        elif c == "ー":
            # The model has long-vowel tokens but native long vowels fit the
            # short ones far better, so length is left to the duration ratio.
            if out and out[-1][1]:
                out[-1] = (out[-1][0] + c, out[-1][1])
        elif c in ("ッ", "ン"):
            out.append((c, []))
        elif c in _KANA:
            cons, vowel = _KANA[c]
            out.append((c, [cons, vowel] if cons else [vowel]))
    return assimilate_nasals(out)


def assimilate_nasals(units: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
    """Give each ン the place of the consonant after it."""
    out = list(units)
    for n, (label, _) in enumerate(out):
        if label == "ン":
            nxt = out[n + 1][1] if n + 1 < len(out) else []
            cons = nxt[0] if nxt and nxt[0] not in _VOWELS else ""
            out[n] = (label, ["m" if cons in _LABIAL else "ŋ" if cons in _VELAR else "n"])
    return out


def fold_sokuon(units: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
    """Attach each ッ to the mora after it (ガッコー -> ガ | ッコー).

    The model hears a geminate as one long closure, not two consonants,
    so ッ gets no tokens of its own; its length shows in the next mora's
    duration.
    """
    out: list[tuple[str, list[str]]] = []
    pending = ""
    for label, phones in units:
        if label == "ッ":
            pending += label
        elif phones:
            out.append((pending + label, phones))
            pending = ""
    return out


def devoice(units: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
    """Drop the i/u of a mora devoiced in standard Japanese.

    A short high vowel after a voiceless consonant is devoiced before
    another voiceless consonant and at the end of the utterance (す in
    です, く in 緑茶, し in ました); the model hears no vowel there. Long
    vowels (ヒー, スー) stay voiced.
    """
    out = list(units)
    for n, (label, phones) in enumerate(units):
        if (
            label.endswith("ー")
            or len(phones) < 2
            or phones[-1] not in ("i", "u")
            or phones[-2] not in _VOICELESS
        ):
            continue
        nxt = units[n + 1][1] if n + 1 < len(units) else None
        if nxt is None or nxt[0] in _VOICELESS:
            out[n] = (label, phones[:-1])
    return out


def fugashi_reading(text: str) -> list[tuple[str, str | None]]:
    """``(surface, katakana pronunciation)`` per morpheme, from fugashi + unidic-lite."""
    import fugashi  # noqa: PLC0415

    tagger = fugashi.Tagger()
    return [(w.surface, getattr(w.feature, "pron", None)) for w in tagger(text)]


def lesson_units(
    text: str,
    vocab: dict[str, int],
    reading: Callable[[str], Sequence[tuple[str, str | None]]] | None = None,
) -> list[LessonUnit]:
    """One unit per mora of the lesson text, tagged with its word.

    Punctuation is dropped. Returns [] when fugashi is not installed or a
    word has no kana reading (a reading of part of the text would not
    fit the recording).
    """
    from app.services.g2p import LessonUnit  # noqa: PLC0415

    try:
        words = (reading or fugashi_reading)(text)
    except ImportError:
        return []
    flat: list[tuple[str, int, str, list[str]]] = []
    index = 0
    for surface, pron in words:
        if not any(c.isalnum() for c in surface):
            continue
        units = fold_sokuon(morae(pron or ""))
        if not units:
            return []
        flat += [(surface, index, label, phones) for label, phones in units]
        index += 1
    # Assimilation and devoicing look at the next mora, which may be in
    # the next word (本が, 学生です).
    phones = devoice(assimilate_nasals([(label, p) for _, _, label, p in flat]))
    out = [
        LessonUnit(word, index, label, ids)
        for (word, index, _, _), (label, p) in zip(flat, phones)
        if (ids := [vocab[t] for t in p if t in vocab])
    ]
    return out
