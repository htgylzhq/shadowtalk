"""Japanese lesson text to mora units.

The reading comes from fugashi with unidic-lite (MeCab + UniDic), using
each morpheme's pronunciation field (katakana, long vowels written as
``ー``, the particle は read as ワ). The reading is split into morae and
each mora becomes one unit of the phoneme CTC model's tokens:

    緑茶 -> リョクチャ -> リョ [rʲ o] | ク [k u] | チャ [tɕ a]
    コーヒー -> コー [k oː] | ヒー [ç iː]
    学校 -> ガッコー -> ガ [ɡ a] | ッ [k] | コー [k oː]

The symbols were picked by CTC likelihood on the model, not taken from
espeak-ng's Japanese voice: plain a/i/u/e/o with ``ː`` long vowels and
ɾ for the r row fit the model better than espeak's ä/o̞/ɯᵝ, doubled
vowels and ɽ. A long vowel stays inside its mora (コー is one unit)
because the model has a single token for it. ッ is a mora of its own
holding the next mora's consonant, and ン takes the place of the next
consonant (m before p/b/m, ŋ before k/g, n otherwise).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

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
        "シ": ("ɕ", "i"),
        "ジ": ("dʑ", "i"),
        "チ": ("tɕ", "i"),
        "ツ": ("ts", "u"),
        "ヂ": ("dʑ", "i"),
        "ヅ": ("z", "u"),
        "ヒ": ("ç", "i"),
        "フ": ("ɸ", "u"),
        "ヤ": ("j", "a"),
        "ユ": ("j", "u"),
        "ヨ": ("j", "o"),
        "ワ": ("w", "a"),
        "ヲ": ("", "o"),
        "ヴ": ("v", "u"),
    }
)
# Small ャュョ after an i-column kana: palatalised consonant + vowel.
_SMALL_Y = {"ャ": "a", "ュ": "u", "ョ": "o"}
_PALATAL = {
    "k": "kʲ",
    "ɡ": "ɡʲ",
    "n": "nʲ",
    "h": "ç",
    "b": "bʲ",
    "p": "pʲ",
    "m": "mʲ",
    "ɾ": "rʲ",
}
# Small ァィゥェォ: the preceding kana's consonant with a new vowel.
_SMALL_V = {"ァ": "a", "ィ": "i", "ゥ": "u", "ェ": "e", "ォ": "o"}
_LONG = {"a": "aː", "i": "iː", "u": "uː", "e": "eː", "o": "oː"}
_LABIAL = {"p", "b", "m", "pʲ", "bʲ", "mʲ"}
_VELAR = {"k", "ɡ", "kʲ", "ɡʲ"}


def to_katakana(text: str) -> str:
    return "".join(
        chr(ord(c) + 0x60) if "\u3041" <= c <= "\u3096" else c for c in text
    )


def morae(kana: str) -> list[tuple[str, list[str]]]:
    """Split a kana reading into ``(mora, phones)``.

    Characters that are not kana are skipped.
    """
    out: list[tuple[str, list[str]]] = []
    for c in to_katakana(kana):
        if c in _SMALL_Y and out and out[-1][1][-1:] == ["i"]:
            label, phones = out[-1]
            cons = phones[0] if len(phones) == 2 else ""
            out[-1] = (label + c, [_PALATAL.get(cons, cons) or "j", _SMALL_Y[c]])
        elif c in _SMALL_V and out and out[-1][1][-1:] and out[-1][1][-1] in _VOWELS:
            label, phones = out[-1]
            cons = phones[0] if len(phones) == 2 else "w" if label == "ウ" else ""
            out[-1] = (label + c, [cons, _SMALL_V[c]] if cons else [_SMALL_V[c]])
        elif c == "ー":
            if out and out[-1][1] and out[-1][1][-1] in _LONG:
                label, phones = out[-1]
                out[-1] = (label + c, phones[:-1] + [_LONG[phones[-1]]])
        elif c in ("ッ", "ン"):
            out.append((c, []))
        elif c in _KANA:
            cons, vowel = _KANA[c]
            out.append((c, [cons, vowel] if cons else [vowel]))
    # ッ and ン depend on the consonant that follows them.
    for n, (label, phones) in enumerate(out):
        if label not in ("ッ", "ン"):
            continue
        nxt = out[n + 1][1] if n + 1 < len(out) else []
        cons = nxt[0] if len(nxt) == 2 else ""
        if label == "ッ":
            out[n] = (label, [cons] if cons else [])
        else:
            out[n] = (label, ["m" if cons in _LABIAL else "ŋ" if cons in _VELAR else "n"])
    return [(label, phones) for label, phones in out if phones]


def fugashi_reading(text: str) -> list[tuple[str, str | None]]:
    """``(surface, katakana pronunciation)`` per morpheme, from fugashi + unidic-lite."""
    import fugashi  # noqa: PLC0415

    tagger = fugashi.Tagger()
    return [(w.surface, getattr(w.feature, "pron", None)) for w in tagger(text)]


def japanese_units(
    text: str,
    vocab: dict[str, int],
    reading: Callable[[str], Sequence[tuple[str, str | None]]] | None = None,
) -> list[tuple[str, list[tuple[str, list[int]]]]]:
    """``(word, [(mora, token_ids), ...])`` per lesson word.

    Punctuation is dropped. Returns [] when fugashi is not installed or a
    word has no kana reading (a reading of part of the text would not
    fit the recording).
    """
    try:
        words = (reading or fugashi_reading)(text)
    except ImportError:
        return []
    out = []
    for surface, pron in words:
        if not any(c.isalnum() for c in surface):
            continue
        units = [
            (label, [vocab[p] for p in phones if p in vocab])
            for label, phones in morae(pron or "")
        ]
        units = [(label, ids) for label, ids in units if ids]
        if not units:
            return []
        out.append((surface, units))
    return out
