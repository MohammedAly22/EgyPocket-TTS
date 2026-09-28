"""Deterministic Egyptian-Arabic text frontend.

The same `normalize_text` runs on the training transcripts and on user text at
inference, so the model always receives one canonical representation:

* Unicode NFKC, invisible characters and tatweel removed, Arabic-Indic digits
  mapped to ASCII and then verbalized the Egyptian way.
* Hamza-on-alef forms unified to ا (the corpus is written that way).
* Diacritics are KEPT. They are pronunciation control: "مَدِينَة" and
  "مَدِينَةْ" must stay different. Each letter carries at most
  [~][shadda][one vowel/tanween/sukun], always in that order, so equivalent
  inputs tokenize identically. Marks that are not attached to a letter are
  dropped.
* ة keeps whatever mark follows it (ة / ةْ / ةَ / ةِ / ةُ / ةً / ةٌ / ةٍ are all
  different model inputs; the tokenizer even gives each pair its own token).
* Punctuation is reduced to "," (pause) and "." (sentence end), the two
  marks the training data carries.
"""

import random
import re
import unicodedata
from collections.abc import Iterable

from egypocket.text.chars import (
    ALEF,
    ARABIC_INDIC_DIGITS,
    COMMA,
    DIACRITICS_SET,
    EGY_MARKER,
    CATT_MOTHALATHA,
    HA,
    INVISIBLE,
    LETTER_MAP,
    LETTERS_SET,
    PERIOD,
    PERSIAN_DIGITS,
    SHADDA,
    TA_MARBUTA,
    TATWEEL,
)
from egypocket.text.numbers import expand_numbers

_SENTENCE_END = set(".!?\u061f\u2026\u06d4")  # . ! ? ؟ … ۔
_CLAUSE = set(",\u060c;\u061b:\u2013\u2014")  # , ، ; ؛ : – —
_NO_SHADDA = {ALEF, "\u0649", TA_MARBUTA, "\u0621"}  # ا ى ة ء
_MARKS = DIACRITICS_SET | {EGY_MARKER}

_TRANSLATE = {**ARABIC_INDIC_DIGITS, **PERSIAN_DIGITS}
_TRANSLATE.update({ord(k): v for k, v in LETTER_MAP.items()})
_TRANSLATE.update({ord(c): None for c in INVISIBLE + TATWEEL})
_TRANSLATE[ord(CATT_MOTHALATHA)] = EGY_MARKER


def normalize_unicode(text: str) -> str:
    """NFKC + letter/digit rewrites + removal of invisible characters and tatweel."""
    text = unicodedata.normalize("NFKC", text)
    return text.translate(_TRANSLATE)


def iter_clusters(text: str) -> Iterable[tuple[str, str]]:
    """Yield (base character, following marks) pairs; marks are raw, not canonical."""
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        j = i + 1
        while j < n and text[j] in _MARKS:
            j += 1
        yield ch, text[i + 1 : j]
        i = j


def canonical_marks(letter: str, marks: str) -> str:
    """[~][shadda][last vowel-like mark], dropping what cannot sit on `letter`."""
    marker = EGY_MARKER in marks and letter != TA_MARBUTA
    shadda = SHADDA in marks and letter not in _NO_SHADDA
    vowel = ""
    for m in marks:
        if m in DIACRITICS_SET and m != SHADDA:
            vowel = m  # the last vowel-like mark wins
    return (EGY_MARKER if marker else "") + (SHADDA if shadda else "") + vowel


def canonicalize_diacritics(text: str) -> str:
    """Put every letter's marks in canonical order; drop marks without a letter."""
    out = []
    for base, marks in iter_clusters(text):
        if base in _MARKS:
            continue  # a mark at the start of the text: no letter to carry it
        if base in LETTERS_SET:
            out.append(base + canonical_marks(base, marks))
        else:
            out.append(base)  # marks after spaces/punctuation are orphans
    return "".join(out)


def strip_diacritics(text: str) -> str:
    """Remove the diacritics and the "~" marker, keeping letters and punctuation."""
    return "".join(c for c in text if c not in _MARKS)


def _clean_characters(text: str) -> tuple[str, list[str]]:
    out: list[str] = []
    dropped: list[str] = []
    for ch in text:
        if ch in LETTERS_SET or ch in _MARKS:
            out.append(ch)
        elif ch.isspace():
            out.append(" ")
        elif ch in _SENTENCE_END:
            out.append(f" {PERIOD} ")
        elif ch in _CLAUSE:
            out.append(f" {COMMA} ")
        elif unicodedata.category(ch) == "Mn":
            # Other combining marks (hamza above/below, maddah, small high
            # letters, ...) sit inside a word: drop them without splitting it.
            continue
        else:
            # Quotes, brackets, hyphens, Latin letters, emoji ... are not part
            # of what the model was trained to read.
            if not (ch in "\"'()[]{}<>«»“”‘’-_/\\|*#" or unicodedata.category(ch).startswith("P")):
                dropped.append(ch)
            out.append(" ")
    return "".join(out), dropped


def _tidy_punctuation(text: str) -> str:
    tokens = text.split()
    merged: list[str] = []
    for tok in tokens:
        if tok in (COMMA, PERIOD):
            if not merged:
                continue  # no leading punctuation
            if merged[-1] in (COMMA, PERIOD):
                if tok == PERIOD:
                    merged[-1] = PERIOD  # ", ." -> "."
                continue
            merged.append(tok)
        else:
            merged.append(tok)
    text = " ".join(merged)
    return re.sub(r" ([,.])", r"\1", text)


def apply_ta_marbuta_lexicon(text: str, lexicon: set[str] | frozenset[str]) -> str:
    """Rewrite a final ه to ة for words the training data restored to ة.

    The corpus writes ة as ه; the data pipeline restores ة on the word types
    where CATT's vowelization shows a taa marbuta, and saves those types as a
    lexicon. Applying it at inference makes "المدينه" and "المدينة" read the
    same way. Marks written on the final letter are kept.
    """
    if not lexicon:
        return text

    def fix(match: re.Match[str]) -> str:
        word = match.group(0)
        if strip_diacritics(word) not in lexicon:
            return word
        # Replace the last ه (the word's final letter) and keep its marks.
        idx = word.rfind(HA)
        return word[:idx] + TA_MARBUTA + word[idx + 1 :]

    return _WORD_RE.sub(fix, text)


_WORD_RE = re.compile(
    "[" + re.escape("".join(sorted(LETTERS_SET))) + re.escape("".join(sorted(_MARKS))) + "]+"
)


def normalize_text(
    text: str, ta_marbuta_lexicon: set[str] | frozenset[str] | None = None
) -> str:
    """The canonical model text for `text` (see module docstring)."""
    return normalize_text_verbose(text, ta_marbuta_lexicon)[0]


def normalize_text_verbose(
    text: str, ta_marbuta_lexicon: set[str] | frozenset[str] | None = None
) -> tuple[str, list[str]]:
    """Like `normalize_text`, also returning the characters that had to be dropped."""
    text = normalize_unicode(text)
    text = expand_numbers(text)
    text, dropped = _clean_characters(text)
    text = canonicalize_diacritics(text)
    text = _tidy_punctuation(text)
    if ta_marbuta_lexicon:
        text = apply_ta_marbuta_lexicon(text, ta_marbuta_lexicon)
    return text, dropped


def subsample_marks(word: str, keep_prob: float, rng: random.Random) -> str:
    """Keep each letter's marks with probability `keep_prob` (all-or-nothing per letter).

    Teaches the model that ANY subset of diacritics is meaningful, so a user
    can fix a single letter (e.g. only "ةْ") without vowelizing the whole word.
    """
    out = []
    for base, marks in iter_clusters(word):
        out.append(base + (marks if marks and rng.random() < keep_prob else ""))
    return "".join(out)


def has_diacritics(text: str) -> bool:
    return any(c in _MARKS for c in text)
