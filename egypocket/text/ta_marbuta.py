"""Restore taa marbuta (ة) in transcripts that spell it as ه.

The Masri-100h transcripts are "orthography unified": every ة is written ه
(حاجه، المدينه، مجموعه). A model trained on that text never sees ة, so
ة / ةْ / ةَ ... could not act as pronunciation controls. We restore ة per word
TYPE using CATT's vowelization of the unrestored corpus as evidence:

* the letter before a final ه carries a FATHA   -> taa marbuta  (حَاجَه، مَرَّه، شْوَيَّه)
* the letter before a final ه carries a DAMMA   -> pronoun ه     (كِتَابُه، عَنْدُه)
* anything else                                  -> no vote       (فِيه، عَلَيْه، مُنَبِّه)

Votes are pooled over every occurrence of a word type and a type is restored
when at least `min_ratio` of its votes say ة. Words whose ه is a pronounced
/h/ (الله، اتجه، شبه ...) are protected, and words ending in "اه" are left alone
because the alef hides the evidence (معاه vs الحياه).

Wrongly restoring a word whose ه sounds like "a" (كده -> كدة) is harmless: both
spellings are in use and read the same way. The restored types are saved as a
lexicon that the inference frontend applies too, so user text is mapped
exactly like the training text.
"""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from egypocket.text.chars import ALEF, DAMMA, FATHA, HA, LETTERS_SET, TA_MARBUTA
from egypocket.text.frontend import canonical_marks, iter_clusters, strip_diacritics

# Stems whose final ه is a root consonant pronounced /h/, never a taa marbuta.
PROTECTED_STEMS = frozenset(
    {
        "اله",
        "لاه",
        "شبه",
        "وجه",
        "اتجه",
        "يتجه",
        "بيتجه",
        "هيتجه",
        "تتجه",
        "بتتجه",
        "متجه",
        "واجه",
        "يواجه",
        "بيواجه",
        "هيواجه",
        "تواجه",
        "بتواجه",
        "نواجه",
        "بنواجه",
        "توجه",
        "يتوجه",
        "بيتوجه",
        "اتوجه",
        "انتبه",
        "ينتبه",
        "نبه",
        "ينبه",
        "منبه",
        "كره",
        "يكره",
        "بيكره",
        "اشتبه",
        "يشتبه",
        "مشتبه",
        "تشابه",
        "متشابه",
        "مشابه",
        "شابه",
        "فقه",
        "تافه",
        "نزيه",
        "وجيه",
        "سفيه",
        "كريه",
        "مكروه",
        "مشبوه",
        "مشوه",
    }
)
# Clitic prefixes stripped (longest first) before the stem lookup.
_CLITICS = ("وبال", "وكال", "وال", "فال", "بال", "كال", "ولل", "فلل", "لل", "ال", "و", "ف", "ب", "ك", "ل")


def _stems(word: str) -> set[str]:
    stems = {word}
    for clitic in _CLITICS:
        if word.startswith(clitic) and len(word) - len(clitic) >= 2:
            stems.add(word[len(clitic) :])
    return stems


def is_candidate(word: str) -> bool:
    """A plain word whose final ه might be a taa marbuta."""
    if len(word) < 3 or not word.endswith(HA) or word[-2] == ALEF:
        return False
    if any(c not in LETTERS_SET for c in word):
        return False
    if "الله" in word or word.endswith("لله"):
        return False
    return not (_stems(word) & PROTECTED_STEMS)


def vote(diacritized_word: str) -> int:
    """+1 (ة), -1 (pronoun ه) or 0 from CATT's mark on the letter before the final ه."""
    letters = [(b, m) for b, m in iter_clusters(diacritized_word) if b in LETTERS_SET]
    if len(letters) < 2 or letters[-1][0] != HA:
        return 0
    marks = canonical_marks(*letters[-2])
    vowel = marks[-1] if marks else ""
    if vowel == FATHA:
        return 1
    if vowel == DAMMA:
        return -1
    return 0


@dataclass
class TaMarbutaVotes:
    positive: Counter[str] = field(default_factory=Counter)
    negative: Counter[str] = field(default_factory=Counter)
    abstain: Counter[str] = field(default_factory=Counter)

    def add(self, plain_text: str, diacritized_text: str | None):
        if diacritized_text is None:
            return
        plain_words = plain_text.split()
        diac_words = diacritized_text.split()
        if len(plain_words) != len(diac_words):
            return
        for plain, diac in zip(plain_words, diac_words, strict=True):
            if not is_candidate(plain) or strip_diacritics(diac) != plain:
                continue
            v = vote(diac)
            if v > 0:
                self.positive[plain] += 1
            elif v < 0:
                self.negative[plain] += 1
            else:
                self.abstain[plain] += 1

    def lexicon(self, min_ratio: float = 0.6) -> set[str]:
        restored = set()
        for word, pos in self.positive.items():
            neg = self.negative.get(word, 0)
            if pos / (pos + neg) >= min_ratio:
                restored.add(word)
        return restored


def restore(text: str, lexicon: set[str] | frozenset[str]) -> str:
    """Rewrite the final ه of every lexicon word in a plain text to ة."""
    return " ".join(
        w[:-1] + TA_MARBUTA if w in lexicon else w for w in text.split()
    )


def build_lexicon(
    pairs: Iterable[tuple[str, str | None]], min_ratio: float = 0.6
) -> tuple[set[str], TaMarbutaVotes]:
    votes = TaMarbutaVotes()
    for plain, diac in pairs:
        votes.add(plain, diac)
    return votes.lexicon(min_ratio), votes
