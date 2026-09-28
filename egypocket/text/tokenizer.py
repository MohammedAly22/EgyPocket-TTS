"""Character-level tokenizer in HuggingFace `tokenizers` JSON format.

Pocket TTS looks text up in a table (LUTConditioner) and loads `tokenizer.json`
files natively, so this tokenizer plugs into both training and the stock
pocket-tts inference code.

Why characters and not sentencepiece: every diacritic is its own token, so a
diacritized word shares all its letter tokens with the plain spelling and the
marks are pure, additive pronunciation information. Subword pieces would turn
"مَدِينَةْ" into pieces that almost never occur in training.

ة is special: ة followed by a vowel mark / tanween / sukun is ONE token
(ةْ, ةَ, ةِ, ةُ, ةً, ةٌ, ةٍ), so the ending's pronunciation (h / t / tan ...) is an
explicit symbol with its own embedding rather than an interaction the model
has to learn across two positions.
"""

import json
from pathlib import Path

from egypocket.text.chars import (
    COMMA,
    DIACRITICS,
    EGY_MARKER,
    LETTERS,
    PERIOD,
    SHADDA,
    TA_MARBUTA,
    VOWEL_MARKS,
)

UNK = "[UNK]"
SPACE = " "
TA_MARBUTA_TOKENS = tuple(TA_MARBUTA + m for m in VOWEL_MARKS)
# ة + one vowel-like mark (not shadda), else any single character.
SPLIT_PATTERN = TA_MARBUTA + "[" + "".join(m for m in VOWEL_MARKS) + "]|[\\s\\S]"


def vocabulary() -> list[str]:
    """Token strings, index = token id."""
    return (
        [UNK, SPACE, COMMA, PERIOD, EGY_MARKER]
        + list(LETTERS)
        + list(DIACRITICS)
        + list(TA_MARBUTA_TOKENS)
    )


def build_tokenizer_json(path: str | Path) -> int:
    """Write tokenizer.json and return its vocabulary size (= lookup_table.n_bins)."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

    vocab = {tok: i for i, tok in enumerate(vocabulary())}
    assert len(vocab) == len(vocabulary()), "duplicate token"
    assert SHADDA in vocab
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token=UNK))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated")
    tokenizer.decoder = decoders.Fuse()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    size = tokenizer.get_vocab_size()
    assert size == len(vocab), (size, len(vocab))
    return size


def load_tokenizer(path: str | Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def vocab_size(path: str | Path) -> int:
    with open(path, encoding="utf-8") as f:
        return len(json.load(f)["model"]["vocab"])
