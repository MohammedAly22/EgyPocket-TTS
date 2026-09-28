import pytest

from egypocket.text.frontend import normalize_text
from egypocket.text.tokenizer import UNK, build_tokenizer_json, load_tokenizer, vocabulary


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    size = build_tokenizer_json(path)
    assert size == len(vocabulary())
    return load_tokenizer(path)


def test_roundtrip(tokenizer):
    text = normalize_text("رحت مدينةْ نصر امبارح, والجو كان حلو قوي.")
    ids = tokenizer.encode(text).ids
    assert UNK not in [tokenizer.id_to_token(i) for i in ids]
    assert tokenizer.decode(ids) == text


def test_space_is_a_token(tokenizer):
    ids = tokenizer.encode("ا ب").ids
    assert [tokenizer.id_to_token(i) for i in ids] == ["ا", " ", "ب"]


def test_ta_marbuta_tokens_distinct(tokenizer):
    forms = ["مدينة", "مدينةْ", "مدينةَ", "مدينةِ", "مدينةُ"]
    last = [tokenizer.encode(f).ids[-1] for f in forms]
    assert len(set(last)) == len(forms)
    assert tokenizer.id_to_token(tokenizer.encode("مدينةْ").ids[-1]) == "ةْ"
    # plain ة is a single token and the preceding letters are shared
    assert tokenizer.encode("مدينةْ").ids[:-1] == tokenizer.encode("مدينة").ids[:-1]


def test_diacritics_are_separate_tokens(tokenizer):
    plain = tokenizer.encode("كتب").ids
    diac = tokenizer.encode("كَتَبَ").ids
    assert len(diac) == 2 * len(plain)
    assert diac[::2] == plain


def test_egyptian_marker_and_shadda(tokenizer):
    toks = [tokenizer.id_to_token(i) for i in tokenizer.encode(normalize_text("ق؞َّ")).ids]
    assert toks == ["ق", "~", "ّ", "َ"]
