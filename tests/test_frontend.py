import random

from egypocket.text.frontend import (
    apply_ta_marbuta_lexicon,
    normalize_text,
    normalize_text_verbose,
    strip_diacritics,
    subsample_marks,
)
from egypocket.text.numbers import expand_numbers, number_to_words


def test_plain_corpus_text_is_unchanged():
    text = "السلام عليكم ورحمه الله وبركاته اول ما بنسمع حاجه عن المصريين القدماء"
    assert normalize_text(text) == text


def test_alef_unification_keeps_diacritics():
    assert normalize_text("أنا إبن آدم") == "انا ابن ادم"
    assert normalize_text("أَنَا") == "اَنَا"
    assert normalize_text("إِنْتَ") == "اِنْتَ"


def test_ta_marbuta_forms_are_distinct_after_frontend():
    forms = ["مدينة", "مدينةْ", "مدينةَ", "مدينةِ", "مدينةُ", "مدينةً"]
    outs = [normalize_text(f) for f in forms]
    assert len(set(outs)) == len(forms)
    assert outs == forms


def test_diacritics_not_stripped():
    assert normalize_text("مَدِينَةْ") == "مَدِينَةْ"
    assert strip_diacritics(normalize_text("مَدِينَةْ")) == "مدينة"


def test_orphan_marks_dropped_and_order_canonical():
    assert normalize_text("َ مدينة") == "مدينة"
    assert normalize_text("شَّ") == normalize_text("شَّ") == "شَّ"
    # two vowels on one letter: the last one wins
    assert normalize_text("بَُ") == "بُ"


def test_egyptian_marker_kept():
    assert normalize_text("ق~َابلت") == "ق~َابلت"
    assert normalize_text("ق؞َابلت") == "ق~َابلت"


def test_punctuation_reduced():
    assert normalize_text("ازيك؟ عامل ايه!") == "ازيك. عامل ايه."
    assert normalize_text("اه، طبعا؛ ماشي: تمام") == "اه, طبعا, ماشي, تمام"
    assert normalize_text("«قال» (كده) - خلاص") == "قال كده خلاص"
    assert normalize_text("،، يلا..") == "يلا."


def test_latin_dropped_and_reported():
    text, dropped = normalize_text_verbose("انا بحب Python جدا")
    assert text == "انا بحب جدا"
    assert "P" in dropped


def test_numbers_egyptian():
    assert number_to_words(3) == "تلاتة"
    assert number_to_words(12) == "اتناشر"
    assert number_to_words(25) == "خمسة وعشرين"
    assert number_to_words(200) == "ميتين"
    assert number_to_words(2000) == "الفين"
    assert number_to_words(1889) == "الف وتمنمية وتسعة وتمانين"
    assert expand_numbers("سنه 1994 مثلا") == "سنه الف وتسعمية واربعة وتسعين مثلا"
    assert expand_numbers("حوالي 20 دقيقه") == "حوالي عشرين دقيقه"
    assert expand_numbers("استهلكت 1.5% من") == "استهلكت واحد فاصلة خمسة في المية من"
    assert expand_numbers("و000 مستوى") == "و صفر مستوى"
    assert normalize_text("عندي ٣ كتب") == "عندي تلاتة كتب"
    assert expand_numbers("الاجتماع الساعة 3:30") == "الاجتماع الساعة تلاتة وتلاتين دقيقة"
    assert expand_numbers("نتقابل 5:00") == "نتقابل الساعة خمسة"


def test_ta_marbuta_lexicon():
    lex = {"المدينه", "حاجه"}
    assert apply_ta_marbuta_lexicon("رحت المدينه", lex) == "رحت المدينة"
    assert apply_ta_marbuta_lexicon("المَدِينَهْ", lex) == "المَدِينَةْ"
    assert apply_ta_marbuta_lexicon("كتابه", lex) == "كتابه"
    assert normalize_text("حاجه حلوه", ta_marbuta_lexicon=lex) == "حاجة حلوه"


def test_subsample_marks():
    rng = random.Random(0)
    word = "مَدِينَةْ"
    assert subsample_marks(word, 1.0, rng) == word
    assert subsample_marks(word, 0.0, rng) == "مدينة"
    for _ in range(20):
        out = subsample_marks(word, 0.5, rng)
        assert strip_diacritics(out) == "مدينة"
