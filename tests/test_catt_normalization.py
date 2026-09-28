"""normalize_catt_output: ^ < > removed, ؞ -> ~, diacritics preserved."""

import pytest

from egypocket.text.catt import normalize_catt_output
from egypocket.text.chars import DIACRITICS
from egypocket.text.frontend import strip_diacritics


SHADDA_FATHA = "َّ"  # canonical order: shadda before the vowel


@pytest.mark.parametrize(
    "raw, expected",
    [
        # Real CATT outputs (ECA model, fatha written before shadda) and their
        # expected clean, canonical forms.
        ("الت^َّق؞ْسِيطْ", f"الت{SHADDA_FATHA}ق~ْسِيطْ"),
        ("ن^َارْ", "نَارْ"),
        ("ب<يتْ", "بيتْ"),
        ("الت^َّرَاْب<يز^َه", f"الت{SHADDA_FATHA}رَاْبيزَه"),
        ("وِق؞َاْبِلْتِ", "وِق~َاْبِلْتِ"),
        ("اْلل^َّه", f"اْلل{SHADDA_FATHA}ه"),
        ("خِدْمِةْ", "خِدْمِةْ"),
        ("مَرْحَبًا", "مَرْحَبًا"),
    ],
)
def test_catt_examples(raw, expected):
    assert normalize_catt_output(raw) == expected


def test_special_symbols_never_survive():
    raw = "ألسَّلَامُ عَلَيْكُمْ وَب^َرَكَاتُه الْم^َصْرِيِّينْ لِلَّاهْرَاْم^َاتْ ب<يت >ب ^^ <<>>"
    out = normalize_catt_output(raw)
    for sym in "^<>":
        assert sym not in out
    assert "؞" not in out


def test_fullwidth_lookalikes_removed():
    assert normalize_catt_output("ت＾َ ب＜ي ＞ر") == "تَ بي ر"


def test_mothalatha_becomes_tilde():
    assert normalize_catt_output("ق؞") == "ق~"
    assert normalize_catt_output("ق؞ْ").count("~") == 1


def test_diacritics_are_preserved():
    raw = "مَدِينَةْ مَدِينَة"
    out = normalize_catt_output(raw)
    assert out == raw
    assert sum(out.count(d) for d in DIACRITICS) == sum(raw.count(d) for d in DIACRITICS)


def test_ta_marbuta_forms_stay_distinct():
    forms = ["مَدِينَة", "مَدِينَةْ", "مَدِينَةَ", "مَدِينَةِ", "مَدِينَةُ"]
    outs = [normalize_catt_output(f) for f in forms]
    assert len(set(outs)) == len(forms)
    assert outs == forms


def test_mark_order_is_canonical():
    # fatha+shadda and shadda+fatha are the same thing -> one representation.
    assert normalize_catt_output("تَّ") == normalize_catt_output("تَّ") == "تَّ"
    # the ~ marker goes right after its letter
    assert normalize_catt_output("قْ؞") == "ق~ْ"


def test_letters_unchanged():
    raw = "وِنْشُوفْ مَجْمُوعَه مْنَ اْغْرَبْ الْحَاْجَاتْ اللِّيْ كَانْ بِيِعْمِلْهَا الْم^َصْرِيِّينْ"
    plain = "ونشوف مجموعه من اغرب الحاجات اللي كان بيعملها المصريين"
    assert strip_diacritics(normalize_catt_output(raw)) == plain


def test_deterministic():
    raw = "الت^َّق؞ْسِيطْ ب<يتْ"
    assert normalize_catt_output(raw) == normalize_catt_output(raw)
    assert normalize_catt_output(normalize_catt_output(raw)) == normalize_catt_output(raw)
