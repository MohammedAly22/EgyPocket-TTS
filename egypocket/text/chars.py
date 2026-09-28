"""Character inventory of the EgyPocket text frontend.

Everything that reaches the acoustic model is built from the symbols defined
here: Arabic letters (after alef unification), the eight Arabic diacritics,
the Egyptian pronunciation marker "~" (CATT's ؞) and two punctuation marks.
"""

# Diacritics (tashkeel). They are pronunciation control, never cosmetic.
FATHATAN = "ً"
DAMMATAN = "ٌ"
KASRATAN = "ٍ"
FATHA = "َ"
DAMMA = "ُ"
KASRA = "ِ"
SHADDA = "ّ"
SUKUN = "ْ"

# Vowel-like marks: at most one of them is kept per letter.
VOWEL_MARKS = (FATHA, DAMMA, KASRA, FATHATAN, DAMMATAN, KASRATAN, SUKUN)
DIACRITICS = VOWEL_MARKS + (SHADDA,)
DIACRITICS_SET = frozenset(DIACRITICS)

# CATT's Egyptian "mothalatha" marker (U+061E) is rewritten to "~" and kept
# as a pronunciation marker attached to the letter it follows.
CATT_MOTHALATHA = "؞"
EGY_MARKER = "~"
# CATT symbols that must never reach the training text.
CATT_REMOVED = ("^", "<", ">")

TATWEEL = "ـ"
TA_MARBUTA = "ة"  # ة
HA = "ه"  # ه
ALEF = "ا"  # ا

# Letters the model sees. Hamza-on-alef forms (أ إ آ ٱ) are unified to ا
# because the training transcripts are written that way.
LETTERS = (
    "ء"  # ء
    "ؤ"  # ؤ
    "ئ"  # ئ
    "ا"  # ا
    "ب"  # ب
    "ة"  # ة
    "ت"  # ت
    "ث"  # ث
    "ج"  # ج
    "ح"  # ح
    "خ"  # خ
    "د"  # د
    "ذ"  # ذ
    "ر"  # ر
    "ز"  # ز
    "س"  # س
    "ش"  # ش
    "ص"  # ص
    "ض"  # ض
    "ط"  # ط
    "ظ"  # ظ
    "ع"  # ع
    "غ"  # غ
    "ف"  # ف
    "ق"  # ق
    "ك"  # ك
    "ل"  # ل
    "م"  # م
    "ن"  # ن
    "ه"  # ه
    "و"  # و
    "ى"  # ى
    "ي"  # ي
)
LETTERS_SET = frozenset(LETTERS)

# Punctuation kept in the model text. The corpus has none, so the data
# pipeline inserts "," and "." from measured pauses; at inference every
# sentence-final mark becomes "." and every clause mark becomes ",".
COMMA = ","
PERIOD = "."
PUNCTUATION = (COMMA, PERIOD)

# Letter rewrites applied before anything else (after NFKC).
LETTER_MAP = {
    "أ": ALEF,  # أ
    "إ": ALEF,  # إ
    "آ": ALEF,  # آ
    "ٱ": ALEF,  # ٱ (wasla)
    "ٰ": ALEF,  # dagger alef
    "ٲ": ALEF,
    "ٳ": ALEF,
    "ڤ": "ف",  # ڤ -> ف
    "پ": "ب",  # پ -> ب
    "چ": "ج",  # چ -> ج
    "گ": "ج",  # گ -> ج (Egyptian ج is /g/)
    "ک": "ك",  # ک -> ك
    "ی": "ي",  # ی -> ي
    "ے": "ي",  # ے -> ي
    "ہ": "ه",  # ہ -> ه
    "ە": "ه",  # ە -> ه
    "ۀ": "ه",  # ۀ -> ه
}

ARABIC_INDIC_DIGITS = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
PERSIAN_DIGITS = {ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")}

# Invisible characters that only confuse tokenization.
INVISIBLE = "​‌‍‎‏‪‫‬‭‮⁦⁧⁨⁩﻿­"
