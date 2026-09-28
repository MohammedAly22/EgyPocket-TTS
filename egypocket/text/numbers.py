"""Egyptian-Arabic verbalization of digits, percentages, currencies, dates and times.

Adapted from the number handling of CATT's `ECAPreProcessor`
(catt_tashkeel/preprocessing/eca_preprocessor.py) so that the TTS frontend
works without CATT at inference time. Numbers are read the Egyptian way
(اتنين، تلاتة، مية، ميتين، الفين ...).
"""

import re

ONES = {
    0: "صفر",
    1: "واحد",
    2: "اتنين",
    3: "تلاتة",
    4: "اربعة",
    5: "خمسة",
    6: "ستة",
    7: "سبعة",
    8: "تمانية",
    9: "تسعة",
}
TEENS = {
    10: "عشرة",
    11: "حداشر",
    12: "اتناشر",
    13: "تلتاشر",
    14: "اربعتاشر",
    15: "خمستاشر",
    16: "ستاشر",
    17: "سبعتاشر",
    18: "تمنتاشر",
    19: "تسعتاشر",
}
TENS = {
    20: "عشرين",
    30: "تلاتين",
    40: "اربعين",
    50: "خمسين",
    60: "ستين",
    70: "سبعين",
    80: "تمانين",
    90: "تسعين",
}
HUNDREDS = {
    100: "مية",
    200: "ميتين",
    300: "تلتمية",
    400: "اربعمية",
    500: "خمسمية",
    600: "ستمية",
    700: "سبعمية",
    800: "تمنمية",
    900: "تسعمية",
}
MONTHS = {
    1: "يناير",
    2: "فبراير",
    3: "مارس",
    4: "ابريل",
    5: "مايو",
    6: "يونيو",
    7: "يوليو",
    8: "اغسطس",
    9: "سبتمبر",
    10: "اكتوبر",
    11: "نوفمبر",
    12: "ديسمبر",
}
CURRENCIES = {"$": "دولار", "€": "يورو", "£": "جنيه استرليني"}

_AR = "؀-ۿ"


def number_to_words(num: int) -> str:
    """Egyptian-Arabic words for a non-negative integer."""
    if num < 0:
        return "سالب " + number_to_words(-num)
    if num < 10:
        return ONES[num]
    if num < 20:
        return TEENS[num]
    if num < 100:
        tens, ones = (num // 10) * 10, num % 10
        return TENS[tens] if ones == 0 else f"{ONES[ones]} و{TENS[tens]}"
    if num < 1000:
        hundreds, rest = (num // 100) * 100, num % 100
        return HUNDREDS[hundreds] if rest == 0 else f"{HUNDREDS[hundreds]} و{number_to_words(rest)}"
    if num < 1_000_000:
        thousands, rest = num // 1000, num % 1000
        if thousands == 1:
            head = "الف"
        elif thousands == 2:
            head = "الفين"
        elif thousands <= 10:
            head = number_to_words(thousands) + " تلاف"
        else:
            head = number_to_words(thousands) + " الف"
        return head if rest == 0 else f"{head} و{number_to_words(rest)}"
    if num < 1_000_000_000:
        millions, rest = num // 1_000_000, num % 1_000_000
        if millions == 1:
            head = "مليون"
        elif millions == 2:
            head = "اتنين مليون"
        else:
            head = number_to_words(millions) + " مليون"
        return head if rest == 0 else f"{head} و{number_to_words(rest)}"
    return " ".join(ONES[int(d)] for d in str(num))


def _digits_to_words(digits: str) -> str:
    # Very long digit strings (phone numbers, codes) are read digit by digit.
    if len(digits) > 9:
        return " ".join(ONES[int(d)] for d in digits)
    return number_to_words(int(digits))


def _decimal_to_words(integer: str, fraction: str) -> str:
    return f"{_digits_to_words(integer)} فاصلة {_digits_to_words(fraction)}"


def _number_words(token: str) -> str:
    if "." in token:
        integer, fraction = token.split(".", 1)
        return _decimal_to_words(integer or "0", fraction)
    return _digits_to_words(token)


def expand_numbers(text: str) -> str:
    """Replace every ASCII-digit number (and %, $, €, £, dates, times) by words.

    Expects Arabic-Indic digits to be already mapped to ASCII.
    """
    if not re.search(r"\d", text):
        return text
    # 1,000 -> 1000 (thousands separators), both ASCII and Arabic comma forms.
    text = re.sub(r"(?<=\d)[,٬](?=\d{3}(?!\d))", "", text)
    # Arabic decimal separator.
    text = re.sub(r"(?<=\d)٫(?=\d)", ".", text)
    # Detach digits glued to Arabic letters: "و000" -> "و 000".
    text = re.sub(rf"([{_AR}])(\d)", r"\1 \2", text)
    text = re.sub(rf"(\d)([{_AR}])", r"\1 \2", text)

    def date_dmy(m: re.Match[str]) -> str:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not 1 <= month <= 12:
            return m.group(0)
        return f"{number_to_words(day)} {MONTHS[month]} سنة {number_to_words(year)}"

    def date_ymd(m: re.Match[str]) -> str:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not 1 <= month <= 12:
            return m.group(0)
        return f"{number_to_words(day)} {MONTHS[month]} سنة {number_to_words(year)}"

    text = re.sub(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", date_dmy, text)
    text = re.sub(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b", date_ymd, text)

    def clock(m: re.Match[str]) -> str:
        hour, minute = int(m.group(2)), int(m.group(3))
        if hour > 24 or minute > 59:
            return m.group(0)
        # "الساعة" is added unless the text already says it ("الساعة 3:30").
        prefix = m.group(1) or "الساعة "
        if minute == 0:
            return f"{prefix}{number_to_words(hour)}"
        return f"{prefix}{number_to_words(hour)} و{number_to_words(minute)} دقيقة"

    text = re.sub(r"(الساع[ةه]\s+)?\b(\d{1,2}):(\d{2})\b", clock, text)

    text = re.sub(
        r"(\d+(?:\.\d+)?)\s*[%٪]", lambda m: f"{_number_words(m.group(1))} في المية", text
    )
    for symbol, word in CURRENCIES.items():
        sym = re.escape(symbol)
        text = re.sub(
            rf"{sym}\s*(\d+(?:\.\d+)?)", lambda m, w=word: f"{_number_words(m.group(1))} {w}", text
        )
        text = re.sub(
            rf"(\d+(?:\.\d+)?)\s*{sym}", lambda m, w=word: f"{_number_words(m.group(1))} {w}", text
        )
        text = text.replace(symbol, f" {word} ")

    text = re.sub(r"\d+\.\d+", lambda m: _number_words(m.group(0)), text)
    text = re.sub(r"\d+", lambda m: _digits_to_words(m.group(0)), text)
    return re.sub(r"\s+", " ", text).strip()
