"""English text tables for the verbalizer / ITN: cardinals, years, ordinals,
letter names, currency, time. Kenyan/British conventions (zed, "and" after
hundreds). Letter names are spelled the way a TTS model will pronounce them
("aitch", "double you"), not as bare capitals.
"""

from __future__ import annotations

DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen"]
TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy",
        8: "eighty", 9: "ninety"}
SCALES = [(1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand")]

# Letter names as pronounceable words. Whisper writes these back as capitals,
# so ITN also accepts bare letters. 2026-09-22 spelling eval: "ay" was spoken
# as I and "cue" as "two", so A is "ey" and Q is "queue".
LETTERS = {
    "a": "ey", "b": "bee", "c": "see", "d": "dee", "e": "ee", "f": "ef", "g": "jee",
    "h": "aitch", "i": "eye", "j": "jay", "k": "kay", "l": "el", "m": "em", "n": "en",
    "o": "oh", "p": "pee", "q": "queue", "r": "ar", "s": "ess", "t": "tee", "u": "you",
    "v": "vee", "w": "double you", "x": "ex", "y": "why", "z": "zed",
}

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]

ABBREVIATIONS = {
    "Dr.": "Doctor", "Mr.": "Mister", "Mrs.": "Missus", "Ms.": "Miss", "Prof.": "Professor",
    "St.": "Saint", "Ave.": "Avenue", "Rd.": "Road", "etc.": "et cetera", "e.g.": "for example",
    "i.e.": "that is", "vs.": "versus", "No.": "number", "no.": "number",
}

CURRENCIES = {
    "ksh": ("shilling", "shillings", "cent", "cents"),
    "kes": ("shilling", "shillings", "cent", "cents"),
    "tsh": ("shilling", "shillings", "cent", "cents"),
    "tzs": ("shilling", "shillings", "cent", "cents"),
    "ush": ("shilling", "shillings", "cent", "cents"),
    "ugx": ("shilling", "shillings", "cent", "cents"),
    "$": ("dollar", "dollars", "cent", "cents"),
    "us$": ("dollar", "dollars", "cent", "cents"),
    "usd": ("dollar", "dollars", "cent", "cents"),
    "€": ("euro", "euros", "cent", "cents"),
    "eur": ("euro", "euros", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
    "gbp": ("pound", "pounds", "penny", "pence"),
}

DECIMAL_WORD = "point"
PLUS_WORD = "plus"
PERCENT_WORD = "percent"

# Function words used for per-span language detection.
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "for", "and", "or", "your", "my",
    "our", "their", "his", "her", "this", "that", "these", "those", "at", "on", "in", "please",
    "call", "number", "ticket", "code", "flight", "order", "reference", "booking", "send", "enter",
    "press", "dial", "text", "contact", "office", "departs", "from", "with", "by", "it", "you",
    "we", "they", "be", "will", "can", "not", "if", "when", "before", "after", "until", "next",
    "last", "morning", "evening", "tomorrow", "today", "yesterday", "has", "have", "had", "been",
    "would", "should", "could", "than", "more", "less", "very", "about", "into", "over", "under",
    "account", "password", "pin", "otp", "id", "plate", "confirm", "continue", "word", "message",
}

# Keywords that mark the following digit string as an identifier (read digit by digit).
ID_KEYWORDS = {
    "number", "no", "num", "ticket", "id", "pin", "otp", "code", "account", "acc", "plate", "order",
    "ref", "reference", "booking", "tracking", "invoice", "receipt", "extension", "ext", "paybill",
    "till", "mpesa", "phone", "mobile", "tel", "telephone", "line", "hotline", "call", "dial",
    "whatsapp", "sms", "to", "text", "password", "passcode", "voucher", "serial", "imei", "policy",
    "member", "membership", "registration", "reg", "licence", "license", "passport",
}
YEAR_KEYWORDS = {"in", "since", "year", "by", "from", "until", "before", "after", "during", "of"}


def cardinal(n: int) -> str:
    if n < 0:
        return "minus " + cardinal(-n)
    if n < 10:
        return DIGITS[n]
    if n < 20:
        return TEENS[n - 10]
    if n < 100:
        tens, unit = divmod(n, 10)
        return TENS[tens] if unit == 0 else f"{TENS[tens]} {DIGITS[unit]}"
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        head = f"{DIGITS[hundreds]} hundred"
        return head if rest == 0 else f"{head} and {cardinal(rest)}"
    for value, name in SCALES:
        if n >= value:
            count, rest = divmod(n, value)
            head = f"{cardinal(count)} {name}"
            if rest == 0:
                return head
            joiner = " and " if rest < 100 else " "
            return f"{head}{joiner}{cardinal(rest)}"
    raise ValueError(n)


def year(n: int) -> str:
    """Year reading: 1964 -> nineteen sixty four, 2005 -> two thousand and five,
    2026 -> twenty twenty six, 2000 -> two thousand."""
    if not 1000 <= n <= 9999:
        return cardinal(n)
    hi, lo = divmod(n, 100)
    if lo == 0:
        return f"{cardinal(hi)} hundred" if hi % 10 else cardinal(n)
    if hi % 10 == 0 and lo < 10:  # 2005, 1007
        return cardinal(n)
    if lo < 10:
        return f"{cardinal(hi)} oh {DIGITS[lo]}"
    return f"{cardinal(hi)} {cardinal(lo)}"


_ORDINAL_SPECIAL = {1: "first", 2: "second", 3: "third", 5: "fifth", 8: "eighth", 9: "ninth",
                    12: "twelfth"}


def ordinal(n: int) -> str:
    if n in _ORDINAL_SPECIAL:
        return _ORDINAL_SPECIAL[n]
    if n < 20:
        return cardinal(n) + "th"
    tens, unit = divmod(n, 10)
    if unit == 0:
        if n < 100:
            return TENS[tens][:-1] + "ieth"
        return cardinal(n) + "th"
    head = cardinal(n - unit)
    return f"{head} {ordinal(unit)}"


def digits(s: str) -> str:
    return " ".join(DIGITS[int(c)] for c in s if c.isdigit())


def spell(s: str) -> str:
    out = []
    for c in s:
        if c.isdigit():
            out.append(DIGITS[int(c)])
        elif c.lower() in LETTERS:
            out.append(LETTERS[c.lower()])
    return " ".join(out)


def money(code: str, whole: int, cents: int | None) -> str:
    unit1, unitn, cent1, centn = CURRENCIES[code.lower()]
    text = f"{cardinal(whole)} {unit1 if whole == 1 else unitn}"
    if cents:
        text += f" and {cardinal(cents)} {cent1 if cents == 1 else centn}"
    return text


def time_of_day(h: int, m: int, suffix: str | None = None) -> str:
    if m == 0:
        text = f"{cardinal(h)} o'clock"
    elif m < 10:
        text = f"{cardinal(h)} oh {DIGITS[m]}"
    else:
        text = f"{cardinal(h)} {cardinal(m)}"
    if suffix:
        text += " " + " ".join(LETTERS[c] for c in suffix.lower())
    return text


def date(d: int, month: int, y: int) -> str:
    return f"{ordinal(d)} of {MONTHS[month - 1]} {year(y)}"


def percent(num_text: str) -> str:
    return f"{num_text} {PERCENT_WORD}"
