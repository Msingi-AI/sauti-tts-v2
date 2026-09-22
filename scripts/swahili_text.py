"""Swahili text utilities: number verbalization, letter names and normalization.

Number verbalization covers plain cardinals (and decimals via "nukta").
Noun-class agreement ("watu wawili", "mlango wa kwanza") requires context
and is out of scope for this rule layer; sentences relying on it should be
pre-verbalized by hand in the training/eval data.

`normalize_for_tts` / `normalize_for_asr` delegate span classification (phone
numbers, codes, acronyms, money, ...) to `verbalizer.py`, which uses the tables
in this module for Swahili spans and `english_text.py` for English spans.
"""

from __future__ import annotations

import re

UNITS = [
    "sifuri", "moja", "mbili", "tatu", "nne",
    "tano", "sita", "saba", "nane", "tisa",
]
DIGITS = UNITS
TENS = {
    1: "kumi", 2: "ishirini", 3: "thelathini", 4: "arobaini", 5: "hamsini",
    6: "sitini", 7: "sabini", 8: "themanini", 9: "tisini",
}

ABBREVIATIONS = {
    "Dkt.": "Daktari",
    "Bw.": "Bwana",
    "Bi.": "Bibi",
    "Prof.": "Profesa",
    "Mh.": "Mheshimiwa",
    "n.k.": "na kadhalika",
    "k.m.": "kwa mfano",
    "k.v.": "kama vile",
}

# Letter names as Kenyan Swahili speakers read acronyms and codes: the English
# letter names, written in Swahili orthography so a Swahili-trained model
# pronounces them right ("kei aa ei" for KRA). NEEDS NATIVE REVIEW.
LETTERS = {
    "a": "ei", "b": "bii", "c": "sii", "d": "dii", "e": "ii", "f": "efu", "g": "jii",
    "h": "eichi", "i": "ai", "j": "jei", "k": "kei", "l": "eli", "m": "emu", "n": "eni",
    "o": "oo", "p": "pii", "q": "kyuu", "r": "aa", "s": "esi", "t": "tii", "u": "yuu",
    "v": "vii", "w": "dabuliyu", "x": "eksi", "y": "wai", "z": "zedi",
}
# Native Swahili alphabet names (classroom reading). Available via
# verbalize(..., letter_style="sw_native"); not the default.
LETTERS_NATIVE = {
    "a": "a", "b": "be", "c": "che", "d": "de", "e": "e", "f": "efe", "g": "ge", "h": "he",
    "i": "i", "j": "je", "k": "ke", "l": "le", "m": "me", "n": "ne", "o": "o", "p": "pe",
    "q": "kyu", "r": "re", "s": "se", "t": "te", "u": "u", "v": "ve", "w": "we", "x": "eksi",
    "y": "ye", "z": "zeti",
}

MONTHS = ["Januari", "Februari", "Machi", "Aprili", "Mei", "Juni", "Julai", "Agosti",
          "Septemba", "Oktoba", "Novemba", "Desemba"]

CURRENCIES = {
    "ksh": ("shilingi", "senti"), "kes": ("shilingi", "senti"), "tsh": ("shilingi", "senti"),
    "tzs": ("shilingi", "senti"), "ush": ("shilingi", "senti"), "ugx": ("shilingi", "senti"),
    "$": ("dola", "senti"), "us$": ("dola", "senti"), "usd": ("dola", "senti"),
    "€": ("yuro", "senti"), "eur": ("yuro", "senti"), "£": ("pauni", "peni"), "gbp": ("pauni", "peni"),
}

DECIMAL_WORD = "nukta"
PLUS_WORD = "plus"
PERCENT_WORD = "asilimia"

STOPWORDS = {
    "na", "ya", "wa", "za", "la", "cha", "vya", "kwa", "ni", "katika", "kwenye", "yako", "yangu",
    "yake", "yetu", "yenu", "yao", "wako", "wangu", "wake", "wetu", "hii", "hiyo", "ile", "hizi",
    "hizo", "huu", "huo", "sasa", "tafadhali", "simu", "nambari", "namba", "tiketi", "saa", "tarehe",
    "shilingi", "bei", "mwaka", "kila", "hadi", "kutoka", "sana", "tuma", "piga", "ndege", "gari",
    "ili", "au", "lakini", "pia", "bado", "tu", "ndio", "ndiyo", "mimi", "wewe", "yeye", "sisi",
    "ninyi", "wao", "hapa", "huko", "leo", "kesho", "jana", "asubuhi", "jioni", "usiku", "mchana",
    "weka", "bonyeza", "ujumbe", "akaunti", "msaada", "kodi", "kisha", "sawa", "basi", "kama",
    "kuhusu", "baada", "kabla", "mpaka", "ama", "je", "si", "sio", "siyo", "hakuna", "kuna",
    "watu", "mtu", "kitu", "vitu", "neno", "maneno", "wasiliana", "nasi", "nawe", "kwenda",
    "itaondoka", "inaondoka", "utaanza", "itaanza",
}

ID_KEYWORDS = {
    "nambari", "namba", "no", "tiketi", "id", "pin", "otp", "kodi", "msimbo", "akaunti", "plate",
    "oda", "ref", "risiti", "ankara", "paybill", "till", "mpesa", "simu", "kwa", "usajili",
    "pasipoti", "kitambulisho", "leseni", "mwanachama", "uanachama", "nenosiri", "siri", "vocha",
    "ticket", "number", "code", "account", "order", "reference", "booking", "password", "line",
}


def cardinal(n: int) -> str:
    """Verbalize an integer as a Swahili cardinal (e.g. 135 -> 'mia moja thelathini na tano')."""
    if n < 0:
        return "hasi " + cardinal(-n)
    if n < 10:
        return UNITS[n]
    if n < 100:
        tens, unit = divmod(n, 10)
        word = TENS[tens]
        return word if unit == 0 else f"{word} na {UNITS[unit]}"
    if n < 1_000:
        hundreds, rest = divmod(n, 100)
        word = f"mia {UNITS[hundreds]}"
    elif n < 1_000_000:
        thousands, rest = divmod(n, 1_000)
        word = f"elfu {cardinal(thousands)}"
    elif n < 1_000_000_000:
        millions, rest = divmod(n, 1_000_000)
        word = f"milioni {cardinal(millions)}"
    else:
        billions, rest = divmod(n, 1_000_000_000)
        word = f"bilioni {cardinal(billions)}"
    if rest == 0:
        return word
    # "na" joins a final single group: units (na tano) or round tens (na hamsini),
    # but not a compound remainder ("mia moja thelathini na tano").
    if rest < 10 or (rest < 100 and rest % 10 == 0):
        return f"{word} na {cardinal(rest)}"
    return f"{word} {cardinal(rest)}"


def year(n: int) -> str:
    """Swahili reads years as full cardinals (1964 -> elfu moja mia tisa sitini na nne)."""
    return cardinal(n)


def ordinal(n: int) -> str:
    # Swahili ordinals need a class prefix ("wa pili", "ya tatu") that lives on the
    # preceding word; emit the bare stem.
    return {1: "kwanza", 2: "pili"}.get(n, cardinal(n))


def digits(s: str) -> str:
    return " ".join(UNITS[int(c)] for c in s if c.isdigit())


def spell(s: str, letters: dict[str, str] | None = None) -> str:
    table = letters or LETTERS
    out = []
    for c in s:
        if c.isdigit():
            out.append(UNITS[int(c)])
        elif c.lower() in table:
            out.append(table[c.lower()])
    return " ".join(out)


def money(code: str, whole: int, cents: int | None) -> str:
    unit, cent = CURRENCIES[code.lower()]
    text = f"{unit} {cardinal(whole)}"
    if cents:
        text += f" na {cent} {cardinal(cents)}"
    return text


def time_of_day(h: int, m: int, suffix: str | None = None) -> str:
    """Clock reading without the Swahili six-hour offset: '7:30' -> 'saa saba na nusu'.
    Written digits in Kenyan text almost always mean the English clock; callers who
    want 'saa moja na nusu' must convert before verbalizing."""
    text = f"saa {cardinal(h)}"
    if m == 30:
        text += " na nusu"
    elif m == 15:
        text += " na robo"
    elif m:
        text += f" na dakika {cardinal(m)}"
    if suffix:
        text += " " + {"am": "asubuhi", "pm": "jioni"}.get(suffix.lower(), "")
    return text.strip()


def date(d: int, month: int, y: int) -> str:
    return f"tarehe {cardinal(d)} {MONTHS[month - 1]} {year(y)}"


def percent(num_text: str) -> str:
    return f"{PERCENT_WORD} {num_text}"


def _verbalize_match(match: re.Match) -> str:
    token = match.group(0).replace(",", "")
    if "." in token:
        whole, frac = token.split(".", 1)
        frac_words = " ".join(UNITS[int(d)] for d in frac if d.isdigit())
        return f"{cardinal(int(whole))} nukta {frac_words}"
    return cardinal(int(token))


def expand_numbers(text: str) -> str:
    """Replace digit sequences (incl. 3,500-style separators and decimals) with cardinals.
    Legacy helper: treats every digit run as a quantity. Prefer verbalizer.verbalize."""
    return re.sub(r"\d[\d,]*(?:\.\d+)?", _verbalize_match, text)


def expand_abbreviations(text: str) -> str:
    for abbr, full in ABBREVIATIONS.items():
        text = text.replace(abbr, full)
    return text


def normalize_for_tts(text: str, lang: str = "auto") -> str:
    """Normalization applied to training/eval text fed to a TTS model.

    lang: 'sw', 'en' or 'auto' (per-span detection, Swahili fallback)."""
    from verbalizer import verbalize
    return verbalize(text, lang=lang)


def normalize_for_asr(text: str, lang: str = "auto") -> str:
    """Normalization applied to both reference and hypothesis before WER/CER."""
    text = normalize_for_tts(text, lang=lang).lower()
    # Keep the ng' apostrophe (a real Swahili grapheme), drop other punctuation.
    text = re.sub(r"(?<![gG])'", " ", text)
    text = re.sub(r"[^\w' ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
