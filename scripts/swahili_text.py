"""Swahili text utilities: number verbalization and normalization.

Number verbalization covers plain cardinals (and decimals via "nukta").
Noun-class agreement ("watu wawili", "mlango wa kwanza") requires context
and is out of scope for this rule layer; sentences relying on it should be
pre-verbalized by hand in the training/eval data.
"""

import re

UNITS = [
    "sifuri", "moja", "mbili", "tatu", "nne",
    "tano", "sita", "saba", "nane", "tisa",
]
TENS = {
    1: "kumi", 2: "ishirini", 3: "thelathini", 4: "arobaini", 5: "hamsini",
    6: "sitini", 7: "sabini", 8: "themanini", 9: "tisini",
}

ABBREVIATIONS = {
    "Dkt.": "Daktari",
    "Bw.": "Bwana",
    "Bi.": "Bibi",
    "Prof.": "Profesa",
    "n.k.": "na kadhalika",
    "k.m.": "kwa mfano",
    "k.v.": "kama vile",
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


def _verbalize_match(match: re.Match) -> str:
    token = match.group(0).replace(",", "")
    if "." in token:
        whole, frac = token.split(".", 1)
        frac_words = " ".join(UNITS[int(d)] for d in frac if d.isdigit())
        return f"{cardinal(int(whole))} nukta {frac_words}"
    return cardinal(int(token))


def expand_numbers(text: str) -> str:
    """Replace digit sequences (incl. 3,500-style separators and decimals) with words."""
    return re.sub(r"\d[\d,]*(?:\.\d+)?", _verbalize_match, text)


def expand_abbreviations(text: str) -> str:
    for abbr, full in ABBREVIATIONS.items():
        text = text.replace(abbr, full)
    return text


def normalize_for_tts(text: str) -> str:
    """Normalization applied to training/eval text fed to a TTS model."""
    text = expand_abbreviations(text)
    text = expand_numbers(text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_asr(text: str) -> str:
    """Normalization applied to both reference and hypothesis before WER/CER."""
    text = normalize_for_tts(text).lower()
    # Keep the ng' apostrophe (a real Swahili grapheme), drop other punctuation.
    text = re.sub(r"(?<![gG])'", " ", text)
    text = re.sub(r"[^\w' ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
