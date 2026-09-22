"""Language-aware verbalizer for Swahili / English / code-switched TTS input.

Finds digit- and capital-bearing spans, classifies them (semiotic class) and
rewrites each as words the model can pronounce, choosing Swahili or English word
forms per span from the surrounding words. Quantities stay cardinals; phone
numbers, identifiers and codes are read digit by digit with group pauses;
acronyms are read letter by letter.

Classes: phone, id, code, acronym, money, percent, time, date, ordinal, year,
decimal, number, spell (markup).

Markup (non-nesting), for callers who need to override the heuristics:
  [spell]...[/spell]  read every character (letter names + digits)
  [num]...[/num]      read digit strings as quantities (cardinals)
  [say]...[/say]      leave untouched (e.g. an all-caps word: [say]STOP[/say])
  [sw]...[/sw]        force Swahili word forms for this stretch
  [en]...[/en]        force English word forms for this stretch

    >>> verbalize("Piga simu 0712 345 678 sasa.")
    'Piga simu sifuri saba moja mbili, tatu nne tano, sita saba nane sasa.'
    >>> verbalize("Your ticket number is 48213.")
    'Your ticket number is four eight two one three.'
    >>> verbalize("Tiketi yako ni KQ-4B7X2.")
    'Tiketi yako ni kei kyuu, nne bii saba eksi mbili.'
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import english_text as en
import swahili_text as sw

LANG_MODULES = {"sw": sw, "en": en}

# All-caps tokens that are read as words, not letter by letter.
PRONOUNCED_ACRONYMS = {
    "NASA", "NATO", "NEMA", "KEMRI", "KEMSA", "SACCO", "UNICEF", "USAID", "UNESCO", "COVID",
    "FIFA", "CAF", "AIDS", "OPEC", "KEBS", "KEPSA", "NACADA", "TUKO", "UKIMWI", "KEFRI",
    "ICPAK", "KASNEB", "CHAN", "AFCON", "ECOWAS", "COMESA", "IGAD", "SADC", "UNEP", "WASCO",
}
# All-caps tokens of five or more letters that are nevertheless spelled out.
SPELLED_ACRONYMS = {"KUCCPS", "KNCHR", "KICTA", "NCPWD", "NHIF", "KPLC", "KNBS", "NCBA", "KCPE",
                    "KCSE", "NSSF", "IEBC", "KTDA", "NCIC", "KWFT"}

ID_KEYWORDS = sw.ID_KEYWORDS | en.ID_KEYWORDS
SCALE_WORDS = {"elfu": 1_000, "milioni": 1_000_000, "bilioni": 1_000_000_000,
               "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}

TAG_RE = re.compile(r"\[(spell|num|say|sw|en)\](.*?)\[/\1\]", re.S)
WORD_RE = re.compile(r"[A-Za-z']+")

CURRENCY_PREFIX = r"(?:KSh|KES|TSh|TZS|USh|UGX|US\$|USD|EUR|GBP|[$€£])"
CURRENCY_SUFFIX = r"(?:KES|TZS|UGX|USD|EUR|GBP|shillings?|bob)"

PATTERNS = [
    ("money", re.compile(rf"(?i)(?<![A-Za-z])({CURRENCY_PREFIX})\s?(\d(?:[\d,]*\d)?)(?:\.(\d{{1,2}}))?(?!\d|\.\d)")),
    ("money", re.compile(rf"(?i)(?<![\d.,])(\d(?:[\d,]*\d)?)(?:\.(\d{{1,2}}))?\s?({CURRENCY_SUFFIX})\b")),
    ("percent", re.compile(r"(?<![\d.,])(\d(?:[\d,]*\d)?)(?:\.(\d+))?\s?%")),
    ("time", re.compile(r"(?<![\d:])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])(?:\s?(am|pm|AM|PM)\b)?")),
    ("date", re.compile(r"(?<![\d.])(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})(?!\d|\.\d)")),
    ("phone", re.compile(r"(?<![\d.,+])\+?\(?\d{2,6}\)?(?:[ \-.]\(?\d{2,6}\)?){1,5}(?![\d])"
                         r"|(?<![\d.,+])\+?\d{7,15}(?![\d])")),
    ("code", re.compile(r"(?<![A-Za-z0-9])[A-Z0-9]+(?:[-/][A-Z0-9]+)*(?![A-Za-z0-9])")),
    ("acronym", re.compile(r"(?<![A-Za-z])(?:[A-Z]\.){2,}|(?<![A-Za-z0-9])[A-Z]{2,6}(?![A-Za-z0-9])")),
    ("ordinal", re.compile(r"(?<![\d.,])(\d+)(st|nd|rd|th)\b")),
    ("number", re.compile(r"(?<![\d.,])\d(?:[\d,]*\d)?(?:\.\d+)?")),
]

_ABBREV_RES = [
    (re.compile(r"(?<![A-Za-z])" + re.escape(k) + (r"(?=\s|$|\d)" if k.endswith(".") else r"(?![A-Za-z])")), v)
    for k, v in {**sw.ABBREVIATIONS, **en.ABBREVIATIONS}.items()
]


@dataclass
class Span:
    cls: str
    text: str          # original surface form
    canonical: str     # alphanumeric form used by the spelling eval (KQ4B7X2, 0712345678, 3500)
    lang: str
    words: str = ""    # verbalized form
    start: int = 0
    end: int = 0


# --------------------------------------------------------------------------- language

def _score_lang(words: list[str]) -> tuple[float, float]:
    score_sw = score_en = 0.0
    for w in words:
        wl = w.lower().strip("'")
        if wl in sw.STOPWORDS:
            score_sw += 2
        elif wl in en.STOPWORDS:
            score_en += 2
        elif len(wl) > 2 and wl[0].islower():
            # Swahili words end in a vowel; English words mostly do not.
            if wl[-1] in "aeiou":
                score_sw += 0.5
            else:
                score_en += 0.5
    return score_sw, score_en


def detect_lang(text: str, start: int = 0, end: int | None = None, default: str = "sw",
                window: int = 5) -> str:
    """Language of the words around text[start:end]; falls back to the whole sentence, then default."""
    end = len(text) if end is None else end
    before = WORD_RE.findall(text[:start])[-window:]
    after = WORD_RE.findall(text[end:])[:window]
    s, e = _score_lang(before + after)
    if s == e:
        s, e = _score_lang(WORD_RE.findall(text))
    if s == e:
        return default
    return "sw" if s > e else "en"


# --------------------------------------------------------------------------- helpers

def _chunk_digits(s: str) -> list[str]:
    n = len(s)
    layout = {7: [3, 4], 8: [4, 4], 9: [3, 3, 3], 10: [4, 3, 3], 11: [4, 4, 3], 12: [3, 3, 3, 3]}.get(n)
    if layout is None:
        if n <= 6:
            return [s]
        layout = [3] * (n // 3)
        if n % 3:
            layout[-1] += n % 3
    out, i = [], 0
    for size in layout:
        out.append(s[i:i + size])
        i += size
    return out


def _digit_groups(mod, groups: list[str]) -> str:
    return ", ".join(mod.digits(g) for g in groups)


def _number_words(mod, clean: str) -> str:
    if "." in clean:
        whole, frac = clean.split(".", 1)
        return f"{mod.cardinal(int(whole or 0))} {mod.DECIMAL_WORD} {mod.digits(frac)}"
    return mod.cardinal(int(clean))


def _spell(mod, s: str, letter_style: str | None) -> str:
    if mod is sw and letter_style == "sw_native":
        return sw.spell(s, sw.LETTERS_NATIVE)
    return mod.spell(s)


def _is_pronounced_word(part: str) -> bool:
    return part.isalpha() and (part in PRONOUNCED_ACRONYMS
                               or (len(part) >= 5 and part not in SPELLED_ACRONYMS))


def _code_words(mod, raw: str, letter_style: str | None) -> str:
    parts = re.split(r"[-/]", raw)
    has_word = any(_is_pronounced_word(p) for p in parts)
    out = []
    for p in parts:
        if not p:
            continue
        if _is_pronounced_word(p):
            out.append(p)
        elif p.isdigit() and has_word:
            out.append(mod.cardinal(int(p)))
        else:
            out.append(_spell(mod, p, letter_style))
    return ", ".join(out)


def _prev_words(text: str, start: int, n: int) -> list[str]:
    return [w.lower().strip(".'") for w in WORD_RE.findall(text[:start])[-n:]]


def _classify_number(text: str, m: re.Match, mod) -> str:
    raw = m.group(0)
    clean = raw.replace(",", "")
    if "." in clean:
        return "decimal"
    prev = _prev_words(text, m.start(), 5)
    weak = {"kwa", "to", "simu", "line"}
    strong_hit = any(w in ID_KEYWORDS and w not in weak for w in prev)
    weak_hit = bool(prev) and prev[-1] in weak and len(clean) >= 4
    if strong_hit or weak_hit:
        return "id"
    if "," not in raw and len(clean) >= 7:
        return "id"
    if "," not in raw and len(clean) >= 2 and clean[0] == "0":
        return "id"
    if "," not in raw and len(clean) == 4 and 1500 <= int(clean) <= 2099:
        return "year"
    return "number"


# --------------------------------------------------------------------------- core

def _find_spans(text: str, lang: str, letter_style: str | None) -> list[Span]:
    taken: list[tuple[int, int]] = []
    spans: list[Span] = []

    def free(a: int, b: int) -> bool:
        return all(b <= s or a >= e for s, e in taken)

    for pattern_cls, pattern in PATTERNS:
        for m in pattern.finditer(text):
            cls = pattern_cls
            a, b = m.span()
            if not free(a, b):
                continue
            raw = m.group(0)
            span_lang = lang if lang in LANG_MODULES else detect_lang(text, a, b)
            mod = LANG_MODULES[span_lang]
            words = canonical = None

            if cls == "phone":
                digits = re.sub(r"\D", "", raw)
                groups = re.findall(r"\d+", raw)
                if not 7 <= len(digits) <= 15:
                    continue
                if len(groups) == 2 and all(len(g) == 4 for g in groups) and "-" in raw:
                    continue  # 2024-2025: a range, not a phone number
                if len(groups) == 1:
                    groups = _chunk_digits(digits)
                words = _digit_groups(mod, groups)
                if raw.startswith("+"):
                    words = f"{mod.PLUS_WORD} {words}"
                canonical = digits
            elif cls == "code":
                if not (re.search(r"[A-Z]", raw) and re.search(r"\d", raw)):
                    continue
                words = _code_words(mod, raw, letter_style)
                canonical = "".join(p for p in re.split(r"[-/]", raw) if not _is_pronounced_word(p))
            elif cls == "acronym":
                letters = raw.replace(".", "")
                if letters in PRONOUNCED_ACRONYMS or (len(letters) >= 5 and letters not in SPELLED_ACRONYMS):
                    continue
                if letters in ("AM", "PM") and text[:a].rstrip()[-1:].isdigit():
                    continue
                words = _spell(mod, letters, letter_style)
                canonical = letters
            elif cls == "money":
                g = m.groups()
                code, whole, cents = (g[0], g[1], g[2]) if g[0] and not g[0][0].isdigit() else (g[2], g[0], g[1])
                whole_n = int(whole.replace(",", ""))
                cents_n = int(cents.ljust(2, "0")) if cents else None
                code_key = code.lower()
                if code_key not in mod.CURRENCIES:
                    code_key = "ksh"
                words = mod.money(code_key, whole_n, cents_n)
                canonical = str(whole_n)
            elif cls == "percent":
                num = m.group(1).replace(",", "") + (f".{m.group(2)}" if m.group(2) else "")
                words = mod.percent(_number_words(mod, num))
                canonical = num
            elif cls == "time":
                h, mi, suffix = int(m.group(1)), int(m.group(2)), m.group(3)
                words = mod.time_of_day(h, mi, suffix)
                if words.startswith("saa ") and _prev_words(text, a, 1) == ["saa"]:
                    words = words[4:]
                canonical = f"{h}{mi:02d}"
            elif cls == "date":
                d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if not (1 <= d <= 31 and 1 <= mo <= 12):
                    continue
                words = mod.date(d, mo, y)
                canonical = f"{d}{mo}{y}"
            elif cls == "ordinal":
                n = int(m.group(1))
                words = mod.ordinal(n)
                canonical = str(n)
            elif cls == "number":
                sub = _classify_number(text, m, mod)
                clean = raw.replace(",", "")
                if sub == "id":
                    words = _digit_groups(mod, _chunk_digits(clean))
                elif sub == "year":
                    words = mod.year(int(clean))
                else:
                    words = _number_words(mod, clean)
                cls = sub
                canonical = clean if sub in ("id", "decimal") else (clean.lstrip("0") or "0")
                if sub == "number":
                    prev = _prev_words(text, a, 1)
                    nxt = WORD_RE.findall(text[b:])[:1]
                    scale = SCALE_WORDS.get(prev[-1] if prev else "") or SCALE_WORDS.get(nxt[0].lower() if nxt else "")
                    if scale:
                        canonical = str(int(clean) * scale)
            taken.append((a, b))
            spans.append(Span(cls, raw, canonical, span_lang, words, a, b))
    spans.sort(key=lambda s: s.start)
    return spans


def _expand_abbreviations(text: str) -> str:
    for pattern, full in _ABBREV_RES:
        text = pattern.sub(full, text)
    return text


def _process(text: str, lang: str, letter_style: str | None) -> tuple[str, list[Span]]:
    """Verbalize one markup-free segment. Returns (words, spans)."""
    text = _expand_abbreviations(text)
    spans = _find_spans(text, lang, letter_style)
    out, pos = [], 0
    for s in spans:
        gap = text[pos:s.start]
        if out and gap.strip() == "" and out[-1] and not out[-1].endswith(","):
            gap = ", "
        out.append(gap)
        out.append(s.words)
        pos = s.end
    out.append(text[pos:])
    return "".join(out), spans


def _process_markup(text: str, lang: str, letter_style: str | None) -> tuple[str, list[Span]]:
    out, spans, pos = [], [], 0
    for m in TAG_RE.finditer(text):
        seg, seg_spans = _process(text[pos:m.start()], lang, letter_style)
        out.append(seg)
        spans.extend(seg_spans)
        tag, body = m.group(1), m.group(2)
        if tag in ("sw", "en"):
            seg, seg_spans = _process(body, tag, letter_style)
            out.append(seg)
            spans.extend(seg_spans)
        elif tag == "say":
            out.append(body)
        else:
            span_lang = lang if lang in LANG_MODULES else detect_lang(text, m.start(), m.end())
            mod = LANG_MODULES[span_lang]
            if tag == "spell":
                groups = re.findall(r"[A-Za-z0-9]+", body)
                words = ", ".join(_spell(mod, g, letter_style) for g in groups)
                canonical = "".join(groups).upper()
            else:  # num
                words = re.sub(r"\d[\d,]*(?:\.\d+)?", lambda n: _number_words(mod, n.group(0).replace(",", "")), body)
                canonical = re.sub(r"[^\d.]", "", body)
            out.append(words)
            spans.append(Span(tag, body, canonical, span_lang, words))
        pos = m.end()
    seg, seg_spans = _process(text[pos:], lang, letter_style)
    out.append(seg)
    spans.extend(seg_spans)
    return "".join(out), spans


def verbalize(text: str, lang: str = "auto", letter_style: str | None = None) -> str:
    """Rewrite digits, codes and acronyms as words.

    lang: 'sw' | 'en' | 'auto' (per-span detection, Swahili fallback).
    letter_style: None (English letter names, transliterated for Swahili spans) or
    'sw_native' (a, be, che, ... for Swahili spans)."""
    words, _ = _process_markup(text, lang, letter_style)
    words = re.sub(r"\s+", " ", words)
    words = re.sub(r"\s+([,.;:!?])", r"\1", words)
    return words.strip()


def classify(text: str, lang: str = "auto") -> list[Span]:
    """Spans the verbalizer would rewrite, with class and canonical alphanumeric form."""
    _, spans = _process_markup(text, lang, None)
    return spans


def expected_items(text: str, lang: str = "auto") -> list[str]:
    """Canonical strings a listener should be able to write back from the audio
    (phone digits, codes, acronyms, quantities). Times and dates are excluded."""
    return [s.canonical for s in classify(text, lang)
            if s.cls not in ("time", "date", "ordinal") and s.canonical]


if __name__ == "__main__":
    import sys
    for line in sys.stdin if len(sys.argv) == 1 else sys.argv[1:]:
        print(verbalize(line.strip()))
