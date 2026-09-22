"""Inverse text normalization (ITN) for Swahili / English ASR output.

Turns spoken numbers and spelled letters back into figures and codes:

    >>> inverse_normalize("piga simu sifuri saba moja mbili, tatu nne tano, sita saba nane sasa")
    'piga simu 0712345678 sasa'
    >>> inverse_normalize("tiketi yako ni kei kyuu, nne bii saba eksi mbili")
    'tiketi yako ni KQ4B7X2'
    >>> inverse_normalize("bei ni shilingi elfu tatu mia tano kwa kilo")
    'bei ni shilingi 3500 kwa kilo'

Rules (mirror of verbalizer.py):
  * a run of digit words with no tens/scale words is a spelled string (kept in
    order, leading zeros preserved); a run with tens/scale words is a quantity;
  * letter names (both languages, plus bare capitals) become letters when they
    form a run of two or more, or sit next to a digit group; common English
    words that double as letter names ("you", "see", "oh") need a run of three
    or an unambiguous neighbour;
  * digit groups separated by pause commas merge into one string (phone
    numbers); letters and digits merge into one code.

Whisper often writes numerals and capitals itself; those pass straight through,
so the module is idempotent on already-normalized text.
"""

from __future__ import annotations

import re

import english_text as en
import swahili_text as sw

UNIT_WORDS = {w: i for i, w in enumerate(sw.UNITS)}
UNIT_WORDS.update({w: i for i, w in enumerate(en.DIGITS)})
TEEN_WORDS = {w: 10 + i for i, w in enumerate(en.TEENS)}
TENS_WORDS = {w: t * 10 for t, w in sw.TENS.items()}
TENS_WORDS.update({w: t * 10 for t, w in en.TENS.items()})
TENS_WORDS["ten"] = 10
SW_SCALES = {"mia": 100, "elfu": 1_000, "milioni": 1_000_000, "bilioni": 1_000_000_000}
EN_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
CONJ = {"na", "and"}
NUMBER_VOCAB = set(UNIT_WORDS) | set(TEEN_WORDS) | set(TENS_WORDS) | set(SW_SCALES) | set(EN_SCALES)
EN_TENS_TEENS = set(en.TENS.values()) | set(en.TEENS) | {"ten"}

LETTER_WORDS: dict[str, str] = {}
MULTI_LETTER_WORDS: dict[tuple[str, ...], str] = {}
for _table in (sw.LETTERS, sw.LETTERS_NATIVE, en.LETTERS):
    for _letter, _name in _table.items():
        parts = tuple(_name.split())
        if len(parts) == 1:
            LETTER_WORDS.setdefault(parts[0], _letter)
        else:
            MULTI_LETTER_WORDS[parts] = _letter
# Older / alternative spellings of letter names that ASR may still produce.
for _name, _letter in {"ay": "a", "cue": "q", "kyu": "q", "que": "q", "zee": "z", "ei": "a"}.items():
    LETTER_WORDS.setdefault(_name, _letter)
# Letter names that are also ordinary words: only accepted inside a clear run.
AMBIGUOUS_LETTER_WORDS = {
    "a", "e", "i", "o", "u", "you", "see", "oh", "why", "are", "eye", "we", "be", "bee", "ay",
    "el", "em", "ex", "he", "me", "ne", "le", "ye", "che", "ii", "aa", "oo", "queue", "cue",
}

# Spelling variants Whisper produces for Swahili number words.
ALIASES = {"ine": "nne", "inne": "nne", "tanu": "tano", "sifuli": "sifuri"}
# Words Whisper tends to glue onto a following number word ("saambili").
GLUE_PREFIXES = {"saa", "na", "ya", "kwa", "ni"}

TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[^\sA-Za-z0-9']")


def _split_compound(word: str) -> list[str] | None:
    """'miatano' -> ['mia', 'tano'], 'saambili' -> ['saa', 'mbili']; None if not a compound."""
    low = word.lower()
    if low in NUMBER_VOCAB or low in LETTER_WORDS or len(low) < 4:
        return None
    for i in range(2, len(low) - 1):
        head, tail = low[:i], low[i:]
        tail = ALIASES.get(tail, tail)
        if tail in NUMBER_VOCAB and (head in NUMBER_VOCAB or head in GLUE_PREFIXES):
            return [head, tail]
    return None


class Tok:
    __slots__ = ("text", "kind")

    def __init__(self, text: str, kind: str):
        self.text = text    # surface form
        self.kind = kind    # word | punct | num | digits | letters | code

    def __repr__(self):
        return f"{self.text}/{self.kind}"


# --------------------------------------------------------------------------- numbers

def _parse_number_words(words: list[str]) -> list[int]:
    """Parse a run of cardinal words (either grammar) into one or more integers."""
    nums: list[int] = []
    total = small = 0
    have = False
    cur_scale = 10 ** 12

    def flush():
        nonlocal total, small, have, cur_scale
        if have:
            nums.append(total + small)
        total = small = 0
        have = False
        cur_scale = 10 ** 12

    i = 0
    while i < len(words):
        w = words[i]
        if w in CONJ:
            i += 1
            continue
        if w in SW_SCALES:  # scale first, multiplier follows: elfu tatu = 3000
            scale = SW_SCALES[w]
            if have and scale >= cur_scale:
                flush()
            if small:  # "mia tano" after a dangling small: treat small as done
                total += small
                small = 0
            mult, j = 0, i + 1
            if j < len(words) and words[j] in TENS_WORDS:
                mult = TENS_WORDS[words[j]]
                j += 1
                if j + 1 < len(words) and words[j] in CONJ and words[j + 1] in UNIT_WORDS:
                    mult += UNIT_WORDS[words[j + 1]]
                    j += 2
            elif j < len(words) and words[j] in UNIT_WORDS:
                mult = UNIT_WORDS[words[j]]
                j += 1
            total += scale * (mult or 1)
            cur_scale = scale
            have = True
            i = j
            continue
        if w in EN_SCALES:  # multiplier first: three thousand
            scale = EN_SCALES[w]
            if scale == 100:
                small = (small or 1) * 100
            else:
                if have and scale >= cur_scale:
                    flush()
                total += (small or 1) * scale
                small = 0
                cur_scale = scale
            have = True
            i += 1
            continue
        if w in TEEN_WORDS or w in TENS_WORDS:
            v = TEEN_WORDS.get(w) or TENS_WORDS[w]
            if have and 10 <= small < 100 and w in EN_TENS_TEENS:
                small = small * 100 + v          # year: nineteen sixty (four)
            elif have and small % 100:
                flush()
                small = v
            else:
                small += v
            have = True
            i += 1
            continue
        if w in UNIT_WORDS:
            v = UNIT_WORDS[w]
            if have and small % 10:
                flush()
            small += v
            have = True
            i += 1
            continue
        break
    flush()
    return nums


def _convert_number_run(run: list[Tok]) -> list[Tok]:
    """A run of number tokens (words, numerals, conjunctions) -> num/digits tokens."""
    out: list[Tok] = []
    words: list[str] = []

    def flush_words():
        if not words:
            return
        if any(w in TEEN_WORDS or w in TENS_WORDS or w in SW_SCALES or w in EN_SCALES for w in words):
            for n in _parse_number_words(words):
                out.append(Tok(str(n), "num"))
        else:
            # units only: a spelled digit string; a conjunction splits it ("mbili na moja")
            group: list[str] = []
            for w in words + [None]:
                if w in UNIT_WORDS:
                    group.append(str(UNIT_WORDS[w]))
                    continue
                if group:
                    out.append(Tok("".join(group), "digits" if len(group) >= 2 else "num"))
                    group = []
                if w is not None:
                    out.append(Tok(w, "word"))
        words.clear()

    # Whisper may already write "0712 345 678": several numerals in one run are
    # digit groups, not quantities.
    numeral_groups = sum(1 for t in run if t.text.isdigit())
    has_word_group = any(not t.text.isdigit() for t in run)
    for t in run:
        if t.text.isdigit():
            flush_words()
            grouped = len(t.text) >= 2 and (t.text[0] == "0" or numeral_groups >= 2 or has_word_group)
            out.append(Tok(t.text, "digits" if grouped else "num"))
        else:
            words.append(t.text.lower())
    flush_words()
    if numeral_groups and has_word_group:
        # "739 moja": the spoken digit continues the written group
        for t in out:
            if t.kind == "num" and t.text.isdigit():
                t.kind = "digits"
    return out


def _numbers_pass(toks: list[Tok]) -> list[Tok]:
    out: list[Tok] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        low = t.text.lower()
        if t.kind == "word" and (low in NUMBER_VOCAB or t.text.isdigit()):
            j = i
            while j < len(toks):
                lj = toks[j].text.lower()
                if toks[j].kind != "word":
                    break
                if lj in NUMBER_VOCAB or toks[j].text.isdigit():
                    j += 1
                elif lj in CONJ and j + 1 < len(toks) and toks[j + 1].kind == "word" \
                        and toks[j + 1].text.lower() in NUMBER_VOCAB and not toks[j - 1].text.isdigit():
                    j += 1
                else:
                    break
            out.extend(_convert_number_run(toks[i:j]))
            i = j
        else:
            out.append(t)
            i += 1
    return out


# --------------------------------------------------------------------------- letters

def _letter_of(toks: list[Tok], i: int) -> tuple[str | None, int, bool]:
    """(letter, tokens consumed, ambiguous) if toks[i] starts a letter name."""
    t = toks[i]
    if t.kind != "word":
        return None, 0, False
    if len(t.text) == 1 and t.text.isupper():
        return t.text, 1, t.text in ("I", "A")  # English words, not letters, on their own
    low = t.text.lower()
    for parts, letter in MULTI_LETTER_WORDS.items():
        if low == parts[0] and all(i + k < len(toks) and toks[i + k].text.lower() == p
                                   for k, p in enumerate(parts)):
            return letter.upper(), len(parts), False
    if low in LETTER_WORDS:
        return LETTER_WORDS[low].upper(), 1, low in AMBIGUOUS_LETTER_WORDS
    return None, 0, False


def _letters_pass(toks: list[Tok]) -> list[Tok]:
    out: list[Tok] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.kind == "word" and len(t.text) >= 2 and t.text.isupper() and t.text.isalpha():
            out.append(Tok(t.text, "letters"))
            i += 1
            continue
        if t.kind == "word" and len(t.text) == 1 and t.text.isupper() and i + 2 < len(toks)                 and toks[i + 1].text == "-" and len(toks[i + 2].text) == 1 and toks[i + 2].text.isupper():
            run, j = [t.text], i + 1
            while j + 1 < len(toks) and toks[j].text == "-" and len(toks[j + 1].text) == 1                     and toks[j + 1].text.isupper():
                run.append(toks[j + 1].text)
                j += 2
            out.append(Tok("".join(run), "letters"))
            i = j
            continue
        if t.kind == "word" and re.search(r"[A-Z]", t.text) and re.search(r"\d", t.text) \
                and t.text.upper() == t.text:
            out.append(Tok(t.text, "code"))
            i += 1
            continue
        letter, used, amb = _letter_of(toks, i)
        if letter is None:
            out.append(t)
            i += 1
            continue
        run, ambiguous, j = [], [], i
        while j < len(toks):
            letter, used, amb = _letter_of(toks, j)
            if letter is None:
                break
            run.append(letter)
            ambiguous.append(amb)
            j += used
        clear = len(run) - sum(ambiguous)
        prev_num = bool(out) and out[-1].kind in ("num", "digits", "code")
        next_num = j < len(toks) and toks[j].kind in ("num", "digits", "code")
        # "I see you" must not become ICU: a run needs at least one unambiguous
        # letter name, or, for a single letter, digits on both sides ("4 bee 7").
        accept = (len(run) >= 2 and (clear >= 1 or prev_num or next_num)) \
            or (len(run) == 1 and (clear == 1 and (prev_num or next_num) or (prev_num and next_num)))
        if accept:
            out.append(Tok("".join(run), "letters"))
            i = j
        else:
            out.append(t)
            i += 1
    return out


# --------------------------------------------------------------------------- merge

def _run_from(toks: list[Tok], k: int) -> list[Tok]:
    """Maximal separator-free alphanumeric run starting at toks[k]."""
    run = []
    while k < len(toks) and toks[k].kind in ("num", "digits", "letters", "code"):
        run.append(toks[k])
        k += 1
    return run


def _looks_like_code(run: list[Tok]) -> bool:
    kinds = {t.kind for t in run}
    return "code" in kinds or ("letters" in kinds and kinds & {"num", "digits"})


def _merge_pass(toks: list[Tok]) -> list[Tok]:
    out: list[Tok] = []
    i = 0
    alnum = ("num", "digits", "letters", "code")
    while i < len(toks):
        t = toks[i]
        if t.kind not in alnum:
            out.append(t)
            i += 1
            continue
        group = [t]
        j = i + 1
        while j < len(toks):
            nxt = toks[j]
            if nxt.kind in alnum:
                # adjacent with no separator: always merge, except two plain quantities
                if group[-1].kind == "num" and nxt.kind == "num":
                    break
                group.append(nxt)
                j += 1
                continue
            if nxt.kind == "punct" and nxt.text in ",-" and j + 1 < len(toks) and toks[j + 1].kind in alnum:
                # A pause comma separates phone-number groups ("0712, 345, 678") and the
                # parts of a code ("KQ, 4B7X2"); it also separates two unrelated items
                # ("OTP, 7391"), which must stay apart.
                left, right = group[-1], toks[j + 1]
                right_run = _run_from(toks, j + 1)
                if (left.kind == "digits" and right.kind == "digits") or _looks_like_code(group) \
                        or _looks_like_code(right_run):
                    group.append(right)
                    j += 2
                    continue
            break
        kinds = {g.kind for g in group}
        text = "".join(g.text for g in group)
        if len(group) == 1:
            out.append(group[0])
        elif kinds <= {"num", "digits"}:
            out.append(Tok(text, "digits"))
        else:
            out.append(Tok(text.upper(), "code"))
        i = j
    return out


# --------------------------------------------------------------------------- api

def _tokenize(text: str) -> list[Tok]:
    toks = []
    for m in TOKEN_RE.finditer(text):
        t = m.group(0)
        if not (t[0].isalnum() or t[0] == "'"):
            toks.append(Tok(t, "punct"))
            continue
        low = t.lower()
        if low in ALIASES:
            toks.append(Tok(ALIASES[low], "word"))
            continue
        parts = _split_compound(t)
        if parts:
            toks.extend(Tok(p, "word") for p in parts)
        else:
            toks.append(Tok(t, "word"))
    return toks


def _render(toks: list[Tok]) -> str:
    out = ""
    for t in toks:
        if t.kind == "punct" and t.text in ",.;:!?)":
            out += t.text
        elif out and not out.endswith("("):
            out += " " + t.text
        else:
            out += t.text
    return out.strip()


def inverse_normalize(text: str) -> str:
    toks = _tokenize(text)
    toks = _numbers_pass(toks)
    toks = _letters_pass(toks)
    toks = _merge_pass(toks)
    return _render(toks)


def extract_items(text: str, already_normalized: bool = False) -> list[str]:
    """Canonical alphanumeric items (digit strings, codes, acronyms) found in text."""
    norm = text if already_normalized else inverse_normalize(text)
    items = []
    for tok in re.findall(r"[A-Za-z0-9]+", norm):
        if re.search(r"\d", tok) and tok.upper() == tok:
            items.append(tok.upper())
        elif len(tok) >= 2 and tok.isalpha() and tok.isupper():
            items.append(tok)
    return items


if __name__ == "__main__":
    import sys
    for line in sys.stdin if len(sys.argv) == 1 else sys.argv[1:]:
        print(inverse_normalize(line.strip()))
