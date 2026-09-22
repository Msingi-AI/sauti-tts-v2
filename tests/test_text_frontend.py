"""Verbalizer (text -> words) and ITN (words -> text) for Swahili / English / code-switch.

    python -m pytest tests/test_text_frontend.py -q
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import english_text as en  # noqa: E402
import swahili_text as sw  # noqa: E402
from itn import extract_items, inverse_normalize  # noqa: E402
from verbalizer import classify, detect_lang, expected_items, verbalize  # noqa: E402


# ----------------------------------------------------------------------------- tables

@pytest.mark.parametrize("n, words", [
    (0, "sifuri"), (15, "kumi na tano"), (135, "mia moja thelathini na tano"),
    (250, "mia mbili na hamsini"), (3500, "elfu tatu mia tano"),
    (45000, "elfu arobaini na tano"), (1964, "elfu moja mia tisa sitini na nne"),
    (6_000_000, "milioni sita"),
])
def test_sw_cardinal(n, words):
    assert sw.cardinal(n) == words


@pytest.mark.parametrize("n, words", [
    (0, "zero"), (15, "fifteen"), (64, "sixty four"), (105, "one hundred and five"),
    (1200, "one thousand two hundred"), (3500, "three thousand five hundred"),
    (50_000, "fifty thousand"), (1_000_001, "one million and one"),
])
def test_en_cardinal(n, words):
    assert en.cardinal(n) == words


@pytest.mark.parametrize("n, words", [
    (1964, "nineteen sixty four"), (2026, "twenty twenty six"), (2000, "two thousand"),
    (2005, "two thousand and five"), (1900, "nineteen hundred"),
])
def test_en_year(n, words):
    assert en.year(n) == words


def test_en_ordinal():
    assert [en.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 21, 30)] == [
        "first", "second", "third", "fourth", "eleventh", "twelfth", "twenty first", "thirtieth"]


# ----------------------------------------------------------------------------- verbalizer

@pytest.mark.parametrize("text, words", [
    # phone numbers: digit by digit, group pauses, leading zero kept
    ("Piga simu 0712 345 678 sasa.",
     "Piga simu sifuri saba moja mbili, tatu nne tano, sita saba nane sasa."),
    ("Please call 0722 555 019 to confirm.",
     "Please call zero seven two two, five five five, zero one nine to confirm."),
    ("Call +254 712 345 678.",
     "Call plus two five four, seven one two, three four five, six seven eight."),
    ("Akaunti yako ni 0110234567.",
     "Akaunti yako ni sifuri moja moja sifuri, mbili tatu nne, tano sita saba."),
    # identifiers after a keyword: digit by digit; leading zero: digit by digit
    ("Nambari ya tiketi ni 48213.", "Nambari ya tiketi ni nne nane mbili moja tatu."),
    ("Your ticket number is 48213.", "Your ticket number is four eight two one three."),
    ("Weka PIN yako 2468 kisha bonyeza sawa.",
     "Weka PIN yako mbili nne sita nane kisha bonyeza sawa."),
    ("Tuma neno sawa kwa 22384.", "Tuma neno sawa kwa mbili mbili tatu nane nne."),
    ("Nambari yako ya siri ni 5931.", "Nambari yako ya siri ni tano tisa tatu moja."),
    # codes: letters + digits, hyphen is a pause
    ("Tiketi yako ni KQ-4B7X2.", "Tiketi yako ni kei kyuu, nne bii saba eksi mbili."),
    ("Your booking reference is KQ-4B7X2.", "Your booking reference is kay queue, four bee seven ex two."),
    ("Flight KQ100 inaondoka saa 2.", "Flight kei kyuu moja sifuri sifuri inaondoka saa mbili."),
    ("Order yako AB123CD imefika.", "Order yako ei bii moja mbili tatu sii dii imefika."),
    # acronyms: spelled unless pronounced
    ("Kikao cha NHIF na KRA kitaanza saa 9.", "Kikao cha eni eichi ai efu na kei aa ei kitaanza saa tisa."),
    ("The KRA office is near NEMA.", "The kay ar ey office is near NEMA."),
    ("COVID-19 iliathiri watu wengi.", "COVID, kumi na tisa iliathiri watu wengi."),
    # pronounceable capitals are words (PIN, SIM, BEI); unpronounceable or listed ones are spelled
    ("Weka SIM kadi na piga ATM ya KCB.", "Weka SIM kadi na piga ei tii emu ya kei sii bii."),
    ("ID yako na USA na CEO wa NEMA.", "ai dii yako na yuu esi ei na sii ii oo wa NEMA."),
    ("Kikao cha ODM na UDA.", "Kikao cha oo dii emu na UDA."),
    ("Report to HR by 9 AM.", "Report to aitch ar by nine ey em."),
    # an all-caps sentence is shouting, not a string of acronyms
    ("BEI YA SUKARI NI SHILINGI 3500 KWA KILO.", "BEI YA SUKARI NI SHILINGI elfu tatu mia tano KWA KILO."),
    ("PIGA SIMU 0712 345 678 SASA.", "PIGA SIMU sifuri saba moja mbili, tatu nne tano, sita saba nane SASA."),
    # quantities stay cardinals
    ("Bei ya sukari ni shilingi 3500 kwa kilo.", "Bei ya sukari ni shilingi elfu tatu mia tano kwa kilo."),
    ("Timu yetu ilishinda kwa mabao 2 dhidi ya 1.", "Timu yetu ilishinda kwa mabao mbili dhidi ya moja."),
    ("Shule ina wanafunzi 1200 na walimu 48.", "Shule ina wanafunzi elfu moja mia mbili na walimu arobaini na nane."),
    ("The hospital will serve 1200 patients.", "The hospital will serve one thousand two hundred patients."),
    ("Simu iligharimu shilingi elfu 45.", "Simu iligharimu shilingi elfu arobaini na tano."),
    # years
    ("Mwaka 1964 Tanganyika na Zanzibar ziliungana.",
     "Mwaka elfu moja mia tisa sitini na nne Tanganyika na Zanzibar ziliungana."),
    ("In 1964 the country became independent.", "In nineteen sixty four the country became independent."),
    # money, percent, decimals, time, date, ordinal
    ("Bei ni KSh 3,500.50 au $20.", "Bei ni shilingi elfu tatu mia tano na senti hamsini au dola ishirini."),
    ("It costs 3,500 shillings.", "It costs three thousand five hundred shillings."),
    ("Asilimia 78 ya wakazi, yaani 78%.", "Asilimia sabini na nane ya wakazi, yaani asilimia sabini na nane."),
    ("Kilo 2.5 za sukari.", "Kilo mbili nukta tano za sukari."),
    ("Meeting at 8:30 on 15/07/2026, 2nd floor.",
     "Meeting at eight thirty on fifteenth of July twenty twenty six, second floor."),
    ("Tutaondoka saa 7:30 jioni.", "Tutaondoka saa saba na nusu jioni."),
    # abbreviations
    ("Dkt. Amina na Dr. Smith.", "Daktari Amina na Doctor Smith."),
    # markup
    ("Send [say]STOP[/say] to 22384.", "Send STOP to two two three eight four."),
    ("[spell]Nyumba24[/spell] ndio password.", "en why you em bee ey two four ndio password."),
    ("Tiketi [num]48213[/num].", "Tiketi elfu arobaini na nane mia mbili kumi na tatu."),
    ("[en]Ticket 48213[/en] tafadhali.", "Ticket four eight two one three tafadhali."),
    # adjacent spans get a pause so they stay separate items
    ("Enter the OTP 7391 to continue.", "Enter the oh tee pee, seven three nine one to continue."),
])
def test_verbalize(text, words):
    assert verbalize(text) == words


@pytest.mark.parametrize("token, spelled", [
    ("PIN", False), ("BEI", False), ("SIM", False), ("NEMA", False), ("UDA", False), ("SAWA", False),
    ("STOP", False), ("KRA", True), ("NHIF", True), ("NSSF", True), ("KDA", True), ("OTP", True),
    ("ATM", True), ("KQ", True), ("ID", True), ("USA", True), ("CEO", True), ("IEBC", True),
    ("KCB", True), ("SMS", True), ("KENHA", False), ("UNICEF", False),
])
def test_spelled_rule(token, spelled):
    from verbalizer import _spelled
    assert _spelled(token) is spelled


def test_forced_language():
    assert verbalize("Tiketi 48213", lang="en") == "Tiketi four eight two one three"
    assert verbalize("Ticket 48213", lang="sw") == "Ticket nne nane mbili moja tatu"


def test_native_letter_style():
    assert verbalize("Kikao cha KRA.", letter_style="sw_native") == "Kikao cha ke re a."


def test_detect_lang():
    assert detect_lang("Nimetuma email kwa manager wako.") == "sw"
    assert detect_lang("Your ticket number is 48213.") == "en"
    t = "Ticket number yako ni 48213."
    assert detect_lang(t, t.index("48213"), t.index("48213") + 5) == "sw"


def test_verbalize_idempotent_on_words():
    text = verbalize("Piga simu 0712 345 678 na KRA.")
    assert verbalize(text) == text


def test_classify_classes():
    spans = classify("Piga simu 0712 345 678, tiketi KQ-4B7X2, KRA, shilingi 3500, saa 8:30, 15/07/2026.")
    assert [s.cls for s in spans] == ["phone", "code", "acronym", "number", "time", "date"]


def test_expected_items():
    assert expected_items("Piga simu 0712 345 678, tiketi KQ-4B7X2 ya KRA, shilingi 3500 saa 8:30.") == [
        "0712345678", "KQ4B7X2", "KRA", "3500"]
    assert expected_items("Simu iligharimu shilingi elfu 45.") == ["45000"]


# ----------------------------------------------------------------------------- ITN

@pytest.mark.parametrize("spoken, written", [
    ("piga simu sifuri saba moja mbili, tatu nne tano, sita saba nane sasa", "piga simu 0712345678 sasa"),
    ("tiketi yako ni kei kyuu, nne bii saba eksi mbili", "tiketi yako ni KQ4B7X2"),
    ("your booking reference is kay queue, four bee seven ex two", "your booking reference is KQ4B7X2"),
    ("your booking reference is kay cue, four bee seven ex two", "your booking reference is KQ4B7X2"),
    ("bei ni shilingi elfu tatu mia tano kwa kilo", "bei ni shilingi 3500 kwa kilo"),
    ("mwaka elfu moja mia tisa sitini na nne", "mwaka 1964"),
    ("in nineteen sixty four", "in 1964"),
    ("twenty twenty six", "2026"),
    ("one thousand two hundred and five patients", "1205 patients"),
    ("mabao mbili dhidi ya moja", "mabao 2 dhidi ya 1"),
    ("mabao mbili na moja", "mabao 2 na 1"),
    ("kikao cha eni eichi ai efu na kei aa ei", "kikao cha NHIF na KRA"),
    ("the kay ar ey office", "the KRA office"),
    ("the kay ar ay office", "the KRA office"),
    ("enter the oh tee pee, seven three nine one", "enter the OTP, 7391"),
    ("zaidi ya watu mia mbili na hamsini", "zaidi ya watu 250"),
    ("shilingi elfu arobaini na tano", "shilingi 45000"),
    # guards: ordinary words that double as letter names stay words
    ("I see you at the office at twelve today.", "I see you at the office at 12 today."),
    ("oh you are here", "oh you are here"),
    # Whisper output that is already written stays as it is
    ("your number is 0712 345 678", "your number is 0712345678"),
    ("Ticket number 48213 and KQ-4B7X2", "Ticket number 48213 and KQ4B7X2"),
    ("Kikao cha KRA na NHIF", "Kikao cha KRA na NHIF"),
    ("A B C then 2 and 1", "ABC then 2 and 1"),
])
def test_inverse_normalize(spoken, written):
    assert inverse_normalize(spoken) == written


def test_extract_items():
    assert extract_items("piga simu sifuri saba moja mbili, tatu nne tano, sita saba nane kwa kei aa ei") == [
        "0712345678", "KRA"]
    assert extract_items("Ticket 48213 and KQ4B7X2 at 8", already_normalized=True) == ["48213", "KQ4B7X2", "8"]


ROUND_TRIP = [
    "Piga simu 0712 345 678 ukihitaji msaada.",
    "Nambari ya tiketi yako ni 48213.",
    "Nambari yako ya usajili ni KQ-4B7X2.",
    "Wasiliana na KRA kupitia nambari 0800 722 222.",
    "Kikao cha NHIF na NSSF kitaanza kesho asubuhi.",
    "Weka PIN yako 2468 kisha bonyeza sawa.",
    "Ndege ya KQ100 itaondoka saa 4 asubuhi.",
    "Akaunti yako ya benki ni 0110234567.",
    "Bei ya sukari ni shilingi 3,500 kwa kilo.",
    "Please call 0722 555 019 to confirm your booking.",
    "Your ticket number is 48213.",
    "Your booking reference is KQ-4B7X2.",
    "Enter the OTP 7391 to continue.",
    "The KRA office is next to the NSSF building.",
    "Your order number is AB123CD.",
    "The total came to 3,500 shillings.",
    "Tafadhali call 0712 345 678 kwa msaada zaidi.",
    "Ticket number yako ni 48213.",
    "Flight KQ100 inaondoka saa 2 asubuhi.",
    "Nitatuma link kwa nambari 0733 900 111.",
    "Lipa na paybill 522533, account number ni 78120.",
    "Mwaka 1964 Tanganyika na Zanzibar ziliungana.",
    "In 1964 the country became independent.",
    "Shule ina wanafunzi 1200 na walimu 48.",
]


@pytest.mark.parametrize("text", ROUND_TRIP)
def test_round_trip(text):
    """verbalize -> ITN gives back every item the eval expects from the audio."""
    words = verbalize(text)
    assert not any(c.isdigit() for c in words), words
    got = extract_items(inverse_normalize(words), already_normalized=True)
    for item in expected_items(text):
        assert item in got, (text, words, got)
