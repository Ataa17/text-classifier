"""Text normalization for Tunisian Derja (Arabic script, Arabizi, French).

Designed to be configurable so we never *over*-normalize: script detection happens on the
raw text (see script_detect.py), and we keep emojis because they carry sentiment.
"""
from __future__ import annotations

import re
import unicodedata

# Arabic diacritics (harakat) and the tatweel/kashida elongation char.
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"

# Letter-variant unification (common for dialectal Arabic).
_LETTER_MAP = {
    "آ": "ا", "أ": "ا", "إ": "ا",  # آأإ -> ا
    "ى": "ي",  # ى -> ي
    "ة": "ه",  # ة -> ه
    "ؤ": "و",  # ؤ -> و
    "ئ": "ي",  # ئ -> ي
}
_LETTER_RE = re.compile("|".join(map(re.escape, _LETTER_MAP)))

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_WS = re.compile(r"\s+")


def strip_urls_mentions(text: str) -> str:
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    return text


def strip_diacritics(text: str) -> str:
    return _DIACRITICS.sub("", text)


def strip_tatweel(text: str) -> str:
    return text.replace(_TATWEEL, "")


def normalize_letters(text: str) -> str:
    return _LETTER_RE.sub(lambda m: _LETTER_MAP[m.group()], text)


def collapse_repeats(text: str, max_run: int = 2) -> str:
    """Cap any character repeated >max_run times (Arabic or Latin): barchaaaa -> barchaa."""
    return re.sub(r"(.)\1{" + str(max_run) + r",}", r"\1" * max_run, text)


def normalize(
    text: str,
    *,
    strip_diacritics_: bool = True,
    strip_tatweel_: bool = True,
    normalize_letters_: bool = True,
    collapse_repeats_: bool = True,
    max_repeat_run: int = 2,
) -> str:
    """Full normalization pipeline. Emojis are preserved."""
    text = unicodedata.normalize("NFC", text)
    text = strip_urls_mentions(text)
    if strip_tatweel_:
        text = strip_tatweel(text)
    if strip_diacritics_:
        text = strip_diacritics(text)
    if normalize_letters_:
        text = normalize_letters(text)
    if collapse_repeats_:
        text = collapse_repeats(text, max_repeat_run)
    return _WS.sub(" ", text).strip()
