"""Detect the dominant script of a comment, kept as a feature (not used to transliterate).

Labels: "arabic"  (Arabic script, i.e. Derja written in Arabic letters)
        "latin"   (Arabizi and/or French — Latin letters)
        "mixed"   (meaningful amounts of both)
        "other"   (mostly emoji/punctuation/digits — no alphabetic content)

NOTE (per project decision): script is assigned per *comment* using the dominant alphabetic
script. A finer per-token split can be added later if the by-script error analysis needs it.
"""
from __future__ import annotations

import re

_ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")
_LATIN = re.compile(r"[A-Za-zÀ-ÿ]")

MIXED_THRESHOLD = 0.2  # min share of the minority script to call it "mixed"


def script_counts(text: str) -> tuple[int, int]:
    return len(_ARABIC.findall(text)), len(_LATIN.findall(text))


def detect_script(text: str) -> str:
    ar, la = script_counts(text)
    total = ar + la
    if total == 0:
        return "other"
    minority = min(ar, la) / total
    if minority >= MIXED_THRESHOLD:
        return "mixed"
    return "arabic" if ar > la else "latin"
