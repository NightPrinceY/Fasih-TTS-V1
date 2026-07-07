"""Diacritization (tashkil) coverage analysis for Arabic text.

Used both to QC the training corpus (flag under-diacritized clips) and, later, to
gate inference text (auto-diacritize anything below threshold before synthesis).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

HARAKAT = set("ًٌٍَُِّْٰٕٓٔ")

# Arabic letters that can carry a haraka (consonants + hamza carriers). Long-vowel
# letters (ا و ي) frequently carry none, so we count "diacritizable" consonants only
# for a fair ratio.
_ARABIC_LETTER_LO, _ARABIC_LETTER_HI = "ؠ", "ي"


def _is_arabic_letter(c: str) -> bool:
    return _ARABIC_LETTER_LO <= c <= _ARABIC_LETTER_HI


@dataclass
class Coverage:
    letters: int
    marks: int
    ratio: float          # marks / diacritizable letters
    level: str            # "full" | "partial" | "none"


def coverage(text: str) -> Coverage:
    """Estimate how thoroughly a string is diacritized."""
    text = unicodedata.normalize("NFC", text)
    letters = sum(1 for c in text if _is_arabic_letter(c) and c not in HARAKAT)
    marks = sum(1 for c in text if c in HARAKAT)
    ratio = (marks / letters) if letters else 0.0
    if ratio > 0.50:
        level = "full"
    elif ratio > 0.05:
        level = "partial"
    else:
        level = "none"
    return Coverage(letters=letters, marks=marks, ratio=round(ratio, 3), level=level)


def needs_diacritization(text: str, threshold: float = 0.30) -> bool:
    """True if text is too sparsely diacritized to trust for Fusha synthesis."""
    return coverage(text).ratio < threshold
