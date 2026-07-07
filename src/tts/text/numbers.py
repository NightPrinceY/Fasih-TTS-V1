"""Expand digits to Arabic words (Dataset A had none, but production text will).

Uses num2words 'ar'. Converts Arabic-Indic digits too. Intentionally simple — integer
runs only; extend for decimals/currency/years if the agent needs them.
"""

from __future__ import annotations

import re

from num2words import num2words

_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_DIGITS = re.compile(r"\d+")


def expand_numbers(text: str) -> str:
    text = text.translate(_ARABIC_INDIC)

    def _repl(m: re.Match) -> str:
        try:
            return num2words(int(m.group()), lang="ar")
        except Exception:  # noqa: BLE001
            return m.group()

    return _DIGITS.sub(_repl, text)
