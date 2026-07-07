"""Arabic (MSA/Fusha) text normalization for TTS.

Design principle: this is a *diacritization-preserving* normalizer. Fusha pronunciation
depends on harakat and on hamza forms, so unlike search/IR normalization we DO NOT:
  * strip tashkil (harakat, shadda, sukun, superscript alef),
  * collapse hamza/alef variants (أ إ آ ء ئ ؤ) — they change pronunciation.

We only apply changes that are safe for speech:
  * Unicode NFC canonicalization,
  * remove tatweel/kashida (ـ) — purely decorative, no phonetic value,
  * remove zero-width / bidi control characters,
  * collapse whitespace,
  * normalize a few punctuation variants used for prosody.
"""

from __future__ import annotations

import re
import unicodedata

# Tatweel / kashida — decorative elongation, no sound.
TATWEEL = "ـ"

# Zero-width and bidirectional control characters that pollute scraped text.
_INVISIBLES = dict.fromkeys(
    map(ord, ["​", "‌", "‍", "‎", "‏", "‪",
              "‫", "‬", "‭", "‮", "﻿"]),
    None,
)

# Arabic tashkil block we must preserve (documented, not stripped).
HARAKAT = "ًٌٍَُِّْٰٕٓٔ"

_WS = re.compile(r"\s+")
# Multiple ./!/؟ etc. collapsed to one; keep sentence-final punctuation for prosody.
_MULTIDOT = re.compile(r"\.{2,}")


def normalize(text: str) -> str:
    """Return speech-safe normalized text with diacritics preserved."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_INVISIBLES)
    text = text.replace(TATWEEL, "")
    # Common Latin punctuation that appears in Arabic text -> Arabic equivalents
    # is intentionally NOT forced here; we keep the author's punctuation but tidy runs.
    text = _MULTIDOT.sub("…", text)  # "..." -> ellipsis
    text = _WS.sub(" ", text).strip()
    return text


def strip_diacritics(text: str) -> str:
    """Remove all tashkil. Use ONLY for analysis/metrics, never before synthesis."""
    return "".join(c for c in unicodedata.normalize("NFC", text) if c not in HARAKAT)
