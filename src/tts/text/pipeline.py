"""Composable Arabic text front-end for the TTS system.

Two entry points with different guarantees:

* ``prepare_training_text`` — transcripts are already diacritized (gold or CATT), so we
  only normalize (diacritics preserved). Fast, no model needed.
* ``prepare_inference_text`` — production text from users/agent may be bare, may contain
  numbers/abbreviations. We normalize -> diacritize-if-needed -> apply the sacred-term
  lexicon so pronunciation of religious vocabulary is never left to chance.

The diacritizer and lexicon are injected lazily so training code never loads the CATT model.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .chunk import chunk_text
from .diacritics import needs_diacritization
from .normalize import normalize
from .numbers import expand_numbers

_LEXICON_PATH = Path("configs/data/lexicon_ar.yaml")


class TextPipeline:
    def __init__(self, diacritizer=None, lexicon: dict[str, str] | None = None) -> None:
        self._diacritizer = diacritizer
        self._lexicon = lexicon if lexicon is not None else self._load_lexicon()

    @staticmethod
    def _load_lexicon() -> dict[str, str]:
        if _LEXICON_PATH.exists():
            data = yaml.safe_load(_LEXICON_PATH.read_text(encoding="utf-8")) or {}
            return dict(data.get("overrides", {}))
        return {}

    def prepare_training_text(self, text: str) -> str:
        """Transcript is already diacritized -> normalize only (preserves tashkil)."""
        return normalize(text)

    def prepare_inference_text(self, text: str, diac_threshold: float = 0.30) -> str:
        # numbers -> words BEFORE diacritization so the words get diacritized too
        t = expand_numbers(normalize(text))
        if needs_diacritization(t, diac_threshold):
            if self._diacritizer is None:
                raise RuntimeError(
                    "Inference text is under-diacritized but no diacritizer was provided. "
                    "Pass Diacritizer() to TextPipeline."
                )
            t = normalize(self._diacritizer.diacritize_texts([t])[0])
        return self._apply_lexicon(t)

    def prepare_chunks(self, text: str, max_chars: int = 160,
                       diac_threshold: float = 0.30) -> list[str]:
        """Full front-end -> list of <=max_chars diacritized chunks ready for XTTS."""
        return chunk_text(self.prepare_inference_text(text, diac_threshold), max_chars)

    def _apply_lexicon(self, text: str) -> str:
        """Force canonical diacritized spelling for critical (e.g. sacred) terms."""
        for surface, canonical in self._lexicon.items():
            text = text.replace(surface, canonical)
        return text
