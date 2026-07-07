"""Arabic diacritization via vendored CATT (encoder-decoder), punctuation-preserving.

CATT strips punctuation before diacritizing. For TTS we must keep punctuation (it drives
prosody), so we diacritize the full sentence for context, then map the diacritized words
back onto the original token positions, leaving punctuation/spacing untouched.

Usage:
    from tts.text.diacritize import Diacritizer
    d = Diacritizer()                       # loads model on GPU if available
    d.diacritize_texts(["ما أجمل الصلاة"])  # -> ["مَا أَجْمَلُ الصَّلَاةِ"]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import torch

_CATT_DIR = Path(__file__).parent / "catt"
_DEFAULT_CKPT = Path("models/catt/best_ed_mlm_ns_epoch_178.pt")

# Characters that belong to an Arabic "word": letters + harakat + super/wasla alef + tatweel.
_WORD = r"ء-يً-ْٰٱـ"
_TOKEN_RE = re.compile(rf"[{_WORD}]+|[^{_WORD}]+")
_IS_WORD_RE = re.compile(rf"[{_WORD}]")


class Diacritizer:
    def __init__(self, ckpt: str | Path | None = None, device: str | None = None,
                 max_seq_len: int = 1024) -> None:
        if str(_CATT_DIR) not in sys.path:
            sys.path.insert(0, str(_CATT_DIR))
        from ed_pl import TashkeelModel  # noqa: E402  (vendored CATT)
        from tashkeel_tokenizer import TashkeelTokenizer  # noqa: E402
        from utils import remove_non_arabic  # noqa: E402

        self._clean = remove_non_arabic
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = TashkeelTokenizer()
        self.model = TashkeelModel(
            self.tokenizer, max_seq_len=max_seq_len, n_layers=3, learnable_pos_emb=False
        )
        ckpt = Path(ckpt) if ckpt else _DEFAULT_CKPT
        try:
            state = torch.load(ckpt, map_location=self.device, weights_only=True)
        except Exception:  # noqa: BLE001  (trusted, user-authorized checkpoint)
            state = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)

    def _reinsert(self, original: str, diac_sentence: str) -> str:
        """Put CATT's diacritized words back onto original token positions."""
        diac_words = diac_sentence.split()
        out, wi = [], 0
        for tok in _TOKEN_RE.findall(original):
            if _IS_WORD_RE.match(tok):
                if wi < len(diac_words):
                    out.append(diac_words[wi])
                    wi += 1
                else:
                    out.append(tok)  # ran out — keep original (safety)
            else:
                out.append(tok)  # punctuation / whitespace preserved verbatim
        # If word counts disagreed, alignment is unsafe -> signal caller to fall back.
        if wi != len(diac_words):
            return ""
        return "".join(out)

    def diacritize_texts(self, texts: list[str], batch_size: int = 16,
                         verbose: bool = False) -> list[str]:
        cleaned = [self._clean(t) for t in texts]
        diac = self.model.do_tashkeel_batch(cleaned, batch_size, verbose)
        results = []
        for orig, ds in zip(texts, diac):
            merged = self._reinsert(orig, ds)
            if not merged:  # fallback: diacritize per punctuation-delimited phrase
                merged = self._phrasewise(orig, batch_size)
            results.append(merged)
        return results

    def _phrasewise(self, text: str, batch_size: int) -> str:
        """Fallback: split on non-word separators, diacritize each Arabic phrase."""
        parts = _TOKEN_RE.findall(text)
        arabic_idx = [i for i, p in enumerate(parts) if _IS_WORD_RE.match(p)]
        phrases = [parts[i] for i in arabic_idx]
        diac = self.model.do_tashkeel_batch([self._clean(p) for p in phrases], batch_size, False)
        for i, d in zip(arabic_idx, diac):
            parts[i] = d if d.strip() else parts[i]
        return "".join(parts)
