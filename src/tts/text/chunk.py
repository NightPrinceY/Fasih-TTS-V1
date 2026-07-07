"""Split long Arabic text into <=max_chars chunks for XTTS (char limit 166 for 'ar').

Splits preferentially at sentence boundaries (. ؟ ! ؛), then clause boundaries (،),
then hard-wraps on word boundaries. Diacritics count as characters, so Fusha hits the
limit sooner — this keeps every chunk safely under it for artifact-free synthesis.
"""

from __future__ import annotations

import re

_SENT = re.compile(r"(?<=[.؟!؛])\s+")
_CLAUSE = re.compile(r"(?<=،)\s*")


def _hardwrap(s: str, max_chars: int) -> list[str]:
    words, out, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur = f"{cur} {w}".strip()
        else:
            if cur:
                out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out


def chunk_text(text: str, max_chars: int = 160) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    cur = ""
    for sent in _SENT.split(text):
        if len(sent) <= max_chars:
            if len(cur) + len(sent) + 1 <= max_chars:
                cur = f"{cur} {sent}".strip()
            else:
                if cur:
                    chunks.append(cur)
                cur = sent
        else:  # sentence itself too long -> split on clauses, then hard-wrap
            if cur:
                chunks.append(cur)
                cur = ""
            buf = ""
            for cl in _CLAUSE.split(sent):
                if len(buf) + len(cl) + 1 <= max_chars:
                    buf = f"{buf} {cl}".strip()
                elif len(cl) > max_chars:
                    if buf:
                        chunks.append(buf)
                        buf = ""
                    chunks.extend(_hardwrap(cl, max_chars))
                else:
                    if buf:
                        chunks.append(buf)
                    buf = cl
            if buf:
                chunks.append(buf)
    if cur:
        chunks.append(cur)
    return chunks
