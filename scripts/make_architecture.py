"""Fasih-TTS architecture diagram — warm ivory theme (see scripts/brand.py)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brand as B  # noqa: E402
from PIL import ImageDraw  # noqa: E402

W, H = 1320, 800
OUT = "assets/architecture.png"


def arrow(d, x1, y1, x2, y2, color):
    d.line([x1, y1, x2, y2], fill=color, width=3)
    a = 7
    d.polygon([(x2, y2), (x2 - a - 3, y2 - a), (x2 - a - 3, y2 + a)], fill=color)


def box(img, x, y, w, h, title, sub, hi=False):
    d = B.card(img, [x, y, x + w, y + h], radius=14,
               fill=(B.TEAL if hi else B.CARD), accent=(None if hi else B.TEAL))
    tc = B.IVORY if hi else B.INK
    sc = (222, 236, 231) if hi else B.GREY
    d.text((x + w / 2, y + h / 2 - 10), title, font=B.poppins(19, "SemiBold"), fill=tc, anchor="mm")
    if sub:
        d.text((x + w / 2, y + h / 2 + 14), sub, font=B.poppins(12, "Medium"), fill=sc, anchor="mm")
    return x + w / 2


def row(img, items, y, h=80):
    n = len(items)
    m, gap = 55, 26
    w = (W - 2 * m - (n - 1) * gap) / n
    d = ImageDraw.Draw(img)
    centers, x = [], m
    for title, sub, hi in items:
        cx = box(img, x, y, w, h, title, sub, hi)
        centers.append((cx, x, w))
        x += w + gap
    for i in range(n - 1):
        arrow(d, centers[i][1] + centers[i][2], y + h / 2, centers[i + 1][1], y + h / 2, (176, 168, 154))
    return centers


def main() -> int:
    img = B.canvas(W, H)
    d = ImageDraw.Draw(img)

    d.text((55, 40), "Fasih-TTS-V1", font=B.poppins(34, "Bold"), fill=B.INK)
    d.text((57, 88), "Arabic (MSA / Fusha) professional-male Text-to-Speech — system architecture",
           font=B.poppins(17, "Medium"), fill=B.GREY)

    B.tracked(d, (57, 140), "TRAINING PIPELINE", B.poppins(14, "SemiBold"), B.TEAL, 3)
    train = [("Dataset", "1517 clips · MP3", False), ("Validate", "QC · manifests", False),
             ("Diacritize", "CATT · tashkeel", False), ("Preprocess", "24 kHz · trim", False),
             ("Fine-tune", "XTTS v2 · FP32", False), ("Evaluate", "CER · WER", False)]
    tc = row(img, train, 165)

    B.tracked(d, (57, 405), "INFERENCE RUNTIME", B.poppins(14, "SemiBold"), B.TEAL, 3)
    infer = [("Raw Arabic text", "even undiacritized", False),
             ("Text Front-End", "normalize · diacritize · chunk", False),
             ("Fasih-TTS-V1", "XTTS v2 fine-tuned", True),
             ("24 kHz Speech", "mono waveform", False),
             ("Serving", "FastAPI · CLI", False)]
    ic = row(img, infer, 430, h=84)

    # connector: fine-tune -> model (dashed teal)
    fx, mx = tc[4][0], ic[2][0]
    for t in range(0, 100, 6):
        p0 = t / 100
        p1 = min((t + 3) / 100, 1)
        def bez(p):
            x = (1 - p) ** 2 * fx + 2 * (1 - p) * p * fx + p * p * mx
            y = (1 - p) ** 2 * (165 + 80) + 2 * (1 - p) * p * 340 + p * p * 430
            return x, y
        d.line([bez(p0), bez(p1)], fill=B.TEAL, width=3)
    d.text((mx + 12, 350), "produces the model", font=B.poppins(13, "Medium"), fill=B.TEAL)

    # benchmarks footer
    B.tracked(d, (57, 600), "BENCHMARKS", B.poppins(14, "SemiBold"), B.TEAL, 3)
    stats = [("1.3%", "CER ≈ human"), ("2.5%", "WER · NeMo (best)"),
             ("~0.60", "real-time factor"), ("~675 ms", "streaming first-audio")]
    cw, gap, y = 290, 25, 625
    x = 57
    for v, l in stats:
        dd = B.card(img, [x, y, x + cw, y + 92], radius=14, accent=B.TEAL)
        dd.text((x + 26, y + 20), v, font=B.poppins(34, "Bold"), fill=B.TEAL)
        dd.text((x + 28, y + 62), l, font=B.poppins(14, "Medium"), fill=B.GREY)
        x += cw + gap

    d.text((57, H - 34),
           "Base: Coqui XTTS v2  ·  Diacritization: CATT  ·  ASR judges: Whisper-large-v3 + NVIDIA NeMo",
           font=B.poppins(13, "Medium"), fill=B.GREY)

    B.save(img, OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
