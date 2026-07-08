"""Fasih-TTS marketing poster — warm ivory theme, calligraphic hero (see scripts/brand.py)."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brand as B  # noqa: E402
from PIL import ImageDraw  # noqa: E402

W, H = 1600, 920
OUT = "assets/poster.png"


def center(d, y, text, font, fill):
    d.text((W / 2, y), text, font=font, fill=fill, anchor="ma")


def main() -> int:
    img = B.canvas(W, H)
    d = ImageDraw.Draw(img)

    # eyebrow + calligraphic hero + wordmark
    lbl = B.poppins(20, "SemiBold")
    e = "ARABIC   ·   MSA / FUSHA   ·   TEXT-TO-SPEECH"
    B.tracked(d, ((W - d.textlength(e, font=lbl) - 5 * 4) / 2, 66), e, lbl, B.TEAL, 4)
    d.text((W / 2, 190), "فصيح", font=B.naskh(150), fill=B.TEAL, anchor="mm")
    center(d, 300, "Fasih-TTS-V1", B.poppins(76, "Bold"), B.INK)
    center(d, 404, "A professional male voice for Modern Standard Arabic — broadcast-grade, real-time",
           B.poppins(26, "Medium"), B.INK2)

    # soundwave
    bars, bw, gap = 66, 8, 9
    x0 = (W - (bars * bw + (bars - 1) * gap)) / 2
    cy = 505
    for i in range(bars):
        h = 8 + int(52 * abs(math.sin(i * 0.42)) * (0.5 + 0.5 * math.sin(i * 0.14)))
        t = i / (bars - 1)
        col = tuple(int(a + (b - a) * t) for a, b in zip(B.TEAL, B.TEAL2))
        x = x0 + i * (bw + gap)
        d.rounded_rectangle([x, cy - h, x + bw, cy + h], radius=4, fill=col)

    # stat cards
    stats = [("1.3%", "CER ≈ human"), ("#1", "SILMA intelligibility"),
             ("~0.60", "real-time factor"), ("675 ms", "streaming first-audio")]
    cw, ch, gx = 350, 138, 30
    sx = (W - (4 * cw + 3 * gx)) / 2
    sy = 578
    for i, (v, l) in enumerate(stats):
        x = sx + i * (cw + gx)
        dd = B.card(img, [x, sy, x + cw, sy + ch], radius=16, accent=B.TEAL)
        dd.text((x + 30, sy + 26), v, font=B.poppins(56, "Bold"), fill=B.TEAL)
        dd.text((x + 32, sy + 96), l, font=B.poppins(20, "Medium"), fill=B.GREY)

    center(d, 758, "Fully diacritized (CATT)      Number expansion      Sacred-term lexicon      Streaming API",
           B.poppins(21, "Medium"), B.INK2)

    d.line([120, 812, W - 120, 812], fill=B.LINE, width=1)
    center(d, 828, "huggingface.co/NightPrince/Fasih-TTS-V1      ·      github.com/NightPrinceY/Fasih-TTS-V1",
           B.poppins(21, "SemiBold"), B.TEAL)
    center(d, 862, "Portfolio:  nightprincey.github.io/Portfolio-App", B.poppins(20, "SemiBold"), B.GOLD)
    center(d, 894, "by Yahya Elnawasany (NightPrince)   ·   Fine-tuned from Coqui XTTS v2",
           B.poppins(18, "Regular"), B.GREY)

    B.save(img, OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
