"""Generate the Fasih-TTS-V1 marketing poster (PNG) — LANDSCAPE, with Arabic shaping.

Output: assets/poster.png  (landscape, 1600x920)
"""

from __future__ import annotations

import math
from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

W, H = 1600, 920
OUT = Path("assets/poster.png")

DEJAVU_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
AR_BLACK = "/usr/share/fonts/truetype/noto/NotoSansArabic-Black.ttf"

FG = (241, 245, 249)
MUTED = (148, 163, 184)
EMERALD = (52, 211, 153)
VIOLET = (167, 139, 250)
GOLD = (251, 191, 36)


def ar(t: str) -> str:
    return get_display(arabic_reshaper.reshape(t))


def font(p, s):
    return ImageFont.truetype(p, s)


def vgradient(top, bot):
    base = Image.new("RGB", (1, H))
    for y in range(H):
        f = y / (H - 1)
        base.putpixel((0, y), tuple(int(top[i] + (bot[i] - top[i]) * f) for i in range(3)))
    return base.resize((W, H))


def center(d, y, text, fnt, fill):
    w = d.textlength(text, font=fnt)
    d.text(((W - w) / 2, y), text, font=fnt, fill=fill)


def main() -> int:
    img = vgradient((16, 30, 58), (8, 40, 45))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 8], fill=EMERALD)

    center(d, 46, "A R A B I C   ·   M S A / F U S H A   ·   T E X T - T O - S P E E C H",
           font(DEJAVU_B, 24), MUTED)

    # hero: Arabic (bare, unambiguous) + latin
    center(d, 66, ar("فصيح"), font(AR_BLACK, 118), FG)
    center(d, 288, "Fasih-TTS-V1", font(DEJAVU_B, 80), EMERALD)
    center(d, 390, "A professional male voice for Modern Standard Arabic  ·  broadcast-grade  ·  real-time",
           font(DEJAVU, 27), FG)

    # soundwave full width
    bars, bw, gap = 70, 9, 8
    total = bars * bw + (bars - 1) * gap
    x0 = (W - total) / 2
    cy = 508
    for i in range(bars):
        h = 9 + int(58 * abs(math.sin(i * 0.42)) * (0.5 + 0.5 * math.sin(i * 0.14)))
        t = i / (bars - 1)
        col = tuple(int(EMERALD[k] + (VIOLET[k] - EMERALD[k]) * t) for k in range(3))
        x = x0 + i * (bw + gap)
        d.rounded_rectangle([x, cy - h, x + bw, cy + h], radius=4, fill=col)

    # 4 stat cards in a row
    stats = [("1.3%", "CER — human-level"), ("×4", "identical / zero variance"),
             ("~0.60", "real-time factor"), ("675 ms", "streaming first-audio")]
    cw, ch, gx = 350, 138, 30
    sx = (W - (4 * cw + 3 * gx)) / 2
    sy = 574
    big = font(DEJAVU_B, 58)
    lab = font(DEJAVU, 24)
    for i, (v, l) in enumerate(stats):
        cx = sx + i * (cw + gx)
        d.rounded_rectangle([cx, sy, cx + cw, sy + ch], radius=18, fill=(21, 34, 57),
                            outline=(45, 60, 85), width=2)
        d.rectangle([cx, sy + 18, cx + 6, sy + ch - 18], fill=EMERALD)
        d.text((cx + 30, sy + 26), v, font=big, fill=GOLD)
        d.text((cx + 32, sy + 92), l, font=lab, fill=MUTED)

    # features
    center(d, 748,
           "✓ Fully diacritized (CATT)      ✓ Number expansion      ✓ Sacred-term lexicon      ✓ Streaming API",
           font(DEJAVU, 24), FG)

    # divider + footer
    d.line([120, 800, W - 120, 800], fill=(45, 60, 85), width=2)
    center(d, 820, "huggingface.co/NightPrince/Fasih-TTS-V1      ·      github.com/NightPrinceY/Fasih-TTS-V1",
           font(DEJAVU_B, 25), EMERALD)
    center(d, 856, "Portfolio:  nightprincey.github.io/Portfolio-App", font(DEJAVU_B, 24), VIOLET)
    center(d, 890, "by Yahya Elnawasany (NightPrince)   ·   Fine-tuned from Coqui XTTS v2   ·   © 2026",
           font(DEJAVU, 21), MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, "PNG")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {W}x{H})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
