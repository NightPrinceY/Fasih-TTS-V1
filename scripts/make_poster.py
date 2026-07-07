"""Generate the Fasih-TTS-V1 marketing poster (PNG) with PIL + proper Arabic shaping.

Output: assets/poster.png  (portrait, 1080x1440)
"""

from __future__ import annotations

from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1440
OUT = Path("assets/poster.png")

DEJAVU_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
AR_BLACK = "/usr/share/fonts/truetype/noto/NotoSansArabic-Black.ttf"
AR_BOLD = "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf"

FG = (241, 245, 249)
MUTED = (148, 163, 184)
EMERALD = (52, 211, 153)
VIOLET = (167, 139, 250)
GOLD = (251, 191, 36)


def ar(t: str) -> str:
    return get_display(arabic_reshaper.reshape(t))


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def vgradient(top, bot):
    base = Image.new("RGB", (1, H))
    for y in range(H):
        f = y / (H - 1)
        base.putpixel((0, y), tuple(int(top[i] + (bot[i] - top[i]) * f) for i in range(3)))
    return base.resize((W, H))


def center(draw, y, text, fnt, fill):
    w = draw.textlength(text, font=fnt)
    draw.text(((W - w) / 2, y), text, font=fnt, fill=fill)
    return w


def main() -> int:
    img = vgradient((16, 30, 58), (8, 40, 45))
    d = ImageDraw.Draw(img)

    # subtle top glow bar
    d.rectangle([0, 0, W, 8], fill=EMERALD)

    # kicker
    kick = font(DEJAVU_B, 24)
    txt = "A R A B I C   ·   M S A / F U S H A   ·   T T S"
    center(d, 92, txt, kick, MUTED)

    # Arabic hero title
    center(d, 138, ar("فَصِيح"), font(AR_BLACK, 152), FG)

    # latin subtitle
    center(d, 402, "Fasih-TTS-V1", font(DEJAVU_B, 74), EMERALD)

    # tagline
    tl = font(DEJAVU, 29)
    center(d, 500, "A professional male voice for Modern Standard Arabic", tl, FG)
    center(d, 540, "broadcast-grade  ·  real-time  ·  human-level", tl, MUTED)

    # soundwave hero
    import math
    bars, bw, gap = 47, 10, 8
    total = bars * bw + (bars - 1) * gap
    x0 = (W - total) / 2
    cy = 682
    for i in range(bars):
        h = 12 + int(90 * abs(math.sin(i * 0.5)) * (0.5 + 0.5 * math.sin(i * 0.18)))
        t = i / (bars - 1)
        col = tuple(int(EMERALD[k] + (VIOLET[k] - EMERALD[k]) * t) for k in range(3))
        x = x0 + i * (bw + gap)
        d.rounded_rectangle([x, cy - h, x + bw, cy + h], radius=5, fill=col)

    # stat cards 2x2
    stats = [("1.3%", "CER — human-level"), ("×4", "identical / zero variance"),
             ("~0.60", "real-time factor"), ("675 ms", "streaming first-audio")]
    cw, ch, gx, gy = 470, 150, 40, 34
    sx = (W - (2 * cw + gx)) / 2
    sy = 806
    big = font(DEJAVU_B, 66)
    lab = font(DEJAVU, 27)
    for i, (v, l) in enumerate(stats):
        cx = sx + (i % 2) * (cw + gx)
        cyy = sy + (i // 2) * (ch + gy)
        d.rounded_rectangle([cx, cyy, cx + cw, cyy + ch], radius=20, fill=(21, 34, 57),
                            outline=(45, 60, 85), width=2)
        d.rectangle([cx, cyy + 20, cx + 6, cyy + ch - 20], fill=EMERALD)
        d.text((cx + 34, cyy + 30), v, font=big, fill=GOLD)
        d.text((cx + 36, cyy + 104), l, font=lab, fill=MUTED)

    # features
    feats = "✓ Diacritization (CATT)    ✓ Numbers    ✓ Sacred lexicon    ✓ Streaming API"
    center(d, 1188, feats, font(DEJAVU, 24), FG)

    # divider
    d.line([90, 1252, W - 90, 1252], fill=(45, 60, 85), width=2)

    # footer
    ft = font(DEJAVU_B, 26)
    center(d, 1284, "huggingface.co/NightPrince/Fasih-TTS-V1", ft, EMERALD)
    center(d, 1324, "github.com/NightPrinceY/Fasih-TTS-V1", ft, VIOLET)
    center(d, 1376, "Fine-tuned from Coqui XTTS v2   ·   © 2026 NightPrince", font(DEJAVU, 22), MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, "PNG")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
