"""Blog banner for the Fasih-TTS article (wide landscape hero). Output: assets/blog_banner.png"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630
OUT = Path("assets/blog_banner.png")
DJ_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
DJ = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
AR = "/usr/share/fonts/truetype/noto/NotoSansArabic-Black.ttf"
FG = (236, 242, 248)
MUTED = (150, 162, 178)
EMERALD = (52, 211, 153)
VIOLET = (167, 139, 250)
GOLD = (251, 191, 36)


def af(size):
    return ImageFont.truetype(AR, size, layout_engine=ImageFont.Layout.RAQM)


def main() -> int:
    strip = Image.new("RGB", (1, H))
    for y in range(H):
        f = y / (H - 1)
        strip.putpixel((0, y), tuple(int(a + (b - a) * f) for a, b in zip((14, 26, 50), (8, 40, 44))))
    img = strip.resize((W, H))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 7], fill=EMERALD)

    d.text((60, 58), "ARABIC · MSA / FUSHA · TEXT-TO-SPEECH", font=ImageFont.truetype(DJ_B, 22), fill=MUTED)
    # Arabic hero + latin
    d.text((60, 90), "فصيح", font=af(94), fill=FG)
    d.text((60, 258), "Fasih-TTS-V1", font=ImageFont.truetype(DJ_B, 70), fill=EMERALD)
    d.text((62, 352), "A professional male Arabic (Fusha) voice — fine-tuned from XTTS v2",
           font=ImageFont.truetype(DJ, 26), fill=FG)
    d.text((62, 390), "#1 intelligibility on the SILMA open-source Arabic TTS benchmark",
           font=ImageFont.truetype(DJ, 24), fill=GOLD)

    # soundwave accent (right/bottom)
    bars, bw, gap = 46, 8, 8
    x0 = 62
    cy = 480
    for i in range(bars):
        h = 8 + int(46 * abs(math.sin(i * 0.5)) * (0.5 + 0.5 * math.sin(i * 0.16)))
        t = i / (bars - 1)
        col = tuple(int(a + (b - a) * t) for a, b in zip(EMERALD, VIOLET))
        x = x0 + i * (bw + gap)
        d.rounded_rectangle([x, cy - h, x + bw, cy + h], radius=4, fill=col)

    d.text((60, H - 52), "huggingface.co/NightPrince/Fasih-TTS-V1   ·   by Yahya Elnawasany (NightPrince)",
           font=ImageFont.truetype(DJ, 20), fill=MUTED)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, "PNG")
    print(f"wrote {OUT} ({W}x{H})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
