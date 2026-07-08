"""Blog banner for Fasih-TTS — warm ivory, calligraphic Arabic hero, modern soft shapes.

Design system
  Ground   #F7F3EB warm ivory   ·   Ink #1F2421 warm near-black
  Accent   #0F766E deep teal (broadcast/tech; deliberately not the cream+terracotta cliché)
  Gold     #B78A2F muted, hairline/soft-shape only   ·   Grey #6B655B warm
  Type     Noto Naskh Arabic (calligraphic hero) + Poppins (wordmark/label)
  Layout   horizontal split — Latin info left, Arabic calligraphy feature right, soft blobs

Output: assets/blog_banner.png  (1600x600)
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1600, 600
OUT = Path("assets/blog_banner.png")
NASKH = "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf"
P_BOLD, P_SEMI, P_MED, P_REG = (f"fonts/Poppins-{s}.ttf" for s in ("Bold", "SemiBold", "Medium", "Regular"))

IVORY = (247, 243, 235)
IVORY2 = (238, 231, 217)
INK = (31, 36, 33)
TEAL = (15, 118, 110)
TEAL2 = (36, 150, 136)
GOLD = (183, 138, 47)
GREY = (107, 101, 91)


def naskh(sz):
    return ImageFont.truetype(NASKH, sz, layout_engine=ImageFont.Layout.RAQM)


def blob(cx, cy, r, color, alpha, blur):
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(lay).ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*color, alpha))
    return lay.filter(ImageFilter.GaussianBlur(blur))


def tracked(d, xy, text, font, fill, tracking):
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=font, fill=fill)
        x += d.textlength(ch, font=font) + tracking


def main() -> int:
    # warm ivory ground with a faint top-left warm lift
    base = Image.new("RGBA", (W, H), (*IVORY, 255))
    base = Image.alpha_composite(base, blob(360, 120, 520, IVORY2, 150, 200))
    # modern soft shapes on the right
    base = Image.alpha_composite(base, blob(1230, 210, 330, TEAL, 34, 150))
    base = Image.alpha_composite(base, blob(1360, 470, 260, GOLD, 30, 150))
    base = Image.alpha_composite(base, blob(1215, 300, 360, (255, 255, 255), 115, 190))
    img = base.convert("RGB")
    d = ImageDraw.Draw(img)

    # top accent hairline
    d.rectangle([0, 0, W, 5], fill=TEAL)

    # --- right: calligraphic Arabic hero ---
    d.text((1215, 300), "فصيح", font=naskh(200), fill=TEAL, anchor="mm")

    # --- left: wordmark + info ---
    tracked(d, (92, 108), "ARABIC · MSA · FUSHA · TEXT-TO-SPEECH", ImageFont.truetype(P_SEMI, 18), TEAL, 3)
    d.text((88, 142), "Fasih-TTS-V1", font=ImageFont.truetype(P_BOLD, 70), fill=INK)
    d.text((92, 246), "A professional male Arabic (Fusha) voice,", font=ImageFont.truetype(P_MED, 27), fill=(74, 70, 63))
    d.text((92, 284), "fine-tuned from Coqui XTTS v2.", font=ImageFont.truetype(P_MED, 27), fill=(74, 70, 63))

    # pill: benchmark claim (no emoji)
    pill = ImageFont.truetype(P_SEMI, 18)
    label = "#1 INTELLIGIBILITY   ·   SILMA OPEN ARABIC TTS BENCHMARK"
    pw = d.textlength(label, font=pill)
    d.rounded_rectangle([92, 352, 92 + pw + 44, 352 + 46], radius=23, fill=TEAL)
    d.text((92 + 22, 352 + 23), label, font=pill, fill=IVORY, anchor="lm")

    # elegant thin soundwave
    x0, cy = 94, 468
    for i in range(52):
        h = 5 + int(30 * abs(math.sin(i * 0.5)) * (0.55 + 0.45 * math.sin(i * 0.17)))
        t = i / 51
        col = tuple(int(a + (b - a) * t) for a, b in zip(TEAL, TEAL2))
        x = x0 + i * 11
        d.rounded_rectangle([x, cy - h, x + 4, cy + h], radius=2, fill=col)

    # bottom hairline + attribution
    d.line([92, 540, W - 92, 540], fill=(216, 208, 193), width=1)
    d.text((92, 556), "huggingface.co/NightPrince/Fasih-TTS-V1", font=ImageFont.truetype(P_SEMI, 17), fill=TEAL)
    d.text((W - 92, 556), "by Yahya Elnawasany (NightPrince)", font=ImageFont.truetype(P_REG, 17), fill=GREY, anchor="ra")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, "PNG")
    print(f"wrote {OUT} ({W}x{H})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
