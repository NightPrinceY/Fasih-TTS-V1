"""Shared visual identity for all Fasih-TTS graphics (banner, poster, diagram, chart).

Warm ivory ground · deep-teal accent · Poppins + Noto Naskh Arabic · soft blurred shapes.
Deliberately not the cream+serif+terracotta cliché.
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

_ROOT = Path(__file__).resolve().parents[1]
_FONTS = _ROOT / "fonts"
NASKH = "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf"

# ---- palette ----
IVORY = (247, 243, 235)
IVORY2 = (239, 232, 219)
CARD = (252, 250, 245)
INK = (31, 36, 33)
INK2 = (74, 70, 63)
TEAL = (15, 118, 110)
TEAL2 = (36, 150, 136)
GOLD = (183, 138, 47)
GREY = (122, 116, 106)
LINE = (223, 216, 202)
GOOD = (15, 118, 110)   # semantic (best)
WARN = (183, 138, 47)


def poppins(size: int, weight: str = "Bold"):
    return ImageFont.truetype(str(_FONTS / f"Poppins-{weight}.ttf"), size)


def naskh(size: int):
    return ImageFont.truetype(NASKH, size, layout_engine=ImageFont.Layout.RAQM)


def canvas(w: int, h: int, *, shapes: bool = True) -> Image.Image:
    """Warm ivory RGBA ground with optional soft decorative shapes + a teal top rule."""
    img = Image.new("RGBA", (w, h), (*IVORY, 255))
    img = Image.alpha_composite(img, _blob(w, h, int(w * 0.22), int(h * 0.2), int(h * 0.9), IVORY2, 150, 200))
    if shapes:
        img = Image.alpha_composite(img, _blob(w, h, int(w * 0.9), int(h * 0.15), int(h * 0.6), TEAL, 22, 170))
        img = Image.alpha_composite(img, _blob(w, h, int(w * 0.85), int(h * 0.95), int(h * 0.5), GOLD, 20, 170))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 5], fill=TEAL)
    return img


def _blob(w, h, cx, cy, r, color, alpha, blur):
    lay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(lay).ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*color, alpha))
    return lay.filter(ImageFilter.GaussianBlur(blur))


def card(img: Image.Image, box, *, radius: int = 18, fill=CARD, accent=None,
         shadow_alpha: int = 34, dy: int = 7, blur: int = 16):
    """Composite a soft drop shadow, then draw a rounded card. `accent` adds a left teal bar."""
    x0, y0, x1, y1 = box
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([x0, y0 + dy, x1, y1 + dy], radius, fill=(20, 30, 25, shadow_alpha))
    img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(blur)))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(box, radius, fill=fill, outline=LINE, width=1)
    if accent:
        d.rounded_rectangle([x0, y0 + 16, x0 + 6, y1 - 16], radius=3, fill=accent)
    return d


def tracked(d, xy, text, font, fill, tracking):
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=font, fill=fill)
        x += d.textlength(ch, font=font) + tracking


def save(img: Image.Image, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, "PNG")
    print(f"wrote {path} ({img.width}x{img.height})")
