"""Render the SILMA MSA benchmark WER comparison as a clean dark bar chart (PNG).

Reads data/manifests/silma_msa_leaderboard.csv. Output: assets/benchmark_msa.png
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 620
OUT = Path("assets/benchmark_msa.png")
DJ_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
DJ = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BG = (13, 17, 23)
FG = (230, 237, 243)
MUTED = (139, 148, 158)
EMERALD = (52, 211, 153)
SLATE = (71, 85, 105)


def main() -> int:
    df = pd.read_csv("data/manifests/silma_msa_leaderboard.csv", index_col=0)
    df = df.sort_values("WER%")  # best first
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 7], fill=EMERALD)

    d.text((50, 40), "SILMA Open-Source Arabic TTS Benchmark (MSA)",
           font=ImageFont.truetype(DJ_B, 32), fill=FG)
    d.text((50, 84), "Word Error Rate via Whisper-large-v3 — lower is better  ·  10 sentences",
           font=ImageFont.truetype(DJ, 20), fill=MUTED)

    x0, x1 = 360, 850
    y = 150
    row_h = 66
    vmax = df["WER%"].max() * 1.12
    name_f = ImageFont.truetype(DJ_B, 24)
    val_f = ImageFont.truetype(DJ_B, 24)
    for _, r in df.iterrows():
        name = str(r["model"])
        ours = "fasih" in name.lower()
        col = EMERALD if ours else SLATE
        d.text((50, y + 8), name, font=name_f, fill=(FG if ours else MUTED))
        bw = int((x1 - x0) * (r["WER%"] / vmax))
        d.rounded_rectangle([x0, y, x0 + max(bw, 4), y + 40], radius=8, fill=col)
        d.text((x0 + max(bw, 4) + 14, y + 8), f'{r["WER%"]}%  WER',
               font=val_f, fill=(EMERALD if ours else FG))
        if ours:
            d.text((50, y + 36), "best", font=ImageFont.truetype(DJ, 16), fill=EMERALD)
        y += row_h

    d.text((50, H - 46),
           "Objective intelligibility metric. SILMA's own benchmark is a human auditory comparison.",
           font=ImageFont.truetype(DJ, 17), fill=MUTED)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, "PNG")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
