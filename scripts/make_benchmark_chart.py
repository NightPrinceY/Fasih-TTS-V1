"""Render the SILMA MSA multi-metric comparison table (PNG).

Merges: silma_msa_scores.csv (Whisper WER/CER), silma_msa_nemo.csv (NeMo WER/CER),
silma_msa_utmos.csv (UTMOS). Output: assets/benchmark_msa.png
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

OUT = Path("assets/benchmark_msa.png")
DJ_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
DJ = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BG = (13, 17, 23)
FG = (230, 237, 243)
MUTED = (139, 148, 158)
EMERALD = (52, 211, 153)
ROW = (22, 27, 34)
HL = (16, 44, 40)


def main() -> int:
    w = pd.read_csv("assets/benchmark/silma_msa_scores.csv", index_col=0).rename(
        columns={"WER%": "wer_wh", "CER%": "cer_wh"})
    n = pd.read_csv("assets/benchmark/silma_msa_nemo.csv", index_col=0).rename(
        columns={"WER%": "wer_ne", "CER%": "cer_ne"})
    u = pd.read_csv("assets/benchmark/silma_msa_utmos.csv", index_col=0)
    df = w.merge(n, on="model").merge(u, on="model").sort_values("wer_ne").reset_index(drop=True)

    cols = [("Model", 300, "l"), ("WER · Whisper", 200, "c"), ("WER · NeMo", 200, "c"),
            ("UTMOS", 160, "c")]
    W = 40 + sum(c[1] for c in cols) + 130
    header_h, row_h = 150, 66
    H = header_h + len(df) * row_h + 70
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 7], fill=EMERALD)

    d.text((40, 34), "SILMA Open-Source Arabic TTS Benchmark (MSA)", font=ImageFont.truetype(DJ_B, 30), fill=FG)
    d.text((40, 76), "Two independent ASR judges + naturalness  ·  10 sentences  ·  lower WER / higher UTMOS = better",
           font=ImageFont.truetype(DJ, 18), fill=MUTED)

    hf = ImageFont.truetype(DJ_B, 20)
    x = 40
    xs = []
    for title, cw, _ in cols:
        xs.append((x, cw))
        d.text((x + (10 if title == "Model" else cw // 2 - d.textlength(title, font=hf) // 2), header_h - 34),
               title, font=hf, fill=MUTED)
        x += cw
    d.line([40, header_h - 2, W - 40, header_h - 2], fill=(48, 54, 61), width=2)

    cf = ImageFont.truetype(DJ_B, 22)
    cf2 = ImageFont.truetype(DJ, 21)
    best_ne = df["wer_ne"].min()
    for r, (_, row) in enumerate(df.iterrows()):
        y = header_h + r * row_h
        ours = "fasih" in str(row["model"]).lower()
        d.rounded_rectangle([40, y + 4, W - 40, y + row_h - 4], radius=8, fill=(HL if ours else ROW))
        vals = [
            (str(row["model"]) + ("  ★" if ours else ""), "l", EMERALD if ours else FG, cf if ours else cf2),
            (f'{row["wer_wh"]}%', "c", FG, cf2),
            (f'{row["wer_ne"]}%', "c", EMERALD if row["wer_ne"] == best_ne else FG, cf2),
            (f'{row["utmos"] if "utmos" in row else row["UTMOS"]:.2f}', "c", FG, cf2),
        ]
        for (xx, cw), (txt, al, col, fnt) in zip(xs, vals):
            tw = d.textlength(txt, font=fnt)
            px = xx + 16 if al == "l" else xx + cw // 2 - tw // 2
            d.text((px, y + row_h // 2 - 14), txt, font=fnt, fill=col)

    d.text((40, H - 44),
           "WER via Whisper-large-v3 and NVIDIA NeMo FastConformer.  UTMOS is an English-trained naturalness proxy.",
           font=ImageFont.truetype(DJ, 15), fill=MUTED)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, "PNG")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
