"""SILMA MSA multi-metric comparison table — warm ivory theme (see scripts/brand.py).

Merges silma_msa_scores.csv (Whisper), silma_msa_nemo.csv (NeMo), silma_msa_utmos.csv (UTMOS).
Output: assets/benchmark_msa.png
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brand as B  # noqa: E402
from PIL import ImageDraw  # noqa: E402

OUT = "assets/benchmark_msa.png"
HL = (228, 241, 237)   # light teal row


def main() -> int:
    w = pd.read_csv("assets/benchmark/silma_msa_scores.csv", index_col=0).rename(columns={"WER%": "wer_wh"})
    n = pd.read_csv("assets/benchmark/silma_msa_nemo.csv", index_col=0).rename(columns={"WER%": "wer_ne"})
    u = pd.read_csv("assets/benchmark/silma_msa_utmos.csv", index_col=0)
    df = w[["model", "wer_wh"]].merge(n[["model", "wer_ne"]], on="model").merge(u, on="model")
    df = df.sort_values("wer_ne").reset_index(drop=True)

    cols = [("Model", 330, "l"), ("WER · Whisper", 200, "c"), ("WER · NeMo", 200, "c"), ("UTMOS", 170, "c")]
    W = 44 + sum(c[1] for c in cols) + 90
    header_h, row_h = 158, 64
    H = header_h + len(df) * row_h + 66

    img = B.canvas(W, H, shapes=False)
    d = ImageDraw.Draw(img)
    d.text((44, 38), "SILMA Open-Source Arabic TTS Benchmark", font=B.poppins(30, "Bold"), fill=B.INK)
    d.text((46, 84), "Two independent ASR judges + naturalness  ·  10 MSA sentences  ·  lower WER / higher UTMOS is better",
           font=B.poppins(16, "Medium"), fill=B.GREY)

    # header
    x = 44
    xs = []
    hf = B.poppins(17, "SemiBold")
    for title, cw, _ in cols:
        xs.append((x, cw))
        tx = x + 14 if title == "Model" else x + cw / 2 - d.textlength(title, font=hf) / 2
        d.text((tx, header_h - 36), title, font=hf, fill=B.GREY)
        x += cw
    d.line([44, header_h - 4, W - 44, header_h - 4], fill=B.LINE, width=2)

    best_ne = df["wer_ne"].min()
    for r, (_, row) in enumerate(df.iterrows()):
        y = header_h + r * row_h
        ours = "fasih" in str(row["model"]).lower()
        name = "Fasih-TTS-V1 (ours)" if ours else str(row["model"])
        if ours:
            B.card(img, [40, y + 5, W - 40, y + row_h - 5], radius=12, fill=HL, accent=B.TEAL,
                   shadow_alpha=26, blur=12)
        vals = [
            (name, "l", B.INK, B.poppins(21, "SemiBold" if ours else "Medium")),
            (f'{row["wer_wh"]}%', "c", B.INK2, B.poppins(20, "Medium")),
            (f'{row["wer_ne"]}%', "c", B.TEAL if row["wer_ne"] == best_ne else B.INK2, B.poppins(20, "SemiBold" if row["wer_ne"] == best_ne else "Medium")),
            (f'{row["utmos"] if "utmos" in row else row["UTMOS"]:.2f}', "c", B.INK2, B.poppins(20, "Medium")),
        ]
        for (xx, cw), (txt, al, col, fnt) in zip(xs, vals):
            tw = d.textlength(txt, font=fnt)
            px = xx + 18 if al == "l" else xx + cw / 2 - tw / 2
            d.text((px, y + row_h / 2 - 14), txt, font=fnt, fill=col)

    d.text((44, H - 44),
           "WER via Whisper-large-v3 and NVIDIA NeMo FastConformer.  UTMOS is an English-trained naturalness proxy.",
           font=B.poppins(14, "Medium"), fill=B.GREY)
    B.save(img, OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
