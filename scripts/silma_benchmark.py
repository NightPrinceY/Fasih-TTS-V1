"""Evaluate Fasih-TTS-V1 on the SILMA open-source Arabic TTS benchmark (MSA, 10 sentences).

Step 1 (this script): synthesize Fasih audio on the 10 MSA benchmark sentences with the full
production front-end (auto-diacritization + number expansion + chunking).

Outputs:
    outputs/silma_msa/fasih_{i}_msa.wav
    outputs/silma_msa/reference_texts.txt   (the 10 reference sentences, one per line)

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/silma_benchmark.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SPACE = "silma-ai/opensource-arabic-tts-benchmark"
OUT = Path("outputs/silma_msa")


def main() -> int:
    tok = None
    for line in open(".env"):
        if line.startswith("HF_TOKEN="):
            tok = line.split("=", 1)[1].strip()

    import pandas as pd
    from huggingface_hub import hf_hub_download

    csv = hf_hub_download(SPACE, "results/msa/Ar_msa_TTS_benchmark.csv",
                          repo_type="space", token=tok)
    texts = pd.read_csv(csv)["Text"].tolist()
    print(f"{len(texts)} MSA benchmark sentences")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reference_texts.txt").write_text("\n".join(texts) + "\n", encoding="utf-8")

    from tts.infer.engine import XttsEngine
    eng = XttsEngine(use_diacritizer=True)
    print("engine ready; synthesizing...")

    for i, text in enumerate(texts, 1):
        wav, sr = eng.synthesize(text)
        sf.write(str(OUT / f"fasih_{i}_msa.wav"), wav, sr)
        print(f"  fasih_{i}_msa.wav  ({len(wav)/sr:.1f}s)  | {text[:45]}")
    print(f"\ndone -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
