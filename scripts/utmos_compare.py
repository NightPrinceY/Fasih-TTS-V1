"""UTMOS (naturalness MOS) comparison across the SILMA MSA models + Fasih.

UTMOS22 (torch.hub, tarepan/SpeechMOS) predicts a naturalness MOS (1-5, higher better).
Runs on every model's 10 MSA benchmark clips. NOTE: UTMOS is trained on mostly-English MOS
data, so absolute Arabic values are a proxy — the RELATIVE ranking across models (all scored
identically) is the useful signal.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/utmos_compare.py
"""

from __future__ import annotations

from pathlib import Path

import librosa
import torch

SPACE = "silma-ai/opensource-arabic-tts-benchmark"
FASIH_DIR = Path("outputs/silma_msa")
OUT = Path("assets/benchmark/silma_msa_utmos.csv")


def num_key(fname: str) -> int:
    for part in Path(fname).stem.split("_"):
        if part.isdigit():
            return int(part)
    return 0


def main() -> int:
    tok = None
    for line in open(".env"):
        if line.startswith("HF_TOKEN="):
            tok = line.split("=", 1)[1].strip()

    import pandas as pd
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(SPACE, repo_type="space", token=tok,
                                  allow_patterns=["results/msa/*"]))
    msa = root / "results" / "msa"
    models = {d.name: sorted([p for p in d.iterdir() if p.suffix in (".wav", ".mp3")],
                             key=lambda p: num_key(p.name))
              for d in msa.iterdir() if d.is_dir()}
    models["fasih (ours)"] = sorted(FASIH_DIR.glob("fasih_*.wav"), key=lambda p: num_key(p.name))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                               trust_repo=True).to(dev).eval()

    @torch.no_grad()
    def utmos(path) -> float:
        y, _ = librosa.load(str(path), sr=16000, mono=True)
        t = torch.from_numpy(y).unsqueeze(0).to(dev)
        return float(predictor(t, 16000))

    rows = []
    for name, files in models.items():
        if len(files) < 10:
            continue
        scores = [utmos(f) for f in files[:10]]
        rows.append({"model": name, "UTMOS": round(sum(scores) / len(scores), 3)})

    df = pd.DataFrame(rows).sort_values("UTMOS", ascending=False).reset_index(drop=True)
    df.index += 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT)
    print("\n" + "=" * 40)
    print("SILMA MSA — UTMOS naturalness (higher is better)")
    print("=" * 40)
    print(df.to_string())
    print(f"\nsaved -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
