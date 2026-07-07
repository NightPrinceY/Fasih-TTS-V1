"""Phase-2 dataset validation for Dataset A.

Builds a full per-clip manifest (audio QC + transcript diacritization coverage) and
prints a health summary. Parallelised across CPU cores.

Usage:
    uv run python scripts/validate_dataset.py
"""

from __future__ import annotations

import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tts.audio.quality import analyze_clip  # noqa: E402
from tts.text.diacritics import coverage  # noqa: E402
from tts.text.normalize import normalize  # noqa: E402

ROOT = Path("data/raw/Arabic-professional-original-voice/data")
OUT = Path("data/manifests/dataset_a_manifest.csv")


def _qc(path: str) -> dict:
    try:
        return analyze_clip(path)
    except Exception as e:  # noqa: BLE001
        return {"path": Path(path).name, "ok": False, "flags": f"load_error:{e}"}


def main() -> int:
    rows = list(csv.DictReader(open(ROOT / "metadata.csv", encoding="utf-8")))
    paths = [str(ROOT / r["file_name"]) for r in rows]
    print(f"Analyzing {len(paths)} clips across CPU cores...")

    with ProcessPoolExecutor(max_workers=16) as ex:
        qc = list(ex.map(_qc, paths, chunksize=16))

    df = pd.DataFrame(qc)
    # merge transcript info
    meta = {r["file_name"]: r["transcription"] for r in rows}
    df["transcription"] = df["path"].map(meta)
    cov = df["transcription"].map(lambda t: coverage(normalize(t or "")))
    df["diac_ratio"] = cov.map(lambda c: c.ratio)
    df["diac_level"] = cov.map(lambda c: c.level)
    df["n_words"] = df["transcription"].map(lambda t: len((t or "").split()))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    # ---- summary ----
    print("\n" + "=" * 60 + "\nDATASET A — HEALTH REPORT\n" + "=" * 60)
    print(f"clips                : {len(df)}")
    print(f"total duration       : {df['duration'].sum()/3600:.2f} h")
    print(f"duration s (min/med/max): {df['duration'].min():.2f} / "
          f"{df['duration'].median():.2f} / {df['duration'].max():.2f}")
    print(f"sample rates         : {sorted(df['sr'].dropna().unique().tolist())}")
    print(f"peak (max)           : {df['peak'].max():.4f}")
    print(f"rms dBFS (med)       : {df['rms_dbfs'].median():.1f}")
    print("\n-- flags --")
    from collections import Counter
    fl = Counter()
    for f in df["flags"].dropna():
        for x in filter(None, str(f).split(";")):
            fl[x] += 1
    for k, v in fl.most_common():
        print(f"  {k:16s}: {v}")
    print(f"\nclean clips (no flags): {(df['ok']==True).sum()} / {len(df)}")
    print("\n-- diacritization --")
    print(df["diac_level"].value_counts().to_string())
    print(f"\nmanifest written -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
