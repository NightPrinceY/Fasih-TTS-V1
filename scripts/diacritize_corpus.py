"""Phase-3: diacritize the undiacritized clips (1147-1517) of Dataset A with CATT.

Gold-diacritized clips (1-1146) are left untouched. Produces:
  * data/interim/metadata_diacritized.csv  — all 1517 rows, ready for preprocessing
  * data/manifests/diacritization_review.csv — the 371 (in/out) for human review

Usage:
    CUDA_VISIBLE_DEVICES=0,1,4,7 uv run python scripts/diacritize_corpus.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tts.text.diacritics import coverage  # noqa: E402
from tts.text.diacritize import Diacritizer  # noqa: E402
from tts.text.normalize import normalize  # noqa: E402

SRC = Path("data/raw/Arabic-professional-original-voice/data/metadata.csv")
OUT = Path("data/interim/metadata_diacritized.csv")
REVIEW = Path("data/manifests/diacritization_review.csv")
THRESHOLD = 0.30


def main() -> int:
    rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
    to_fix = [r for r in rows if coverage(normalize(r["transcription"])).ratio < THRESHOLD]
    print(f"{len(rows)} clips total; {len(to_fix)} need diacritization (ratio < {THRESHOLD})")

    d = Diacritizer()
    print(f"CATT loaded on {d.device}. Diacritizing...")
    fixed = d.diacritize_texts([r["transcription"] for r in to_fix], batch_size=16, verbose=True)
    fixed_by_file = {r["file_name"]: new for r, new in zip(to_fix, fixed)}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    REVIEW.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "transcription", "diac_source"])
        for r in rows:
            fn = r["file_name"]
            if fn in fixed_by_file:
                w.writerow([fn, normalize(fixed_by_file[fn]), "catt"])
            else:
                w.writerow([fn, normalize(r["transcription"]), "gold"])

    with open(REVIEW, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "original_plain", "catt_diacritized", "coverage"])
        for r in to_fix:
            new = fixed_by_file[r["file_name"]]
            w.writerow([r["file_name"], r["transcription"], normalize(new),
                        coverage(new).ratio])

    print(f"\nwrote {OUT} (1517 rows: 1146 gold + {len(to_fix)} catt)")
    print(f"wrote {REVIEW} ({len(to_fix)} rows for human review)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
