"""Phase-5b: materialize the XTTS (LJSpeech-format) dataset from our manifests.

Creates:
    data/processed/xtts/wavs/*.wav              (symlinks to wav24 masters)
    data/processed/xtts/metadata_train.txt      "id|text|text"
    data/processed/xtts/metadata_eval.txt

Long clips (>max_wav_seconds) are excluded — XTTS GPT caps audio length and would skip
them anyway; excluding keeps the manifest honest.
"""

from __future__ import annotations

import csv
from pathlib import Path

WAV24 = Path("data/processed/wav24").resolve()
OUT = Path("data/processed/xtts")
MANI = Path("data/manifests")
MAX_WAV_SECONDS = 11.5  # XTTS GPT default max (~255995 samples @ 22050)
MAX_TEXT_CHARS = 166    # XTTS 'ar' char limit; longer text -> truncated audio (mismatch)


def build_split(split: str) -> int:
    rows = list(csv.DictReader(open(MANI / f"{split}.csv", encoding="utf-8")))
    wavs = OUT / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(OUT / f"metadata_{'eval' if split=='val' else 'train'}.txt", "w",
              encoding="utf-8") as f:
        for r in rows:
            if float(r["duration"]) > MAX_WAV_SECONDS:
                continue
            text = r["text"].replace("|", " ").replace("\n", " ").strip()
            if len(text) > MAX_TEXT_CHARS:  # XTTS would truncate audio -> skip
                continue
            stem = Path(r["file"]).stem
            link = wavs / r["file"]
            if not link.exists():
                link.symlink_to(WAV24 / r["file"])
            f.write(f"{stem}|{text}|{text}\n")
            kept += 1
    return kept, len(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        kept, total = build_split(split)
        print(f"{split}: kept {kept}/{total} (dropped {total-kept} > {MAX_WAV_SECONDS}s)")
    print(f"XTTS dataset -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
