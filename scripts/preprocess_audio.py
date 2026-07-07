"""Phase-4: build the clean training set from diacritized transcripts + MP3s.

Produces 24 kHz mono WAV masters and generic train/val manifests. Framework-specific
manifests (XTTS / F5) are derived from these in Phases 5-6.

Usage:
    uv run python scripts/preprocess_audio.py
"""

from __future__ import annotations

import csv
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tts.audio.preprocess import process_clip  # noqa: E402
from tts.text.normalize import normalize  # noqa: E402

META = Path("data/interim/metadata_diacritized.csv")
RAW = Path("data/raw/Arabic-professional-original-voice/data")
WAV_OUT = Path("data/processed/wav24")
MANI = Path("data/manifests")
TARGET_SR = 24000
MAX_DUR = 15.0          # clips longer than this are flagged (kept, filterable by trainer)
MIN_DUR = 1.0
VAL_FRAC = 0.05
SEED = 42


def _work(args):
    fn, text, src = args
    r = process_clip(str(RAW / fn), str(WAV_OUT / fn.replace(".mp3", ".wav")), TARGET_SR)
    return {"file": fn.replace(".mp3", ".wav"), "text": normalize(text),
            "diac_source": src, "duration": r["duration"], "ok": r["ok"]}


def main() -> int:
    rows = list(csv.DictReader(open(META, encoding="utf-8")))
    tasks = [(r["file_name"], r["transcription"], r["diac_source"]) for r in rows]
    print(f"Preprocessing {len(tasks)} clips -> {WAV_OUT} @ {TARGET_SR} Hz ...")

    with ProcessPoolExecutor(max_workers=16) as ex:
        out = list(ex.map(_work, tasks, chunksize=16))

    # filter + flag
    good, dropped = [], []
    for o in out:
        if not o["ok"] or o["duration"] < MIN_DUR:
            dropped.append(o)
            continue
        o["flags"] = "too_long" if o["duration"] > MAX_DUR else ""
        good.append(o)

    total_h = sum(o["duration"] for o in good) / 3600
    n_long = sum(1 for o in good if o["flags"] == "too_long")
    print(f"kept {len(good)} clips ({total_h:.2f} h); dropped {len(dropped)}; "
          f"{n_long} flagged too_long (>{MAX_DUR}s)")

    # stratified train/val split (by diac_source so val has both gold + catt)
    rng = random.Random(SEED)
    val = set()
    from collections import defaultdict
    by_src = defaultdict(list)
    for o in good:
        by_src[o["diac_source"]].append(o["file"])
    for src, files in by_src.items():
        rng.shuffle(files)
        k = max(1, int(len(files) * VAL_FRAC))
        val.update(files[:k])

    MANI.mkdir(parents=True, exist_ok=True)
    cols = ["file", "text", "diac_source", "duration", "flags"]
    for split, keep in (("train", lambda f: f not in val), ("val", lambda f: f in val)):
        rowset = [o for o in good if keep(o["file"])]
        with open(MANI / f"{split}.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rowset)
        print(f"  {split}: {len(rowset)} clips "
              f"({sum(o['duration'] for o in rowset)/3600:.2f} h) -> {MANI/f'{split}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
