"""Download the target-voice dataset (Dataset A) from the Hugging Face Hub.

Read-only, resumable snapshot into data/raw/. Token is read from .env (HF_TOKEN);
it is never printed or written anywhere.

Usage:
    uv run python scripts/download_data.py                     # Dataset A (target voice)
    uv run python scripts/download_data.py --repo <id> --out <dir>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

DATASET_A = "NightPrince/Arabic-professional-original-voice"


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DATASET_A, help="HF dataset repo id")
    ap.add_argument("--out", default=None, help="target dir under data/raw/")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set (put it in .env).")
        return 1

    from huggingface_hub import snapshot_download

    out = args.out or f"data/raw/{args.repo.split('/')[-1]}"
    Path(out).mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo} -> {out}")
    path = snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=out,
        token=token,
        resume_download=True,
    )
    n = sum(1 for _ in Path(path).rglob("*") if _.is_file())
    print(f"Done. {n} files in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
