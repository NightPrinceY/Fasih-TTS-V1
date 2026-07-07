"""CLI: synthesize Arabic text to a WAV with the fine-tuned XTTS voice.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/say.py "مَرْحَبًا بِكَ" --out outputs/hello.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tts.infer.engine import XttsEngine  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--out", default="outputs/say.wav")
    ap.add_argument("--no-diac", action="store_true", help="skip diacritizer (text already diacritized)")
    args = ap.parse_args()

    eng = XttsEngine(use_diacritizer=not args.no_diac)
    wav, sr = eng.synthesize(args.text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, wav, sr)
    print(f"wrote {args.out} ({len(wav)/sr:.1f}s @ {sr} Hz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
