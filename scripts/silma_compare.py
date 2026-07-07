"""SILMA MSA benchmark — objective WER/CER comparison across all models + Fasih.

Downloads every competitor's benchmark audio from the SILMA Space, transcribes all of them
(and Fasih's) with Whisper-large-v3, and computes WER/CER against the 10 reference sentences.

NOTE: SILMA's own benchmark is a *human auditory* comparison; they consider WER/CER
"insufficient" for Arabic naturalness. This objective table is a supplementary intelligibility
measure, not a naturalness ranking.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/silma_compare.py
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tts.text.normalize import strip_diacritics  # noqa: E402
from tts.text.numbers import expand_numbers  # noqa: E402

import jiwer  # noqa: E402

SPACE = "silma-ai/opensource-arabic-tts-benchmark"
FASIH_DIR = Path("outputs/silma_msa")
OUT = Path("assets/benchmark/silma_msa_scores.csv")          # aggregate leaderboard
DETAIL = Path("assets/benchmark/silma_msa_detailed.csv")     # per-clip provenance


def norm(t: str) -> str:
    t = expand_numbers(unicodedata.normalize("NFC", str(t)))
    t = strip_diacritics(t)
    t = re.sub("[إأآا]", "ا", t).replace("ى", "ي").replace("ة", "ه").replace("ـ", "")
    t = re.sub(r"[^؀-ۿ\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


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
    from faster_whisper import WhisperModel
    from huggingface_hub import hf_hub_download, snapshot_download

    refs = pd.read_csv(hf_hub_download(SPACE, "results/msa/Ar_msa_TTS_benchmark.csv",
                                       repo_type="space", token=tok))["Text"].tolist()
    refs_n = [norm(r) for r in refs]

    root = Path(snapshot_download(SPACE, repo_type="space", token=tok,
                                  allow_patterns=["results/msa/*"]))
    msa = root / "results" / "msa"
    models = {d.name: sorted([p for p in d.iterdir() if p.suffix in (".wav", ".mp3")],
                             key=lambda p: num_key(p.name))
              for d in msa.iterdir() if d.is_dir()}
    # add Fasih
    models["fasih (ours)"] = sorted(FASIH_DIR.glob("fasih_*.wav"), key=lambda p: num_key(p.name))

    print("loading Whisper large-v3 ...")
    asr = WhisperModel("large-v3", device="cuda", compute_type="float16")

    def transcribe(p):
        segs, _ = asr.transcribe(str(p), language="ar", beam_size=5)
        return " ".join(s.text for s in segs)

    rows, detail = [], []
    for name, files in models.items():
        if len(files) < 10:
            print(f"  skip {name}: only {len(files)} files"); continue
        wers, cers = [], []
        for i in range(10):
            hyp = norm(transcribe(files[i]))
            w, c = jiwer.wer(refs_n[i], hyp), jiwer.cer(refs_n[i], hyp)
            wers.append(w); cers.append(c)
            detail.append({"model": name, "sentence": i + 1, "audio_file": files[i].name,
                           "reference_norm": refs_n[i], "asr_transcription": hyp,
                           "WER%": round(w * 100, 1), "CER%": round(c * 100, 1)})
        rows.append({"model": name, "WER%": round(sum(wers)/10*100, 1),
                     "CER%": round(sum(cers)/10*100, 1)})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detail).to_csv(DETAIL, index=False)  # full per-clip provenance
    df = pd.DataFrame(rows).sort_values("WER%").reset_index(drop=True)
    df.index += 1
    df.to_csv(OUT)
    print("\n" + "=" * 46)
    print("SILMA MSA benchmark — objective WER/CER (Whisper-large-v3)")
    print("=" * 46)
    print(df.to_string())
    print(f"\nsaved -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
