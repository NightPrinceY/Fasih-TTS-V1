"""Assemble the Fasih-TTS-Benchmark HF dataset: all eval audio + texts + per-clip WER/CER.

Gathers every test set we produced (SILMA benchmark, samples, consistency, variance, stress),
scores each clip with Whisper-large-v3, and writes an AudioFolder-style dataset (metadata.csv +
audio/) into a staging dir ready to push.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/build_eval_dataset.py
"""

from __future__ import annotations

import csv
import re
import shutil
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tts.text.normalize import strip_diacritics  # noqa: E402
from tts.text.numbers import expand_numbers  # noqa: E402

import jiwer  # noqa: E402

STAGE = Path("data/eval_dataset_staging")

SAMPLES = {
    "greeting": "السَّلَامُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ وَبَرَكَاتُهُ. أَنَا مُسْلِم، مُسَاعِدُكَ الصَّوْتِيُّ.",
    "fiqh": "الْوُضُوءُ شَرْطٌ لِصِحَّةِ الصَّلَاةِ، وَيَبْدَأُ بِالنِّيَّةِ ثُمَّ غَسْلِ الْوَجْهِ وَالْيَدَيْنِ.",
    "reflection": "مَا أَجْمَلَ أَنْ تَبْدَأَ يَوْمَكَ بِذِكْرِ اللَّهِ وَقِرَاءَةِ وِرْدِكَ الْيَوْمِيِّ.",
}


def norm(t: str) -> str:
    t = expand_numbers(unicodedata.normalize("NFC", str(t)))
    t = strip_diacritics(t)
    t = re.sub("[إأآا]", "ا", t).replace("ى", "ي").replace("ة", "ه").replace("ـ", "")
    t = re.sub(r"[^؀-ۿ\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def read_lines(p):
    return [ln.strip() for ln in open(p, encoding="utf-8") if ln.strip() and not ln.startswith("#")]


def main() -> int:
    items = []  # (src_path, text, test_set)
    refs = read_lines("outputs/silma_msa/reference_texts.txt")
    for i in range(10):
        items.append((f"outputs/silma_msa/fasih_{i+1}_msa.wav", refs[i], "silma_msa"))
    for name, text in SAMPLES.items():
        items.append((f"outputs/samples_final/{name}.wav", text, "samples"))
    econs = read_lines("configs/eval_sentences_ar.txt")
    for i, t in enumerate(econs):
        items.append((f"outputs/consistency_test/{i:02d}.wav", t, "consistency"))
    for r in range(4):
        items.append((f"outputs/variance_test/00_r{r}.wav", econs[0], "variance"))
    for i, t in enumerate(read_lines("configs/stress_sentences_ar.txt")):
        items.append((f"outputs/stress_test/{i:02d}.wav", t, "stress"))

    items = [(p, t, s) for p, t, s in items if Path(p).exists()]
    print(f"{len(items)} clips to package")

    from faster_whisper import WhisperModel
    asr = WhisperModel("large-v3", device="cuda", compute_type="float16")

    def transcribe(p):
        segs, _ = asr.transcribe(str(p), language="ar", beam_size=5)
        return " ".join(s.text for s in segs)

    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "audio").mkdir(parents=True)
    rows = []
    for src, text, tset in items:
        rel = f"audio/{tset}/{Path(src).name}"
        dest = STAGE / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest)
        hyp = norm(transcribe(src))
        rows.append({
            "file_name": rel, "text": text, "test_set": tset,
            "wer_pct": round(jiwer.wer(norm(text), hyp) * 100, 1),
            "cer_pct": round(jiwer.cer(norm(text), hyp) * 100, 1),
            "asr_transcription": hyp,
        })
        print(f"  {rel}  CER {rows[-1]['cer_pct']}%")

    with open(STAGE / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # also carry the full 6-model SILMA comparison for reference
    shutil.copy("assets/benchmark/silma_msa_scores.csv", STAGE / "silma_all_models_scores.csv")
    shutil.copy("assets/benchmark/silma_msa_detailed.csv", STAGE / "silma_all_models_detailed.csv")

    import statistics as st
    by = {}
    for r in rows:
        by.setdefault(r["test_set"], []).append(r["cer_pct"])
    print("\nper-set mean CER:", {k: round(st.mean(v), 1) for k, v in by.items()})
    print(f"staged -> {STAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
