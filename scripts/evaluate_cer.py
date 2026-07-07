"""Phase-7: objective intelligibility eval via Arabic ASR (CER).

Transcribes audio with Whisper (large-v3) and computes CER against the intended text.
Also measures the ASR error floor on ORIGINAL human clips, so TTS CER is judged against
the human ceiling (Whisper itself errs on Arabic, so absolute CER isn't meaningful alone).

Both hyp and ref are diacritics-stripped + orthography-normalized for a fair comparison
(ASR doesn't reliably emit diacritics).

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/evaluate_cer.py
"""

from __future__ import annotations

import csv
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tts.text.normalize import strip_diacritics  # noqa: E402

import jiwer  # noqa: E402

SENTENCES = Path("configs/eval_sentences_ar.txt")
MANIFEST = Path("data/manifests/train.csv")
WAV24 = Path("data/processed/wav24")


def norm(t: str) -> str:
    """Diacritics-stripped, orthography-normalized text for fair CER."""
    t = strip_diacritics(unicodedata.normalize("NFC", t))
    t = re.sub("[إأآا]", "ا", t)         # unify alef forms
    t = t.replace("ى", "ي").replace("ة", "ه").replace("ـ", "")
    t = re.sub(r"[^؀-ۿ\s]", " ", t)  # keep Arabic letters only
    return re.sub(r"\s+", " ", t).strip()


def main() -> int:
    import argparse

    from faster_whisper import WhisperModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", default=str(SENTENCES))
    ap.add_argument("--audio_dir", default="outputs/consistency_test")
    ap.add_argument("--no_human", action="store_true", help="skip the human ASR-floor set")
    args = ap.parse_args()

    print("loading Whisper large-v3 ...")
    asr = WhisperModel("large-v3", device="cuda", compute_type="float16")

    def transcribe(path: str) -> str:
        segs, _ = asr.transcribe(path, language="ar", beam_size=5)
        return " ".join(s.text for s in segs)

    def cer_set(pairs, label):
        cers = []
        print(f"\n=== {label} ===")
        for name, ref, wav in pairs:
            if not Path(wav).exists():
                continue
            hyp = transcribe(wav)
            c = jiwer.cer(norm(ref), norm(hyp))
            cers.append(c)
            flag = "  <-- HIGH" if c > 0.20 else ""
            print(f"  {name:10s} CER {c*100:5.1f}%{flag}")
        if cers:
            cers.sort()
            n = len(cers)
            print(f"  --> mean {sum(cers)/n*100:.1f}%  median {cers[n//2]*100:.1f}%  "
                  f"worst {cers[-1]*100:.1f}%  (n={n})")
        return cers

    lines = [ln.strip() for ln in open(args.sentences, encoding="utf-8")
             if ln.strip() and not ln.startswith("#")]
    tts = [(f"clip_{i:02d}", lines[i], f"{args.audio_dir}/{i:02d}.wav")
           for i in range(len(lines))]
    # include variance clips only for the default consistency set
    if args.audio_dir == "outputs/consistency_test":
        tts += [(f"rep_{r}", lines[0], f"outputs/variance_test/00_r{r}.wav") for r in range(4)]

    tts_cers = cer_set(tts, "TTS OUTPUT (fine-tuned XTTS)")
    human_cers = []
    if not args.no_human:
        rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))[:8]
        human = [(f"human_{r['file']}", r["text"], str(WAV24 / r["file"])) for r in rows]
        human_cers = cer_set(human, "HUMAN ORIGINAL (ASR floor)")

    if tts_cers and human_cers:
        import statistics as st
        print("\n" + "=" * 50)
        print(f"TTS   mean CER : {st.mean(tts_cers)*100:.1f}%")
        print(f"HUMAN mean CER : {st.mean(human_cers)*100:.1f}%  (ASR floor)")
        gap = (st.mean(tts_cers) - st.mean(human_cers)) * 100
        print(f"GAP           : {gap:+.1f} pts  "
              f"({'TTS ~ human intelligibility' if gap < 8 else 'TTS notably worse — inspect'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
