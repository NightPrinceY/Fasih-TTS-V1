"""SILMA MSA benchmark — 2nd ASR judge (NVIDIA NeMo Arabic FastConformer).

Cross-validates the Whisper WER/CER ranking with an independent ASR. Runs in the ISOLATED
nemo venv:  nemo_stt/.venv/bin/python scripts/nemo_compare.py

Reads outputs/silma_competitors/<model>/ + outputs/silma_msa/ (fasih) + reference_texts.txt.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import jiwer
from num2words import num2words

MODEL = "nemo_stt/models/stt_ar_fastconformer_hybrid_large_pcd_v1.0.nemo"
COMP = Path("outputs/silma_competitors")
FASIH = Path("outputs/silma_msa")
REFS = Path("outputs/silma_msa/reference_texts.txt")
OUT = Path("assets/benchmark/silma_msa_nemo.csv")

_HARAKAT = set("ًٌٍَُِّْٰٕٓٔ")
_AI = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def norm(t: str) -> str:
    t = unicodedata.normalize("NFC", str(t)).translate(_AI)
    t = re.sub(r"\d+", lambda m: num2words(int(m.group()), lang="ar"), t)
    t = "".join(c for c in t if c not in _HARAKAT)
    t = re.sub("[إأآا]", "ا", t).replace("ى", "ي").replace("ة", "ه").replace("ـ", "")
    t = re.sub(r"[^؀-ۿ\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def num_key(name: str) -> int:
    for p in Path(name).stem.split("_"):
        if p.isdigit():
            return int(p)
    return 0


def main() -> int:
    refs = [norm(x) for x in REFS.read_text(encoding="utf-8").splitlines() if x.strip()]

    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict
    m = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL)
    m.eval()
    try:
        with open_dict(m.cfg):
            m.cfg.decoding.greedy.use_cuda_graphs = False
            m.cfg.decoding.greedy.use_cuda_graph_decoder = False
        m.change_decoding_strategy(m.cfg.decoding)
    except Exception as e:
        print("cuda-graph disable warn:", e)
    print("NeMo model loaded")

    def text_of(h):
        return h.text if hasattr(h, "text") else (h if isinstance(h, str) else str(h))

    def transcribe(paths):
        out = m.transcribe([str(p) for p in paths], batch_size=4, verbose=False)
        if isinstance(out, tuple):
            out = out[0]
        return [norm(text_of(h)) for h in out]

    models = {d.name: sorted(d.iterdir(), key=lambda p: num_key(p.name))
              for d in COMP.iterdir() if d.is_dir()}
    models["fasih (ours)"] = sorted(FASIH.glob("fasih_*.wav"), key=lambda p: num_key(p.name))

    import pandas as pd
    rows = []
    for name, files in models.items():
        files = files[:10]
        if len(files) < 10:
            continue
        hyps = transcribe(files)
        wers = [jiwer.wer(refs[i], hyps[i]) for i in range(10)]
        cers = [jiwer.cer(refs[i], hyps[i]) for i in range(10)]
        rows.append({"model": name, "WER%": round(sum(wers)/10*100, 1),
                     "CER%": round(sum(cers)/10*100, 1)})

    df = pd.DataFrame(rows).sort_values("WER%").reset_index(drop=True)
    df.index += 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT)
    print("\n" + "=" * 46)
    print("SILMA MSA — NeMo FastConformer WER/CER (2nd judge)")
    print("=" * 46)
    print(df.to_string())
    print(f"\nsaved -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
