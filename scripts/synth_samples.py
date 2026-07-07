"""Batch-synthesize a set of sentences from the fine-tuned XTTS (loads model once).

Used for consistency stress-testing and eval sample generation. Inference settings are
locked for production-style determinism (low temperature + repetition penalty).

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/synth_samples.py \
        --model models/xtts_ar_v1_best/model.pth --ref data/processed/wav24/10.wav \
        --sentences configs/eval_sentences_ar.txt --out outputs/consistency_test
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torchaudio

sys_path = str(Path(__file__).resolve().parents[1] / "src")
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)
from tts.text.chunk import chunk_text  # noqa: E402

BASE = Path("models/xtts_v2_base")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/xtts_ar_v1_best/model.pth")
    ap.add_argument("--ref", default="data/processed/wav24/10.wav")
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--out", default="outputs/consistency_test")
    ap.add_argument("--temperature", type=float, default=0.65)
    ap.add_argument("--repeat", type=int, default=1, help="synthesize each line N times")
    args = ap.parse_args()

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    cfg = XttsConfig()
    cfg.load_json(str(BASE / "config.json"))
    m = Xtts.init_from_config(cfg)
    m.load_checkpoint(cfg, checkpoint_path=args.model, vocab_path=str(BASE / "vocab.json"),
                      use_deepspeed=False)
    m.cuda().eval()
    gpt_cond, spk = m.get_conditioning_latents(audio_path=[args.ref])
    print("model loaded")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = [ln.strip() for ln in open(args.sentences, encoding="utf-8")
             if ln.strip() and not ln.startswith("#")]

    rtfs = []
    gap = torch.zeros(1, int(24000 * 0.12))  # 120 ms between chunks
    for i, text in enumerate(lines):
        chunks = chunk_text(text, max_chars=160)
        for r in range(args.repeat):
            t = time.time()
            pieces = []
            for ci, ch in enumerate(chunks):
                res = m.inference(ch, "ar", gpt_cond, spk, temperature=args.temperature,
                                  repetition_penalty=2.0, enable_text_splitting=False)
                pieces.append(torch.tensor(res["wav"]).unsqueeze(0))
                if ci < len(chunks) - 1:
                    pieces.append(gap)
            el = time.time() - t
            wav = torch.cat(pieces, dim=1)
            dur = wav.shape[1] / 24000
            rtfs.append(el / dur)
            tag = f"{i:02d}" + (f"_r{r}" if args.repeat > 1 else "")
            torchaudio.save(str(out / f"{tag}.wav"), wav, 24000)
            print(f"{tag}: {dur:4.1f}s  RTF {el/dur:.2f}  chunks={len(chunks)}  | {text[:40]}")
    print(f"\n{len(rtfs)} clips -> {out}  | RTF mean {sum(rtfs)/len(rtfs):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
