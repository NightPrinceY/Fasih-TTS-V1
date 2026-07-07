"""Synthesize Arabic speech with a fine-tuned XTTS v2 model + measure latency.

Loads the fine-tuned checkpoint against the base config/vocab, conditions on a reference
clip of the target voice, and writes a WAV. Also reports latency (RTF).

Usage:
    CUDA_VISIBLE_DEVICES=4 uv run python scripts/synthesize_xtts.py \
        --checkpoint experiments/xtts_ar_v1/<run>/best_model.pth \
        --ref data/processed/wav24/1.wav \
        --text "السَّلَامُ عَلَيْكُمْ" --out outputs/xtts_test.wav
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

BASE = Path("models/xtts_v2_base")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ref", default="data/processed/wav24/1.wav", help="speaker reference wav")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="outputs/xtts_test.wav")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    import torchaudio
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    config = XttsConfig()
    config.load_json(str(BASE / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config, checkpoint_path=args.checkpoint,
        vocab_path=str(BASE / "vocab.json"), use_deepspeed=False,
    )
    model.cuda().eval()

    t0 = time.time()
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[args.ref])
    t_cond = time.time() - t0

    t1 = time.time()
    out = model.inference(
        args.text, "ar", gpt_cond_latent, speaker_embedding,
        temperature=args.temperature, enable_text_splitting=True,
    )
    t_inf = time.time() - t1

    wav = torch.tensor(out["wav"]).unsqueeze(0)
    dur = wav.shape[1] / 24000
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(args.out, wav, 24000)

    print(f"cond latents : {t_cond:.2f}s")
    print(f"inference    : {t_inf:.2f}s for {dur:.2f}s audio -> RTF {t_inf/dur:.3f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
