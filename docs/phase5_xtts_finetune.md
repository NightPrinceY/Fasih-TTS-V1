# Phase 5 — XTTS v2 Fine-tune (primary)

## Setup
- Installed `coqui-tts 0.27.5`; **pinned `transformers>=4.57,<5.0`** (5.x removes
  `isin_mps_friendly`, breaking coqui XTTS import).
- Base model: `coqui/XTTS-v2` (model 1.87 GB, dvae, mel_stats, speakers, vocab) → `models/xtts_v2_base`.
- **Validated the XTTS tokenizer preserves Arabic diacritics** (round-trips diacritized text
  with zero loss) — confirms XTTS can use our diacritized Fusha.
- Dataset: LJSpeech-format, `data/processed/xtts/` — **train 1329 / eval 68** (clips >11.5s
  excluded, ~2.27 h).

## Key engineering finding — Turing forces FP32
- **FP16 (mixed precision) → persistent NaN loss.** XTTS's GPT overflows under FP16 autocast
  in the forward pass; Turing (sm_75) has no bf16 fallback. Eval was finite (autocast off).
- **FP32 → stable** (loss ~0.15 → 0.13, decreasing). This is the only stable precision on 2080 Ti.

## Memory (11 GiB, shared with a ~1 GB co-tenant)
- FP32 + `batch_size=2` peaked ~10.9 GB (too tight).
- **Chosen: `batch_size=1`, `grad_accum=24`** (effective ~24) + **gradient checkpointing** on the
  inner HF GPT2 stack (`gradient_checkpointing_enable`, `use_reentrant=False`). Peak ~10.3 GB.
- Dominant cost is FP32 AdamW optimizer state (~4 GB), not activations.

## Run
- GPU 1 (single), FP32, 12 epochs, lr 5e-6, output `experiments/xtts_ar_v1/`.
- Checkpoints every 1000 steps + per epoch (crash-recoverable via `restore_path`).
- Launch: `CUDA_VISIBLE_DEVICES=1 uv run python scripts/train_xtts.py --config configs/training/xtts_finetune.yaml`
- The `(nan)` shown in the running-average is a display artifact (one anomalous batch); per-step
  losses stay finite and the model keeps improving.

## Next
On completion: synthesize eval samples, then Phase 6 (F5-TTS challenger) and Phase 7 (CER/UTMOS/latency bake-off).
