"""Phase-5c: fine-tune XTTS v2 on the Arabic professional-male corpus.

Single-GPU, FP16 (Turing). Run:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/train_xtts.py --config configs/training/xtts_finetune.yaml
Smoke test (few samples, 1 epoch, verifies loop + memory):
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/train_xtts.py --config ... --smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate the loop")
    ap.add_argument("--restore", default=None, help="checkpoint .pth to resume from")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    base = Path(cfg["base_model_dir"])
    dataset = BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name="arabic_pro",
        path=cfg["dataset_path"],
        meta_file_train=cfg["meta_train"],
        meta_file_val=cfg["meta_eval"],
        language=cfg["language"],
    )

    model_args = GPTArgs(
        max_conditioning_length=cfg["max_conditioning_length"],
        min_conditioning_length=cfg["min_conditioning_length"],
        max_wav_length=cfg["max_wav_length"],
        max_text_length=cfg["max_text_length"],
        mel_norm_file=str(base / "mel_stats.pth"),
        dvae_checkpoint=str(base / "dvae.pth"),
        xtts_checkpoint=str(base / "model.pth"),
        tokenizer_file=str(base / "vocab.json"),
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    audio = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

    run_config = GPTTrainerConfig(
        epochs=3 if args.smoke else cfg["epochs"],
        output_path=cfg["output_path"],
        model_args=model_args,
        audio=audio,
        run_name="xtts_ar_v1",
        project_name="arabic_tts",
        run_description="XTTS v2 fine-tune, Arabic professional male (diacritized Fusha)",
        batch_size=cfg["batch_size"],
        batch_group_size=48,
        eval_batch_size=cfg["batch_size"],
        num_loader_workers=cfg["num_workers"],
        eval_split_max_size=256,
        print_step=1 if args.smoke else cfg["print_step"],
        save_step=cfg["save_step"],
        save_n_checkpoints=cfg["save_n_checkpoints"],
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8,
                          "weight_decay": cfg["weight_decay"]},
        lr=cfg["lr"],
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000, 150000, 300000], "gamma": 0.5,
                             "last_epoch": -1},
        mixed_precision=cfg["mixed_precision"],
    )

    train_samples, eval_samples = load_tts_samples(
        [dataset], eval_split=True, eval_split_max_size=256, eval_split_size=0.01
    )
    if args.smoke:
        train_samples, eval_samples = train_samples[:40], eval_samples[:8]
    print(f"train samples: {len(train_samples)} | eval samples: {len(eval_samples)}")

    model = GPTTrainer.init_from_config(run_config)

    # Gradient checkpointing on the inner HF GPT2 stack cuts peak VRAM (recompute in
    # backward), keeping all <=11.5s clips fitting safely in 11 GiB next to a co-tenant.
    enabled = 0
    for m in model.modules():
        if m.__class__.__name__ == "GPT2Model" and hasattr(m, "gradient_checkpointing_enable"):
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            enabled += 1
    print(f"gradient checkpointing enabled on {enabled} GPT2 stack(s)")

    trainer = Trainer(
        TrainerArgs(restore_path=args.restore, skip_train_epoch=False,
                    start_with_eval=False, grad_accum_steps=cfg["grad_accum"]),
        run_config,
        output_path=cfg["output_path"],
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
