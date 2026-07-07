# Arabic Professional Fusha TTS

High-quality **Modern Standard Arabic (Fusha)** professional male Text-to-Speech for the
"Muslim" voice agent. Research-grade, reproducible, single-speaker fine-tuning of XTTS v2.

## Status: working end-to-end ✅
Raw Arabic text → auto-diacritized + number-expanded + chunked → natural speech, via HTTP/CLI.

- **Model:** XTTS v2 fine-tuned on 1297 clips (~2.4 h), val loss 2.622.
- **Quality (CER):** 1.3% mean, 2.2% worst — **on par with the human originals (1.8%)**;
  no instability across repeats/variety/long text. RTF ~0.6, streaming first-audio ~675 ms.

## Hardware (verified)
8× RTX 2080 Ti (11 GiB, **Turing sm_75 → FP32 training only, no bf16/FA2**), Ubuntu 24.04/WSL2,
CUDA 12.x, Python 3.12, uv. Use GPUs 0,1,4,7. Keep the machine awake (sleep kills CUDA).

## Setup
```bash
cp .env.example .env      # add HF_TOKEN
uv sync --extra xtts --extra diacritize --extra eval --extra serve
uv run python scripts/check_env.py
```

## Use it
```bash
# CLI
CUDA_VISIBLE_DEVICES=1 uv run python scripts/say.py "بارك الله فيك" --out outputs/hello.wav

# HTTP API (batch + streaming)
CUDA_VISIBLE_DEVICES=1 uv run uvicorn scripts.serve:app --host 0.0.0.0 --port 8000
curl -X POST localhost:8000/tts -H 'Content-Type: application/json' \
     -d '{"text":"الصلوات المفروضة 5 في اليوم"}' --output out.wav
# /tts/stream -> raw PCM16 mono @ 24 kHz for low-latency playback
```

## Reproduce the pipeline
```bash
uv run python scripts/download_data.py          # Dataset A
uv run python scripts/validate_dataset.py       # Phase 2 QC
CUDA_VISIBLE_DEVICES=1 uv run python scripts/diacritize_corpus.py   # Phase 3 (CATT)
uv run python scripts/preprocess_audio.py       # Phase 4 -> 24 kHz + manifests
uv run python scripts/build_xtts_dataset.py     # LJSpeech layout
scripts/run_xtts_tmux.sh                         # Phase 5 fine-tune (tmux, resumable)
CUDA_VISIBLE_DEVICES=1 uv run python scripts/evaluate_cer.py       # Phase 7
```

## Layout
`configs/` drive everything · `src/tts/` package (text/ audio/ infer/) · `scripts/` CLIs ·
`experiments/` runs · `models/xtts_ar_v1_best/` final model · `docs/` phase reports ·
`data/ models/ outputs/ logs/` gitignored. Tests: `uv run pytest` (13 pass).

See `docs/` for the full per-phase audit and reports.
