# NeMo STT — Arabic FastConformer

LiveKit-compatible HTTP speech-to-text server using
**`nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`**. Used in this project as (1) the second
independent ASR judge for the TTS benchmark, and (2) a deployable STT for the voice agent.

Kept in an **isolated venv** (`nemo_stt/.venv`) so NeMo's heavy deps don't touch the TTS env.

## Setup
```bash
uv venv nemo_stt/.venv --python 3.12
uv pip install --python nemo_stt/.venv/bin/python "nemo_toolkit[asr]" fastapi "uvicorn[standard]" ffmpeg-python num2words jiwer
# model (459 MB) -> nemo_stt/models/stt_ar_fastconformer_hybrid_large_pcd_v1.0.nemo
```

## Run the server
```bash
CUDA_VISIBLE_DEVICES=0 nemo_stt/.venv/bin/python nemo_stt/server.py    # :3005
curl -X POST localhost:3005/transcribe --data-binary @clip.wav -H 'Content-Type: audio/wav'
```

`POST /transcribe` accepts 16 kHz mono PCM / WAV / MP3 → `{"text": "...", "is_final": true}`.
CUDA graphs are disabled in `load_model()` (works around a decoder bug on some CUDA/PyTorch combos).

## As a benchmark judge
`scripts/nemo_compare.py` (run with `nemo_stt/.venv/bin/python`) transcribes all SILMA models'
audio and computes WER/CER — the second judge behind the numbers in the model card.

*Note: `.venv/` and `models/` are gitignored; only `server.py` + this README are tracked.*
