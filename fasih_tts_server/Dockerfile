# Fasih-TTS server — Arabic (MSA/Fusha) XTTS, GPU. Self-hosted TTS for the Muslim agent.
# Build from the repo root:
#   docker build -f fasih_tts_server/Dockerfile -t nightprincey/muslim-fasih-tts:v1 .
# Run:
#   docker run --gpus all -p 3006:3006 nightprincey/muslim-fasih-tts:v1
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    MODEL_DIR=/app/model \
    CATT_CKPT=/app/models/catt/best_ed_mlm_ns_epoch_178.pt \
    TTS_HOST=0.0.0.0 \
    TTS_PORT=3006 \
    TTS_DEVICE=cuda

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 curl && rm -rf /var/lib/apt/lists/*

# Python deps (torch/torchaudio come from the base image)
COPY fasih_tts_server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Bake the model in (weights + config + vocab + precomputed speaker latents) — offline startup
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('NightPrince/Fasih-TTS-V1', local_dir='/app/model', \
allow_patterns=['model.pth','config.json','vocab.json','speaker_latents.pt'])"

# Bake the CATT diacritizer checkpoint
RUN mkdir -p /app/models/catt && \
    curl -fsSL -o /app/models/catt/best_ed_mlm_ns_epoch_178.pt \
    https://github.com/abjadai/catt/releases/download/v2/best_ed_mlm_ns_epoch_178.pt

# App code: Arabic front-end package + sacred-term lexicon + server
COPY src/tts ./tts
COPY configs/data/lexicon_ar.yaml ./configs/data/lexicon_ar.yaml
COPY fasih_tts_server/server.py ./server.py

EXPOSE 3006

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD python -c "import urllib.request,json,sys; \
r=json.load(urllib.request.urlopen('http://127.0.0.1:3006/health')); \
sys.exit(0 if r.get('status')=='ok' else 1)"

CMD ["python", "server.py"]
