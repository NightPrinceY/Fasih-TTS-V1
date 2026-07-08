# Fasih-TTS Server (Docker)

Self-hosted GPU TTS microservice for the **Muslim** voice agent — the same Arabic (MSA/Fusha)
front-end + [Fasih-TTS-V1](https://huggingface.co/NightPrince/Fasih-TTS-V1) XTTS voice used in the
demo, wrapped in a FastAPI service that drops into `docker-compose` beside `nemo_stt` (STT) and the
MCP servers.

## API

| Method | Path         | Body / Result |
|--------|--------------|---------------|
| GET    | `/health`    | `{"status":"ok","model":"Fasih-TTS-V1","device":"cuda"}` |
| GET    | `/info`      | model / voice / sample-rate metadata |
| POST   | `/v1/tts`    | `{"text": "...", "temperature": 0.65, "auto_diacritize": true}` → `audio/wav` (24 kHz) |
| POST   | `/tts/stream`| same body → raw **PCM16 mono @ 24 kHz** stream (`audio/L16`) for low-latency agents |

The model, config, vocab, precomputed speaker latents and the CATT diacritizer checkpoint are all
**baked into the image at build time**, so the container starts offline with no runtime downloads.

## Build

From the **repo root** (the build context needs `src/tts` and the lexicon):

```bash
docker build -f fasih_tts_server/Dockerfile -t nightprincey/muslim-fasih-tts:v1 .
```

## Run

```bash
docker run --gpus all -p 3006:3006 --name fasih-tts nightprincey/muslim-fasih-tts:v1
```

Verify:

```bash
curl localhost:3006/health
curl -X POST localhost:3006/v1/tts -H 'Content-Type: application/json' \
  -d '{"text":"السلام عليكم ورحمة الله وبركاته"}' --output out.wav
```

## Environment

| Var | Default | Purpose |
|-----|---------|---------|
| `TTS_PORT` | `3006` | listen port |
| `TTS_HOST` | `0.0.0.0` | bind address |
| `TTS_DEVICE` | `cuda` | `cuda` / `cpu` |
| `MODEL_DIR` | `/app/model` | baked model dir |
| `CATT_CKPT` | `/app/models/catt/best_ed_mlm_ns_epoch_178.pt` | diacritizer checkpoint |

## docker-compose (agent integration)

```yaml
  fasih-tts:
    image: nightprincey/muslim-fasih-tts:v1
    ports: ["3006:3006"]
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices: [{ driver: nvidia, count: 1, capabilities: [gpu] }]
```

Point the agent at it with `FASIH_TTS_URL=http://fasih-tts:3006` and call `/v1/tts` (or
`/tts/stream`) from a LiveKit TTS plugin.
