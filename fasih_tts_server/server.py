"""Fasih-TTS Server — Arabic (MSA/Fusha) XTTS (GPU).

Self-hosted TTS for the Muslim voice agent (drop-in for the cloud TTS). Full Arabic front-end
(normalize -> numbers -> CATT diacritization -> sacred-term lexicon -> chunking) + XTTS synthesis.

Endpoints:
  GET  /health                      -> {"status": "ok", ...}
  GET  /info                        -> model/voice info
  POST /v1/tts   {"text": "..."}    -> WAV (audio/wav)     [batch]
  POST /tts/stream {"text": "..."}  -> raw PCM16 mono @ 24 kHz [low-latency]

Env:
  MODEL_DIR   default /app/model    (config.json, vocab.json, model.pth, speaker_latents.pt)
  CATT_CKPT   default /app/models/catt/best_ed_mlm_ns_epoch_178.pt
  TTS_HOST    default 0.0.0.0
  TTS_PORT    default 3006
  TTS_DEVICE  default cuda
"""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model")
CATT_CKPT = os.environ.get("CATT_CKPT", "/app/models/catt/best_ed_mlm_ns_epoch_178.pt")
HOST = os.environ.get("TTS_HOST", "0.0.0.0")
PORT = int(os.environ.get("TTS_PORT", "3006"))
DEVICE = os.environ.get("TTS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
SR = 24000
MODEL_ID = "Fasih-TTS-V1"

app = FastAPI(title="Fasih-TTS Server", version="1.0",
              description="Self-hosted Arabic (Fusha) TTS — Fasih-TTS-V1 (XTTS v2 fine-tune).")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model = None
_gpt = None
_spk = None
_pipe = None


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Arabic text (diacritized or not).")
    temperature: float = Field(0.65, ge=0.1, le=1.0)
    auto_diacritize: bool = Field(True, description="Add tashkeel via CATT if text is under-diacritized.")


def load():
    global _model, _gpt, _spk, _pipe
    if _model is not None:
        return
    from tts.text.pipeline import TextPipeline
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    cfg = XttsConfig()
    cfg.load_json(f"{MODEL_DIR}/config.json")
    m = Xtts.init_from_config(cfg)
    m.load_checkpoint(cfg, checkpoint_path=f"{MODEL_DIR}/model.pth",
                      vocab_path=f"{MODEL_DIR}/vocab.json", use_deepspeed=False)
    m.to(DEVICE).eval()
    lat = torch.load(f"{MODEL_DIR}/speaker_latents.pt", map_location=DEVICE)
    _gpt, _spk = lat["gpt_cond_latent"].to(DEVICE), lat["speaker_embedding"].to(DEVICE)

    diac = None
    try:
        from tts.text.diacritize import Diacritizer
        diac = Diacritizer(ckpt=CATT_CKPT, device=DEVICE)
    except Exception as e:  # noqa: BLE001
        print("diacritizer unavailable:", e)
    globals()["_pipe"] = TextPipeline(diacritizer=diac)
    globals()["_model"] = m
    print(f"Fasih-TTS ready on {DEVICE}")


@app.on_event("startup")
def _startup():
    load()


def _chunks(text: str, auto: bool) -> list[str]:
    if auto:
        return _pipe.prepare_chunks(text)
    from tts.text.chunk import chunk_text
    from tts.text.normalize import normalize
    return chunk_text(normalize(text), 160)


@torch.inference_mode()
def _synth(text: str, temperature: float, auto: bool) -> np.ndarray:
    gap = np.zeros(int(SR * 0.12), dtype=np.float32)
    chunks = _chunks(text, auto)
    pieces = []
    for i, ch in enumerate(chunks):
        out = _model.inference(ch, "ar", _gpt, _spk, temperature=float(temperature),
                               repetition_penalty=2.0, enable_text_splitting=False)
        pieces.append(np.asarray(out["wav"], dtype=np.float32))
        if i < len(chunks) - 1:
            pieces.append(gap)
    return np.concatenate(pieces) if pieces else np.zeros(1, np.float32)


@app.get("/health")
def health():
    return {"status": "ok" if _model is not None else "loading", "model": MODEL_ID, "device": DEVICE}


@app.get("/info")
def info():
    return {"model": MODEL_ID, "language": "ar", "sample_rate": SR, "device": DEVICE,
            "voice": "professional male (MSA/Fusha)"}


@app.post("/v1/tts")
def v1_tts(req: TTSRequest):
    if _model is None:
        raise HTTPException(503, "model loading")
    t = time.time()
    wav = _synth(req.text, req.temperature, req.auto_diacritize)
    buf = io.BytesIO()
    sf.write(buf, wav, SR, format="WAV", subtype="PCM_16")
    dur = len(wav) / SR
    return Response(content=buf.getvalue(), media_type="audio/wav",
                   headers={"X-Audio-Duration": f"{dur:.2f}", "X-RTF": f"{(time.time()-t)/max(dur,1e-3):.2f}"})


@app.post("/tts/stream")
def tts_stream(req: TTSRequest):
    if _model is None:
        raise HTTPException(503, "model loading")

    @torch.inference_mode()
    def gen():
        for ch in _chunks(req.text, req.auto_diacritize):
            for piece in _model.inference_stream(ch, "ar", _gpt, _spk,
                                                 temperature=float(req.temperature),
                                                 repetition_penalty=2.0, enable_text_splitting=False):
                yield (np.clip(piece.cpu().numpy(), -1, 1) * 32767).astype("<i2").tobytes()

    return StreamingResponse(gen(), media_type="audio/L16; rate=24000; channels=1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
