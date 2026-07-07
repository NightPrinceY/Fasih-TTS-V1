"""HTTP TTS API for the 'Fasih' voice agent (fine-tuned Arabic XTTS).

Endpoints:
  GET  /health                      -> {"status": "ok"}
  POST /tts        {"text": "..."}  -> full WAV (audio/wav)
  POST /tts/stream {"text": "..."}  -> streamed raw PCM16 mono @ 24 kHz (low latency)

Run:
  CUDA_VISIBLE_DEVICES=1 uv run uvicorn scripts.serve:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tts.infer.engine import XttsEngine  # noqa: E402

app = FastAPI(title="Arabic Fusha TTS", version="1.0")
_engine: XttsEngine | None = None


class TTSRequest(BaseModel):
    text: str


def engine() -> XttsEngine:
    global _engine
    if _engine is None:
        _engine = XttsEngine(use_diacritizer=True)
    return _engine


@app.on_event("startup")
def _warmup() -> None:
    engine().synthesize("تَجْرِبَةٌ.")  # load + warm the model


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "sr": XttsEngine.SR}


@app.post("/tts")
def tts(req: TTSRequest) -> Response:
    wav, sr = engine().synthesize(req.text)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")


@app.post("/tts/stream")
def tts_stream(req: TTSRequest) -> StreamingResponse:
    def gen():
        for chunk in engine().stream(req.text):
            yield (np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes()

    # raw PCM16 mono @ 24 kHz — agent plays as it arrives
    return StreamingResponse(gen(), media_type="audio/L16; rate=24000; channels=1")
