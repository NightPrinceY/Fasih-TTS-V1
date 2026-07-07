"""
NeMo STT Server - Arabic FastConformer
LiveKit-compatible HTTP API for speech-to-text.
Model: nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0
Input: 16kHz mono PCM or WAV
"""
import logging
import os
import tempfile

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

MODEL_NAME = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
_MODEL_FILENAME = "stt_ar_fastconformer_hybrid_large_pcd_v1.0.nemo"
# Prefer env; then local nemo_stt/models/ (no HF download); else Docker /app/
_server_dir = os.path.dirname(os.path.abspath(__file__))
_local_model = os.path.join(_server_dir, "models", _MODEL_FILENAME)
MODEL_PATH = os.getenv("NEMO_MODEL_PATH") or (
    _local_model if os.path.isfile(_local_model) else f"/app/{_MODEL_FILENAME}"
)
SAMPLE_RATE = 16000

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NeMo STT Server", version="0.1.0")
asr_model = None


def load_model():
    global asr_model
    if asr_model is not None:
        return
    try:
        import nemo.collections.asr as nemo_asr
        if os.path.isfile(MODEL_PATH):
            logger.info("Loading model from %s", MODEL_PATH)
            asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL_PATH)
        else:
            logger.info("Model file not found, loading from_pretrained %s", MODEL_NAME)
            asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name=MODEL_NAME)
        asr_model.eval()
        # Disable CUDA graphs — two separate flags both need to be off.
        # use_cuda_graphs controls the greedy path; use_cuda_graph_decoder
        # controls the loop_labels path. Both hit the same broken cu_call()
        # that returns 5 values instead of 6 on this CUDA/PyTorch combo.
        try:
            from omegaconf import open_dict
            with open_dict(asr_model.cfg):
                asr_model.cfg.decoding.greedy.use_cuda_graphs = False
                asr_model.cfg.decoding.greedy.use_cuda_graph_decoder = False
            asr_model.change_decoding_strategy(asr_model.cfg.decoding)
            logger.info("CUDA graphs disabled for RNNT decoding")
        except Exception as _e:
            logger.warning("Could not disable CUDA graphs: %s", _e)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        raise


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/health")
async def health():
    """Health check for LiveKit / load balancers."""
    return {"status": "ok", "model": "stt_ar_fastconformer_hybrid_large_pcd_v1.0"}


@app.post("/transcribe")
async def transcribe(request: Request):
    """
    Transcribe audio to text.
    Accepts:
    - Raw PCM: 16kHz, mono, 16-bit signed (Content-Type: application/octet-stream)
    - WAV file: 16kHz mono (Content-Type: audio/wav or multipart/form-data)
    Returns: {"text": "...", "is_final": true}
    """
    if asr_model is None:
        load_model()

    content_type = request.headers.get("content-type", "")
    body = await request.body()

    if not body or len(body) < 1000:
        raise HTTPException(400, "Audio too short (min ~1s at 16kHz)")

    wav_path = None
    try:
        if "wav" in content_type or body[:4] == b"RIFF":
            wav_path = _to_16k_wav(body, ".wav")
        elif "mp3" in content_type or body[:3] == b"ID3" or body[:2] == b"\xff\xfb":
            wav_path = _to_16k_wav(body, ".mp3")
        else:
            wav_path = _pcm_to_wav_temp(body)

        wav_size = os.path.getsize(wav_path) if wav_path and os.path.exists(wav_path) else 0
        logger.info("WAV path=%s size=%d bytes", wav_path, wav_size)
        output = asr_model.transcribe([str(wav_path)])
        logger.info("Transcribe output type=%s len=%s first=%r", type(output).__name__, len(output) if output else 0, output[0] if output else None)
        if not output:
            text = ""
        elif isinstance(output, tuple) and len(output) >= 1:
            # (best_hypotheses, all_hypotheses) when extract_nbest
            hyps = output[0]
            first = hyps[0] if hyps else None
            if hasattr(first, "text"):
                text = first.text or ""
            elif isinstance(first, str):
                text = first
            else:
                text = str(first) if first else ""
        elif hasattr(output[0], "text"):
            text = output[0].text or ""
        elif isinstance(output[0], str):
            text = output[0]
        else:
            text = str(output[0]) if output[0] else ""
            logger.info("Raw output type: %s, repr: %r", type(output[0]), output[0])

        return JSONResponse({"text": text.strip(), "is_final": True})
    except Exception as e:
        logger.exception("Transcription error: %s", e)
        raise HTTPException(500, str(e))
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except OSError:
                pass


def _pcm_to_wav_temp(pcm_bytes: bytes) -> str:
    """Convert raw PCM 16kHz mono 16-bit to WAV file."""
    import wave
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    with wave.open(wav_path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm_bytes)
    return wav_path


def _bytes_to_wav_temp(data: bytes) -> str:
    """Write bytes to temp WAV file (if already WAV) or try to parse."""
    if data[:4] == b"RIFF":
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(data)
            return f.name
    return _pcm_to_wav_temp(data)


def _to_16k_wav(audio_bytes: bytes, suffix: str) -> str:
    """Convert any audio to 16kHz mono WAV via ffmpeg."""
    import ffmpeg
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    wav_path = tempfile.mktemp(suffix=".wav")
    try:
        stream = ffmpeg.input(tmp_path)
        stream = ffmpeg.output(
            stream, wav_path,
            acodec="pcm_s16le", ac=1, ar=SAMPLE_RATE,
            loglevel="error",
        )
        ffmpeg.run(stream, overwrite_output=True)
        return wav_path
    except ffmpeg.Error as e:
        err = (e.stderr or b"").decode(errors="replace")
        raise RuntimeError(f"FFmpeg conversion failed: {err}") from e
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    port = int(os.getenv("NEMO_STT_PORT", "3005"))
    host = os.getenv("NEMO_STT_HOST", "0.0.0.0")
    logger.info("Starting NeMo STT server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)
