"""Fasih-TTS-V1 — live Arabic (MSA/Fusha) TTS demo on Hugging Face ZeroGPU."""

import io
import json
import os
import sys
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import numpy as np
import spaces
import torch
from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_REPO = "NightPrince/Fasih-TTS-V1"
CAPTURE_REPO = "NightPrince/fasih-space-captures"
CATT_URL = "https://github.com/abjadai/catt/releases/download/v2/best_ed_mlm_ns_epoch_178.pt"
SR = 24000

# --- download model (weights + config + vocab + precomputed speaker latents) ---
model_dir = snapshot_download(
    MODEL_REPO, allow_patterns=["model.pth", "config.json", "vocab.json", "speaker_latents.pt"]
)

# --- CATT diacritizer checkpoint ---
Path("models/catt").mkdir(parents=True, exist_ok=True)
CATT_CKPT = "models/catt/best_ed_mlm_ns_epoch_178.pt"
if not os.path.exists(CATT_CKPT):
    try:
        urllib.request.urlretrieve(CATT_URL, CATT_CKPT)
    except Exception as e:  # noqa: BLE001
        print("CATT download failed:", e)

# --- load XTTS (CPU; moved to GPU inside the @spaces.GPU call) ---
from TTS.tts.configs.xtts_config import XttsConfig  # noqa: E402
from TTS.tts.models.xtts import Xtts  # noqa: E402

config = XttsConfig()
config.load_json(f"{model_dir}/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=f"{model_dir}/model.pth",
                      vocab_path=f"{model_dir}/vocab.json", use_deepspeed=False)
model.eval()

_lat = torch.load(f"{model_dir}/speaker_latents.pt", map_location="cpu")
GPT_COND, SPK = _lat["gpt_cond_latent"], _lat["speaker_embedding"]

# --- Arabic text front-end (normalize -> numbers -> diacritize -> lexicon -> chunk) ---
from tts.text.chunk import chunk_text  # noqa: E402
from tts.text.normalize import normalize  # noqa: E402
from tts.text.pipeline import TextPipeline  # noqa: E402
import ui  # noqa: E402

try:
    from tts.text.diacritize import Diacritizer
    _diac = Diacritizer(ckpt=CATT_CKPT, device="cpu")
    pipe = TextPipeline(diacritizer=_diac)
    DIAC_OK = True
except Exception as e:  # noqa: BLE001
    print("diacritizer unavailable:", e)
    pipe, DIAC_OK = TextPipeline(diacritizer=None), False


def _log_capture(text: str, auto_diacritize: bool, temperature: float,
                  wav: np.ndarray, latency_s: float) -> None:
    """Best-effort: save the (text, audio, metadata) triple to a private dataset.

    Never allowed to break the user-facing request — any failure here is
    logged server-side and swallowed.
    """
    if not os.environ.get("SPACE_ID"):
        return
    try:
        import soundfile as sf
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()  # picks up HF_TOKEN from the Space's secret automatically
        now = datetime.now(timezone.utc)
        uid = uuid.uuid4().hex[:8]
        audio_path = f"data/{now:%Y-%m-%d}/{now:%H%M%S}_{uid}.wav"

        buf = io.BytesIO()
        sf.write(buf, wav, SR, format="WAV")
        buf.seek(0)
        api.upload_file(path_or_fileobj=buf, path_in_repo=audio_path,
                        repo_id=CAPTURE_REPO, repo_type="dataset")

        row = {
            "file_name": audio_path,
            "text": text,
            "auto_diacritize": bool(auto_diacritize),
            "temperature": float(temperature),
            "duration_seconds": round(len(wav) / SR, 3),
            "sample_rate": SR,
            "latency_seconds": round(latency_s, 3),
            # Full human-readable timestamp, e.g. "Friday, July 10, 2026 at 03:45:12 PM UTC"
            "timestamp_human": now.strftime("%A, %B %d, %Y at %I:%M:%S %p UTC"),
            "timestamp_iso": now.isoformat(),
        }

        try:
            meta_path = hf_hub_download(CAPTURE_REPO, "metadata.jsonl", repo_type="dataset")
            existing = Path(meta_path).read_text(encoding="utf-8")
        except Exception:
            existing = ""

        updated = existing + json.dumps(row, ensure_ascii=False) + "\n"
        api.upload_file(
            path_or_fileobj=io.BytesIO(updated.encode("utf-8")),
            path_in_repo="metadata.jsonl",
            repo_id=CAPTURE_REPO,
            repo_type="dataset",
        )
    except Exception as e:  # noqa: BLE001
        print("capture logging failed (non-fatal):", e)


# Chunks per GPU call. Long text is spoken over several calls, so length is unlimited.
CHUNKS_PER_CALL = 10

EXAMPLES = [
    ("تحية", "السلام عليكم ورحمة الله وبركاته، أنا مسلم، مساعدك الصوتي. كيف يمكنني مساعدتك اليوم؟"),
    ("أرقام", "أركان الإسلام 5، وأركان الإيمان 6."),
    ("فقه", "الوضوء شرط لصحة الصلاة، ويبدأ بالنية ثم غسل الوجه واليدين إلى المرفقين."),
    ("نص طويل", "العلم نور يهدي صاحبه إلى الحق، ويرفع قدره بين الناس. وقد حث الإسلام على طلب العلم، "
                "فجعله فريضة على كل مسلم. ومن سلك طريقا يلتمس فيه علما، سهل الله له به طريقا إلى الجنة."),
]


def _prepare(text: str, auto_diacritize: bool) -> list[str]:
    """Arabic front-end on CPU, outside the GPU allocation."""
    text = (text or "").strip()
    if not text:
        raise gr.Error("اكتب نصًا عربيًا أولًا. Enter some Arabic text first.")
    if auto_diacritize and DIAC_OK:
        # CATT reads at most 1024 characters at once, so diacritize long text piece by piece.
        # 600 leaves room for numbers growing into words.
        return [c for piece in chunk_text(normalize(text), 600) for c in pipe.prepare_chunks(piece)]
    return chunk_text(normalize(text), 160)


def _gpu_seconds(chunks: list[str], temperature: float) -> int:
    return 15 + 6 * len(chunks)


@spaces.GPU(duration=_gpu_seconds)
def _generate_group(chunks: list[str], temperature: float) -> list[np.ndarray]:
    m = model.to("cuda")
    gpt, spk = GPT_COND.to("cuda"), SPK.to("cuda")
    wavs = []
    for ch in chunks:
        out = m.inference(ch, "ar", gpt, spk, temperature=float(temperature),
                          repetition_penalty=2.0, enable_text_splitting=False)
        wavs.append(np.asarray(out["wav"], dtype=np.float32))
    return wavs


def _generate(chunks: list[str], temperature: float) -> np.ndarray:
    """Speak any number of chunks, one GPU call per CHUNKS_PER_CALL chunks."""
    wavs = []
    for i in range(0, len(chunks), CHUNKS_PER_CALL):
        wavs += _generate_group(chunks[i:i + CHUNKS_PER_CALL], temperature)
    gap = np.zeros(int(SR * 0.12), dtype=np.float32)
    pieces = []
    for i, w in enumerate(wavs):
        pieces.append(w)
        if i < len(wavs) - 1:
            pieces.append(gap)
    return np.concatenate(pieces) if pieces else np.zeros(1, np.float32)


def synthesize(text: str, auto_diacritize: bool = True, temperature: float = 0.65):
    """Speak Modern Standard Arabic (Fusha) text in the Fasih professional male voice.

    Args:
        text: Arabic text, with or without diacritics. Numbers are read as words.
        auto_diacritize: add tashkil with the CATT diacritizer before speaking.
        temperature: sampling temperature, 0.3 to 1.0. Lower is steadier.

    Returns:
        24 kHz mono audio.
    """
    t0 = time.time()
    wav = _generate(_prepare(text, auto_diacritize), temperature)
    _log_capture(text, auto_diacritize, temperature, wav, time.time() - t0)
    return SR, wav


def speak(text: str, auto_diacritize: bool, temperature: float):
    t0 = time.time()
    chunks = _prepare(text, auto_diacritize)
    wav = _generate(chunks, temperature)
    elapsed = time.time() - t0
    _log_capture(text, auto_diacritize, temperature, wav, elapsed)
    return (SR, wav), ui.stats_html(len(wav) / SR, elapsed, len(chunks)), ui.read_html(chunks)


def count(text: str):
    return ui.counter_html(len(text or ""))


with gr.Blocks(title="Fasih-TTS-V1 · Arabic Fusha text to speech") as demo:
    gr.HTML(ui.HERO)
    with gr.Row(elem_id="studio", equal_height=False):
        with gr.Column(scale=6):
            text = gr.Textbox(value=EXAMPLES[0][1], lines=5, max_lines=10, rtl=True, show_label=False,
                              placeholder="اكتب نصًا بالعربية الفصحى…", elem_id="text-in")
            counter = gr.HTML(count(EXAMPLES[0][1]))
            with gr.Row(elem_id="chips"):
                chips = [(gr.Button(label, size="sm", variant="secondary"), value) for label, value in EXAMPLES]
            btn = gr.Button("انطق", variant="primary", elem_id="speak-btn")
            with gr.Accordion("إعدادات · Settings", open=False):
                auto_diac = gr.Checkbox(value=True, label="تشكيل تلقائي (CATT) · Auto-diacritize")
                temperature = gr.Slider(0.3, 1.0, value=0.65, step=0.05,
                                        label="درجة التنوع · Temperature (lower is steadier)")
        with gr.Column(scale=5):
            audio = gr.Audio(label="صوت فصيح · Fasih", type="numpy", interactive=False, autoplay=True,
                             buttons=["download"], elem_id="audio-out")
            stats = gr.HTML(ui.STATS_EMPTY)
            read = gr.HTML(ui.READ_EMPTY)
    gr.HTML(ui.EVIDENCE)
    gr.HTML(ui.BOUNDARY)
    gr.HTML(ui.FOOTER)

    # Counted in the browser: per-keystroke server calls can return out of order.
    text.change(None, text, counter, js=ui.COUNTER_JS, queue=False, show_progress="hidden",
                api_visibility="private")
    for chip, value in chips:
        chip.click(lambda v=value: v, None, text, queue=False, show_progress="hidden",
                   api_visibility="private")
    btn.click(speak, [text, auto_diac, temperature], [audio, stats, read], api_visibility="private")
    # Stable API for gradio_client and MCP callers: same name, inputs and audio output as before.
    gr.Button(visible=False).click(synthesize, [text, auto_diac, temperature], audio, api_name="predict")


if __name__ == "__main__":
    demo.queue(max_size=20).launch(mcp_server=True, theme=ui.THEME, css=ui.CSS, head=ui.HEAD, js=ui.FORCE_DARK)
