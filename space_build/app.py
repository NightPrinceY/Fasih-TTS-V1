"""Fasih-TTS-V1 — live Arabic (MSA/Fusha) TTS demo on Hugging Face ZeroGPU."""

import os
import sys
import urllib.request
from pathlib import Path

import gradio as gr
import numpy as np
import spaces
import torch
from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_REPO = "NightPrince/Fasih-TTS-V1"
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

try:
    from tts.text.diacritize import Diacritizer
    _diac = Diacritizer(ckpt=CATT_CKPT, device="cpu")
    pipe = TextPipeline(diacritizer=_diac)
    DIAC_OK = True
except Exception as e:  # noqa: BLE001
    print("diacritizer unavailable:", e)
    pipe, DIAC_OK = TextPipeline(diacritizer=None), False


@spaces.GPU(duration=120)
def synthesize(text: str, auto_diacritize: bool = True, temperature: float = 0.65):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Please enter some Arabic text.")
    if auto_diacritize and DIAC_OK:
        chunks = pipe.prepare_chunks(text)
    else:
        chunks = chunk_text(normalize(text), 160)

    m = model.to("cuda")
    gpt, spk = GPT_COND.to("cuda"), SPK.to("cuda")
    gap = np.zeros(int(SR * 0.12), dtype=np.float32)
    pieces = []
    for i, ch in enumerate(chunks):
        out = m.inference(ch, "ar", gpt, spk, temperature=float(temperature),
                          repetition_penalty=2.0, enable_text_splitting=False)
        pieces.append(np.asarray(out["wav"], dtype=np.float32))
        if i < len(chunks) - 1:
            pieces.append(gap)
    wav = np.concatenate(pieces) if pieces else np.zeros(1, np.float32)
    return SR, wav


DESC = """
**Fasih** (فَصِيح) — a professional male **Modern Standard Arabic (Fusha)** voice, fine-tuned from
Coqui XTTS v2. Type Arabic (even without diacritics — it auto-adds tashkīl via CATT) and hear it.
Ranked #1 for intelligibility on the SILMA open-source Arabic TTS benchmark.
[Model](https://huggingface.co/NightPrince/Fasih-TTS-V1) ·
[Benchmark](https://huggingface.co/datasets/NightPrince/Fasih-TTS-Benchmark) ·
[Code](https://github.com/NightPrinceY/Fasih-TTS-V1)
"""

demo = gr.Interface(
    fn=synthesize,
    inputs=[
        gr.Textbox(label="النص العربي — Arabic text", lines=3, rtl=True,
                   value="السلام عليكم ورحمة الله وبركاته، كيف يمكنني مساعدتك اليوم؟"),
        gr.Checkbox(label="تشكيل تلقائي — Auto-diacritize (CATT)", value=True),
        gr.Slider(0.3, 1.0, value=0.65, step=0.05, label="Temperature"),
    ],
    outputs=gr.Audio(label="Fasih output", type="numpy"),
    title="🕌 Fasih-TTS-V1 — Arabic (Fusha) Professional Male TTS",
    description=DESC,
    examples=[
        ["الصلوات المفروضة خمس في اليوم والليلة، وهي عمود الدين.", True, 0.65],
        ["الوضوء شرط لصحة الصلاة، ويبدأ بالنية ثم غسل الوجه واليدين.", True, 0.65],
        ["بارك الله فيك، وجعل يومك مليئا بالخير والبركة.", True, 0.6],
    ],
    cache_examples=False,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
