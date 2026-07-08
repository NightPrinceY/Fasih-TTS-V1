# Fasih: A #1‑Intelligibility Arabic (Fusha) Voice, Fine‑tuned on 2.4 Hours and 2019 GPUs

![Fasih-TTS-V1](https://raw.githubusercontent.com/NightPrinceY/Fasih-TTS-V1/main/assets/blog_banner.png?v=2)

Modern Standard Arabic (Fusha) is spoken and understood by hundreds of millions of people, yet it
remains one of the harder languages to synthesize well — mostly because the same written word can
be pronounced several ways depending on **diacritics (tashkīl)** that are almost always omitted in
real text. In this article I walk through how I built **[Fasih‑TTS‑V1](https://huggingface.co/NightPrince/Fasih-TTS-V1)**,
a professional male Fusha voice fine‑tuned from **Coqui XTTS v2** on just **~2.4 hours** of audio and
a pair of six‑year‑old GPUs — and, more importantly, how I evaluated it **honestly** with two
independent ASR judges and a naturalness metric, so the numbers actually mean something.

> **TL;DR**
> - 🗣️ **Model:** [NightPrince/Fasih-TTS-V1](https://huggingface.co/NightPrince/Fasih-TTS-V1) — single‑speaker Arabic (MSA/Fusha) male voice.
> - 🏆 **Result:** **#1 for intelligibility** on the [SILMA open‑source Arabic TTS benchmark](https://huggingface.co/spaces/silma-ai/opensource-arabic-tts-benchmark) — lowest Word/Character Error Rate across **two** ASR judges (Whisper‑large‑v3 and NVIDIA NeMo).
> - 🧪 **Benchmark dataset (reproducible):** [NightPrince/Fasih-TTS-Benchmark](https://huggingface.co/datasets/NightPrince/Fasih-TTS-Benchmark).
> - ▶️ **Live demo (ZeroGPU):** [NightPrince/Fasih-TTS](https://huggingface.co/spaces/NightPrince/Fasih-TTS).
> - 💻 **Code + full pipeline:** [github.com/NightPrinceY/Fasih-TTS-V1](https://github.com/NightPrinceY/Fasih-TTS-V1).

---

## Why Arabic Fusha TTS is genuinely hard

Three things make Fusha a tough target:

1. **Diacritics decide pronunciation.** Written MSA drops short vowels, but broadcast‑quality Fusha
   *pronounces* full case endings (iʿrāb). The word `العلم` can be *al‑ʿilm* ("knowledge") or
   *al‑ʿalam* ("flag"). A model fed undiacritized text is guessing.
2. **The text front‑end matters as much as the acoustic model.** Numbers, abbreviations, and
   sacred/technical vocabulary all need normalization before a single sample is generated.
3. **Data is scarce.** Clean, single‑speaker, *diacritized* Fusha corpora are rare — you rarely get
   the tens of hours that English TTS enjoys.

Fasih was built for a specific product — a spoken religious‑Q&A assistant — where a **mispronounced
word is unacceptable**. That priority shaped every decision below.

## What Fasih is

| Feature | Detail |
|---|---|
| Voice | Single professional **male**, news‑anchor register |
| Language | Modern Standard Arabic (Fusha) |
| Base model | Coqui **XTTS v2** (fine‑tuned) |
| Diacritization | Built‑in **CATT** — handles even *undiacritized* input |
| Text front‑end | normalize → number expansion → tashkīl → sacred‑term lexicon → chunking |
| Output | 24 kHz mono, **RTF ≈ 0.6**, streaming first‑audio **≈ 675 ms** |
| Intelligibility | **#1** on SILMA (both ASR judges) |
| License | Coqui Public Model License (non‑commercial) |

## How I built it

### The data: 2.4 hours, fully diacritized

The training set is **1,517 single‑speaker clips (~2.9 h)**, mono, recorded for the assistant. A
full audit showed it was clean — consistent sample rate, no clipping, steady loudness — with one
real gap: **371 clips (24%) had undiacritized transcripts**. Since mixing diacritized and plain
text teaches inconsistent pronunciation, I auto‑diacritized those with **[CATT](https://github.com/abjadai/catt)**
(a SOTA encoder‑decoder Arabic diacritizer). A sanity check — strip a *gold* clip's diacritics,
re‑diacritize, compare — reproduced the human tashkīl almost character‑for‑character.

After preprocessing (24 kHz resample, silence trim, peak‑normalize) and dropping clips that exceeded
XTTS's limits, the fine‑tune set was **1,297 clips (~2.4 h)**.

### The engineering war story: Turing forces FP32

I trained on **RTX 2080 Ti (Turing, sm_75)** GPUs. This is where it got interesting:

- **BF16?** Not on Turing — no hardware support.
- **FP16 (mixed precision)?** XTTS's GPT head **overflowed and produced `NaN` loss** under FP16
  autocast, from the very first step. Eval was fine (autocast off), which made it a confusing bug.

The fix was simply to **train in FP32** — the only stable precision on Turing for this model. That
doubles memory, so I used `batch_size=1` with `grad_accum=24` and **gradient checkpointing**, and
set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to survive on 11 GB next to other jobs.
Best validation loss: **2.622**. If you ever fine‑tune XTTS on older hardware and see instant `NaN`s —
that's why.

### The Arabic text front‑end

At inference, raw assistant text flows through a small pipeline before it reaches the model:

`normalize → expand numbers → diacritize (if needed) → sacred‑term lexicon → chunk`

Two Arabic‑specific gotchas worth flagging:

- **Diacritics inflate character count.** XTTS caps Arabic input at 166 characters, and a fully
  diacritized sentence can be ~1.7× longer than its bare form — so long responses must be **chunked
  at sentence/clause boundaries** and stitched back together.
- **Number gender agreement** (`خمس` vs `خمسة`) is subtle; `num2words` gets it right most of the
  time but not always — a known limitation.

## Architecture

![Fasih architecture](https://raw.githubusercontent.com/NightPrinceY/Fasih-TTS-V1/main/assets/architecture.png?v=2)

## Does it actually work? An honest, two‑judge benchmark

This is the part I care most about. It's easy to post a single self‑reported WER and call your model
"the best." I wanted numbers I'd trust, so I evaluated on the **[SILMA open‑source Arabic TTS
benchmark](https://huggingface.co/spaces/silma-ai/opensource-arabic-tts-benchmark)** (10 fixed MSA
sentences), pulled **every competing model's audio** from the benchmark, and scored all of them —
plus Fasih — with **two independent ASRs** (Whisper‑large‑v3 and NVIDIA NeMo Arabic FastConformer),
then added **UTMOS** for naturalness.

![SILMA benchmark](https://raw.githubusercontent.com/NightPrinceY/Fasih-TTS-V1/main/assets/benchmark_msa.png?v=2)

| Model | WER · Whisper | WER · NeMo | UTMOS |
|---|:--:|:--:|:--:|
| **Fasih‑TTS‑V1 (ours)** | **6.5** | **2.5** | 3.16 |
| xtts (base) | 10.3 | 2.5 | 2.99 |
| chatterbox | 12.8 | 5.4 | 3.20 |
| silma_tts | 11.1 | 5.8 | 3.15 |
| omnivoice | 15.3 | 7.3 | 3.62 |
| habibi_specialized | 21.9 | 23.3 | 2.33 |

**What two judges reveal that one hides:**

- **Intelligibility:** Fasih has the **best‑or‑tied lowest WER on *both* ASRs** — a clear lead on
  Whisper, tied with base XTTS on the stronger Arabic ASR (NeMo). A single Whisper run would have
  *overstated* the lead; the second judge keeps it honest.
- **Naturalness:** on UTMOS, Fasih is **#3, not #1**. The smoothest‑sounding model (`omnivoice`) is
  also the *least accurate* (15.3% WER). Fasih is deliberately tuned toward **pronunciation
  correctness** — the right trade‑off for a religious assistant, but I won't pretend it's the most
  natural voice in the set.

Every per‑clip reference, ASR transcription, and score is published in the
[benchmark dataset](https://huggingface.co/datasets/NightPrince/Fasih-TTS-Benchmark) so anyone can
audit the ranking line by line. (UTMOS is English‑trained, so treat its absolute Arabic values as a
proxy — the real naturalness test is human listening.)

## Listen for yourself

Same sentences, three models. (SILMA's benchmark is itself a *human listening* comparison — so trust
your ears.)

**A classical verse — الخيل والليل والبيداء تعرفني…**

| Fasih (ours) | XTTS (base) | SILMA TTS |
|---|---|---|
| <audio controls src="https://raw.githubusercontent.com/NightPrinceY/Fasih-TTS-V1/main/assets/blog_audio/s1_fasih.mp3"></audio> | <audio controls src="https://raw.githubusercontent.com/NightPrinceY/Fasih-TTS-V1/main/assets/blog_audio/s1_xtts.mp3"></audio> | <audio controls src="https://raw.githubusercontent.com/NightPrinceY/Fasih-TTS-V1/main/assets/blog_audio/s1_silma.mp3"></audio> |

**A modern sentence — تحدث النوبة القلبية عندما يتوقف سريان الدم لجزء من القلب**

| Fasih (ours) | XTTS (base) | SILMA TTS |
|---|---|---|
| <audio controls src="https://raw.githubusercontent.com/NightPrinceY/Fasih-TTS-V1/main/assets/blog_audio/s3_fasih.mp3"></audio> | <audio controls src="https://raw.githubusercontent.com/NightPrinceY/Fasih-TTS-V1/main/assets/blog_audio/s3_xtts.mp3"></audio> | <audio controls src="https://raw.githubusercontent.com/NightPrinceY/Fasih-TTS-V1/main/assets/blog_audio/s3_silma.mp3"></audio> |

## Try it

The easiest way is the **[live ZeroGPU demo](https://huggingface.co/spaces/NightPrince/Fasih-TTS)** —
type Arabic (even *without* diacritics) and it adds tashkīl automatically before synthesizing.

Or load it directly:

```python
from huggingface_hub import snapshot_download
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

path = snapshot_download("NightPrince/Fasih-TTS-V1")
config = XttsConfig(); config.load_json(f"{path}/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=f"{path}/model.pth",
                      vocab_path=f"{path}/vocab.json", use_deepspeed=False)
model.cuda().eval()

gpt_cond, spk = model.get_conditioning_latents(audio_path=["reference.wav"])
out = model.inference("السَّلَامُ عَلَيْكُمْ وَرَحْمَةُ اللَّهِ", "ar", gpt_cond, spk,
                      temperature=0.65, repetition_penalty=2.0)
```

The full production front‑end (diacritization, numbers, chunking) and a FastAPI streaming server are
in the [code repo](https://github.com/NightPrinceY/Fasih-TTS-V1).

## Limitations

- **Naturalness is good, not best** — UTMOS #3; the model favors correctness.
- **Number gender agreement** isn't always right.
- Source audio is **128 kbps MP3** (no lossless originals) — a soft ceiling on fidelity.
- **~2.4 h, single speaker** — excellent for its domain; long‑form prosody could still improve.
- **Qur'anic recitation is out of scope** — that requires *tajwīd* and human reciters, not TTS.

## What's next

- An F5‑TTS challenger and a proper **human naturalness** comparison (via an Arabic TTS arena).
- FP16 *inference* to roughly halve the 675 ms first‑audio latency.
- A context‑aware Arabic number normalizer.

## Resources

- Model: [NightPrince/Fasih-TTS-V1](https://huggingface.co/NightPrince/Fasih-TTS-V1)
- Benchmark dataset: [NightPrince/Fasih-TTS-Benchmark](https://huggingface.co/datasets/NightPrince/Fasih-TTS-Benchmark)
- Live demo: [NightPrince/Fasih-TTS](https://huggingface.co/spaces/NightPrince/Fasih-TTS)
- Code + full training/eval pipeline: [github.com/NightPrinceY/Fasih-TTS-V1](https://github.com/NightPrinceY/Fasih-TTS-V1)

## Acknowledgments

Built on **[Coqui XTTS v2](https://huggingface.co/coqui/XTTS-v2)**; diacritization by
**[CATT](https://github.com/abjadai/catt)**; benchmark sentences and competitor audio from
**[SILMA AI](https://huggingface.co/spaces/silma-ai/opensource-arabic-tts-benchmark)**; ASR judges
Whisper‑large‑v3 and NVIDIA NeMo FastConformer.

*By Yahya Elnawasany ([NightPrince](https://huggingface.co/NightPrince)) — [portfolio](https://nightprincey.github.io/Portfolio-App/). Fasih is non‑commercial (CPML).*
