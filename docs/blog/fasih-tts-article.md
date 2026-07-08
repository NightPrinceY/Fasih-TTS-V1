# Introducing Fasih-TTS-V1: a #1-Intelligibility Arabic (Fusha) Voice — and an Open Benchmark to Back It Up

![Fasih-TTS-V1](https://raw.githubusercontent.com/NightPrinceY/Fasih-TTS-V1/main/assets/blog_banner.png?v=2)

Arabic is spoken by more than 400 million people, yet Modern Standard Arabic (Fusha) is still poorly
served by open text-to-speech — and, just as importantly, **there is almost no rigorous, reproducible
way to *evaluate* Arabic TTS**. The team behind the excellent [SILMA benchmark](https://huggingface.co/spaces/silma-ai/opensource-arabic-tts-benchmark)
said it plainly: standard metrics "are often insufficient for accurately capturing the nuances of
Arabic speech," so they fell back to human listening.

Today I'm releasing **[Fasih-TTS-V1](https://huggingface.co/NightPrince/Fasih-TTS-V1)** (فَصِيح,
*"eloquent"*) — a professional male Fusha voice fine-tuned from Coqui XTTS v2 — **together with the
evaluation infrastructure the ecosystem was missing**: an open, reproducible, two-ASR-judge benchmark
and a fully-scored dataset. Fasih ranks **#1 for intelligibility** on the SILMA benchmark across
*both* judges — and I'm publishing every clip, transcript, and score so anyone can check the claim
line by line.

## What I'm releasing

This is a complete, open release — not just weights:

- **The model** — [NightPrince/Fasih-TTS-V1](https://huggingface.co/NightPrince/Fasih-TTS-V1): a single-speaker Arabic (MSA/Fusha) male voice with a built-in diacritization + text front-end.
- **An open benchmark dataset** — [NightPrince/Fasih-TTS-Benchmark](https://huggingface.co/datasets/NightPrince/Fasih-TTS-Benchmark): 31 scored clips plus the full 6-model SILMA comparison, with **per-clip references, ASR transcriptions, and WER/CER** for total transparency.
- **A live demo** — [NightPrince/Fasih-TTS](https://huggingface.co/spaces/NightPrince/Fasih-TTS) on ZeroGPU: type Arabic (even without diacritics) and hear it.
- **The full pipeline, open** — [github.com/NightPrinceY/Fasih-TTS-V1](https://github.com/NightPrinceY/Fasih-TTS-V1): data validation, CATT diacritization, training, a two-judge evaluation harness, a FastAPI streaming server, and a deployable NVIDIA NeMo Arabic STT service.

## Table of contents

- [Why this matters](#why-this-matters)
- [What Fasih is](#what-fasih-is)
- [How I built it](#how-i-built-it)
- [Architecture](#architecture)
- [An honest, two-judge benchmark](#an-honest-two-judge-benchmark)
- [Listen for yourself](#listen-for-yourself)
- [Get started](#get-started)
- [Limitations](#limitations)
- [What's next, and how to contribute](#whats-next-and-how-to-contribute)
- [License and citation](#license-and-citation)
- [Acknowledgments](#acknowledgments)

## Why this matters

Three things make Fusha genuinely hard, and each shaped the release:

1. **Diacritics decide pronunciation.** Written MSA drops short vowels, but broadcast-quality Fusha
   *pronounces* full case endings (iʿrāb). `العلم` is *al-ʿilm* ("knowledge") or *al-ʿalam* ("flag")
   depending on marks that real text omits. A model fed raw text is guessing.
2. **The text front-end matters as much as the acoustic model.** Numbers, abbreviations, and
   sacred/technical vocabulary must be normalized before a single sample is generated.
3. **Evaluation is unsolved in the open.** Without a shared, reproducible way to measure Arabic TTS,
   "state of the art" claims can't be checked — which is exactly the gap this release targets.

Fasih was built for a real product — a spoken religious-Q&A assistant — where a **mispronounced word
is unacceptable**. That priority pointed the whole project at *correctness first*, and made honest
measurement non-negotiable.

## What Fasih is

| | |
|---|---|
| Voice | Single professional **male**, news-anchor register |
| Language | Modern Standard Arabic (Fusha) |
| Base model | Coqui **XTTS v2** (fine-tuned) |
| Diacritization | Built-in **CATT** — handles even *undiacritized* input |
| Front-end | normalize → number expansion → tashkīl → sacred-term lexicon → chunking |
| Output | 24 kHz mono · **RTF ≈ 0.6** · streaming first-audio **≈ 675 ms** |
| Intelligibility | **#1** on the SILMA benchmark (both ASR judges) |
| License | Coqui Public Model License (non-commercial) |

## How I built it

### Data: 2.4 hours, fully diacritized

The training set is **1,517 single-speaker clips (~2.9 h)**, mono. A full audit showed clean audio —
consistent sample rate, no clipping, steady loudness — with one real gap: **371 clips (24%) had
undiacritized transcripts**. Because mixing diacritized and plain text teaches inconsistent
pronunciation, I auto-diacritized those with **[CATT](https://github.com/abjadai/catt)**, a SOTA
encoder-decoder Arabic diacritizer. A sanity check — strip a *gold* clip's diacritics, re-diacritize,
compare — reproduced the human tashkīl almost character-for-character. After preprocessing and
dropping over-long clips, the fine-tune set was **1,297 clips (~2.4 h)**.

### The Arabic text front-end

At inference, raw text flows through a small pipeline before it reaches the model:

`normalize → expand numbers → diacritize (if needed) → sacred-term lexicon → chunk`

Two Arabic-specific gotchas worth sharing:

- **Diacritics inflate character count.** XTTS caps Arabic input at 166 characters, and a fully
  diacritized sentence can be ~1.7× longer than its bare form — so long responses must be **chunked**
  at sentence/clause boundaries and stitched back together.
- **Number gender agreement** (`خمس` vs `خمسة`) is subtle; `num2words` is right most of the time,
  not always.

### The training journey: Turing forces FP32

I trained on **RTX 2080 Ti (Turing, sm_75)** GPUs — and hit a wall worth documenting:

- **BF16?** Not supported on Turing.
- **FP16 (mixed precision)?** XTTS's GPT head **overflowed and produced `NaN` loss** under FP16
  autocast, from the very first step, while eval (autocast off) stayed finite — a genuinely confusing bug.

The fix was to train in **FP32**, the only stable precision here. That doubles memory, so I used
`batch_size=1` with `grad_accum=24`, **gradient checkpointing**, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to fit 11 GB alongside other jobs. Best validation
loss: **2.622**. If you ever fine-tune XTTS on older hardware and see instant `NaN`s — now you know why.

## Architecture

![Fasih architecture](https://raw.githubusercontent.com/NightPrinceY/Fasih-TTS-V1/main/assets/architecture.png?v=2)

## An honest, two-judge benchmark

This is the contribution I'm proudest of. It is easy to post a single self-reported WER and declare
victory. I wanted numbers I would trust — and that the community can reproduce — so I:

1. took the **10 fixed MSA sentences** from the SILMA benchmark,
2. pulled **every competing model's audio** straight from the benchmark Space,
3. synthesized Fasih on the same sentences with its full front-end, and
4. scored all six models with **two independent ASRs** — Whisper-large-v3 **and** NVIDIA NeMo Arabic
   FastConformer — then added **UTMOS** for naturalness.

![SILMA benchmark](https://raw.githubusercontent.com/NightPrinceY/Fasih-TTS-V1/main/assets/benchmark_msa.png?v=2)

| Model | WER · Whisper | WER · NeMo | UTMOS |
|---|:--:|:--:|:--:|
| **Fasih-TTS-V1 (ours)** | **6.5** | **2.5** | 3.16 |
| xtts (base) | 10.3 | 2.5 | 2.99 |
| chatterbox | 12.8 | 5.4 | 3.20 |
| silma_tts | 11.1 | 5.8 | 3.15 |
| omnivoice | 15.3 | 7.3 | 3.62 |
| habibi_specialized | 21.9 | 23.3 | 2.33 |

**What using two judges reveals — and one wouldn't:**

- **Intelligibility:** Fasih has the **best-or-tied lowest WER on *both* ASRs** — a clear lead on
  Whisper, tied with base XTTS on the stronger Arabic ASR (NeMo). A single Whisper run would have
  *overstated* the lead; the second judge keeps it honest.
- **Naturalness:** on UTMOS, Fasih is **#3, not #1**. The smoothest-sounding model (`omnivoice`) is
  also the *least accurate* (15.3% WER). Fasih is deliberately tuned toward **pronunciation
  correctness** — the right trade-off for a religious assistant, and I won't pretend otherwise.

Every per-clip reference, transcription, and score is in the
[benchmark dataset](https://huggingface.co/datasets/NightPrince/Fasih-TTS-Benchmark), and the whole
thing reruns from two scripts. That reproducibility — not the ranking itself — is what I hope is
useful to the next person building Arabic TTS. (UTMOS is English-trained, so treat its absolute
Arabic values as a proxy; the real naturalness test remains human listening.)

## Listen for yourself

Same sentences, three models — trust your ears.

**A classical verse — *الخيل والليل والبيداء تعرفني*…**

| Fasih (ours) | XTTS (base) | SILMA TTS |
|---|---|---|
| <audio controls src="https://raw.githubusercontent.com/NightPrinceY/Fasih-TTS-V1/main/assets/blog_audio/s1_fasih.mp3"></audio> | <audio controls src="https://raw.githubusercontent.com/NightPrinceY/Fasih-TTS-V1/main/assets/blog_audio/s1_xtts.mp3"></audio> | <audio controls src="https://raw.githubusercontent.com/NightPrinceY/Fasih-TTS-V1/main/assets/blog_audio/s1_silma.mp3"></audio> |

**A modern sentence — *تحدث النوبة القلبية عندما يتوقف سريان الدم لجزء من القلب***

| Fasih (ours) | XTTS (base) | SILMA TTS |
|---|---|---|
| <audio controls src="https://raw.githubusercontent.com/NightPrinceY/Fasih-TTS-V1/main/assets/blog_audio/s3_fasih.mp3"></audio> | <audio controls src="https://raw.githubusercontent.com/NightPrinceY/Fasih-TTS-V1/main/assets/blog_audio/s3_xtts.mp3"></audio> | <audio controls src="https://raw.githubusercontent.com/NightPrinceY/Fasih-TTS-V1/main/assets/blog_audio/s3_silma.mp3"></audio> |

## Get started

The fastest path is the **[live ZeroGPU demo](https://huggingface.co/spaces/NightPrince/Fasih-TTS)** —
type Arabic (even without diacritics) and it adds tashkīl automatically. Or load it directly:

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

The full production front-end and a FastAPI streaming server are in the
[repository](https://github.com/NightPrinceY/Fasih-TTS-V1).

## Limitations

- **Naturalness is good, not best** — UTMOS #3; the model favors correctness over smoothness.
- **Number gender agreement** isn't always right.
- Source audio is **128 kbps MP3** (no lossless originals) — a soft ceiling on fidelity.
- **~2.4 h, single speaker** — excellent for its domain; long-form prosody can still improve.
- **Qur'anic recitation is out of scope** — it requires *tajwīd* and human reciters, not TTS.

## What's next, and how to contribute

Arabic TTS gets better faster when the evaluation is open. If you build Arabic voices, please:

- **Run the harness on your model** — the two-judge scripts and dataset make an apples-to-apples
  comparison a few commands away.
- **Extend the benchmark** — more sentences, dialects, or a third ASR judge are all welcome.
- **Help with naturalness** — a human Arabic listening study is the missing piece UTMOS only proxies.

On my side: an F5-TTS challenger, FP16 *inference* to roughly halve the 675 ms latency, and a
context-aware Arabic number normalizer.

## License and citation

Fasih is fine-tuned from Coqui XTTS v2 and distributed under the **Coqui Public Model License** —
non-commercial, attribution required; derivatives inherit these terms.

```bibtex
@software{fasih_tts_v1_2026,
  author = {Yahya Elnawasany (NightPrince)},
  title  = {Fasih-TTS-V1: Arabic Fusha Professional-Male Text-to-Speech},
  year   = {2026},
  url    = {https://github.com/NightPrinceY/Fasih-TTS-V1},
  note   = {Fine-tuned from Coqui XTTS v2}
}
```

## Acknowledgments

Built on **[Coqui XTTS v2](https://huggingface.co/coqui/XTTS-v2)**; diacritization by
**[CATT](https://github.com/abjadai/catt)**; benchmark sentences and competitor audio from
**[SILMA AI](https://huggingface.co/spaces/silma-ai/opensource-arabic-tts-benchmark)**; ASR judges
Whisper-large-v3 and NVIDIA NeMo FastConformer; hosting and ZeroGPU by Hugging Face.

*By Yahya Elnawasany ([NightPrince](https://huggingface.co/NightPrince)) — [portfolio](https://nightprincey.github.io/Portfolio-App/). If Fasih or the benchmark helps your work, a star on the [repo](https://github.com/NightPrinceY/Fasih-TTS-V1) and a note on what you built would mean a lot.*
