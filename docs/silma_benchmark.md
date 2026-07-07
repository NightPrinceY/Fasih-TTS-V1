# SILMA Open-Source Arabic TTS Benchmark — Fasih-TTS-V1

Evaluation of Fasih-TTS-V1 on the
[SILMA Open-Source Arabic TTS Benchmark](https://huggingface.co/spaces/silma-ai/opensource-arabic-tts-benchmark)
(MSA split — 10 fixed sentences).

## Methodology

SILMA's benchmark is a **human auditory** comparison; they state that quantitative metrics
(WER, CER, SIM, UTMOS) are "often insufficient for accurately capturing the nuances of Arabic
speech." We therefore treat the numbers below as a **supplementary objective intelligibility
measure**, and also publish Fasih's **audio** on the same 10 sentences for the intended
listening comparison.

- **Sentences:** the 10 MSA sentences from `results/msa/Ar_msa_TTS_benchmark.csv` in the Space.
- **Competitor audio:** downloaded directly from the Space (chatterbox, habibi_specialized,
  omnivoice, silma_tts, xtts).
- **Fasih audio:** synthesized with the full production front-end (auto-diacritization via CATT +
  number expansion + chunking) — `scripts/silma_benchmark.py`.
- **Scoring:** each model's audio transcribed with **Whisper-large-v3**; WER/CER computed against
  the reference text (both diacritics-stripped, orthography-normalized, digits expanded to words)
  — `scripts/silma_compare.py`.

## Results (objective, WER/CER — lower is better)

| Rank | Model | WER % | CER % |
|:--:|--|:--:|:--:|
| **1** | **Fasih-TTS-V1 (ours)** | **6.5** | **2.0** |
| 2 | XTTS (base) | 10.3 | 6.4 |
| 3 | silma_tts | 11.1 | 6.0 |
| 4 | chatterbox | 12.8 | 6.6 |
| 5 | omnivoice | 15.3 | 7.2 |
| 6 | habibi_specialized | 21.9 | 9.7 |

**Fasih-TTS-V1 ranks #1 by intelligibility**, ahead of base XTTS and SILMA's own model. Its full
Arabic front-end (diacritization + normalization) is a direct contributor to the low error rate.

![chart](../assets/benchmark_msa.png)

## Caveats
- Intelligibility ≠ naturalness. The naturalness ranking requires the human listening comparison;
  Fasih's audio is provided (`assets/benchmark/msa/`) for that.
- One ASR pass, single reference; Whisper itself errs on Arabic (~1–2% CER floor on clean speech).
- Sentence 7 is a Qur'anic verse read as plain MSA text for benchmark comparability — **not**
  tajwīd recitation, which remains out of scope for the model.
