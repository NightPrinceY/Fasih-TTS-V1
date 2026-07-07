# Phase 7 — Evaluation & Phase 8 — Production Serving

## Phase 7 — Objective evaluation (CER via Whisper large-v3)

CER = character error rate between the ASR transcript of the synthesized audio and the
intended text (both diacritics-stripped + orthography-normalized). Judged against the ASR
floor measured on the original human clips.

| Set | Mean CER | Worst | Notes |
|-----|----------|-------|-------|
| TTS varied (8) | 1.3% | 2.2% | — |
| TTS same sentence ×4 | 2.0% | 2.0% | zero run-to-run variance |
| TTS long (chunked) | ~0.8% | 0.9% | chunking seamless |
| TTS hard stress (6) | 2.1% | 8.2% | worst = 5-item list |
| Human originals (floor) | 1.8% | 4.8% | reference |

**Verdict:** TTS intelligibility ≈ human, with no XTTS instability (no loops/cutoffs/
hallucinations) across 24 generations. Latency RTF ~0.6; streaming time-to-first-audio ~675 ms.

Scripts: `scripts/evaluate_cer.py` (`--sentences`, `--audio_dir`, `--no_human`),
`scripts/synth_samples.py`. Sample sets: `configs/eval_sentences_ar.txt`,
`configs/stress_sentences_ar.txt`.

**Not covered by CER (still needs a human ear):** naturalness/prosody (UTMOS optional) and
diacritic/iʿrāb correctness. Listen to `outputs/stress_test/05.wav` (the pillars list).

## Phase 8 — Production package

Full front-end + engine + API. Raw agent text → correct diacritized, number-expanded,
chunked speech.

- `src/tts/text/chunk.py` — split long text ≤160 chars at sentence/clause boundaries.
- `src/tts/text/numbers.py` — digits → Arabic words (num2words). *Known gap:* gender
  agreement (`خمسة` vs `خمس`).
- `src/tts/text/pipeline.py` — `prepare_chunks()` = normalize → numbers → diacritize-if-needed
  → sacred-term lexicon → chunk.
- `src/tts/infer/engine.py` — `XttsEngine`: loads model once, caches target-voice latents,
  `synthesize()` and `stream()`.
- `scripts/serve.py` — FastAPI: `/health`, `/tts` (WAV), `/tts/stream` (PCM16 24 kHz).
- `scripts/say.py` — CLI.

### Run
```bash
CUDA_VISIBLE_DEVICES=1 uv run uvicorn scripts.serve:app --host 0.0.0.0 --port 8000
curl -X POST localhost:8000/tts -H 'Content-Type: application/json' \
     -d '{"text":"بارك الله فيك"}' --output out.wav
```

### Open items / future
- FP16 inference to cut the 675 ms first-audio latency (~halve it).
- Arabic number gender-agreement normalizer.
- Optional F5-TTS challenger bake-off; UTMOS naturalness score.
- Human review of `data/manifests/diacritization_review.csv` (371 CATT-diacritized clips).
