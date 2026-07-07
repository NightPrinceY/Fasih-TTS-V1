# Phase 3 — Arabic Text Pipeline

**Goal:** make every training transcript correctly diacritized + normalized, and provide a
reusable inference front-end.

## Components (`src/tts/text/`)
- `normalize.py` — diacritization-preserving normalizer (NFC, strip tatweel/invisibles,
  collapse whitespace). Never strips harakat or hamza forms (they change pronunciation).
- `diacritics.py` — tashkīl coverage analysis + `needs_diacritization` gate.
- `diacritize.py` — **CATT** encoder-decoder wrapper (vendored `catt/`, MIT, user-authorized),
  **punctuation-preserving** via full-sentence diacritization + word-level re-insertion.
- `pipeline.py` — `TextPipeline`: `prepare_training_text` (normalize only) and
  `prepare_inference_text` (normalize → diacritize-if-needed → sacred-term lexicon).
- `catt/` — vendored CATT inference code + `LICENSE`.

## What we did
- Verified CATT quality: re-diacritizing stripped gold clips reproduces the human
  diacritization almost character-identical.
- Diacritized the **371** plain clips (1147–1517). Result: **1517/1517 fully diacritized**
  (1146 gold + 371 CATT).
- Outputs:
  - `data/interim/metadata_diacritized.csv` — training-ready (1517 rows, `diac_source` col).
  - `data/manifests/diacritization_review.csv` — the 371 in/out pairs for human review.

## Known limitation (flagged)
Automatic diacritization is ~95%+ but makes occasional case/voice errors (observed
`تنتظر` → passive where audio is active). **Recommend a human review pass** over the 371
before the final training run; the review CSV is ready for it.

## Deferred to inference (Phase 8)
- Number→Arabic-words expansion (Dataset A has no digits; production text does).
- Abbreviation expansion; sacred-term lexicon expansion by a domain reviewer
  (`configs/data/lexicon_ar.yaml` seeded).

## Tests
`tests/test_text.py` — 7 golden tests (normalize, strip, coverage, gate). All pass.
