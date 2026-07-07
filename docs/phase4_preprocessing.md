# Phase 4 — Audio Preprocessing & Manifests

**Input:** `data/interim/metadata_diacritized.csv` (1517 fully-diacritized rows) + raw MP3s.
**Output:** `data/processed/wav24/*.wav` + `data/manifests/{train,val}.csv`.

## Processing (light-touch — source is lossy 128 kbps MP3)
1. Resample 44.1 kHz → **24 kHz** mono (F5 native; XTTS resamples to 22.05k downstream).
2. Energy-based silence trim (`top_db=30`) + 50 ms padding.
3. Peak-normalize to **−1 dBFS** (no loudness pumping that would raise the noise floor).
4. Write 16-bit PCM WAV.

Intentionally avoided denoise/EQ/compression that amplify MP3 artifacts.

## Result
- Kept **1517 / 1517** clips, **2.83 h** total; 0 dropped; 8 flagged `too_long` (>15 s, kept).
- **train 1442 (2.69 h)** / **val 75** — split stratified by `diac_source` (val = 57 gold + 18 catt).

## Manifest schema (`data/manifests/{train,val}.csv`)
`file, text, diac_source, duration, flags`
- `file` → relative to `data/processed/wav24/`
- `text` → normalized + fully diacritized
- framework-specific manifests (XTTS LJSpeech-style, F5) are generated from these in Phases 5-6.
