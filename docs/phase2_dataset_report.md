# Phase 2 — Dataset A Validation Report

**Date:** 2026-07-06 · **Source:** `NightPrince/Arabic-professional-original-voice`
**Manifest:** `data/manifests/dataset_a_manifest.csv` (per-clip QC + diacritization)

## Summary
Clean, consistent, single-speaker professional Fusha corpus. One substantive fix required
(diacritize 371 transcripts). No audio-quality blockers.

| Metric | Value |
|--------|-------|
| Clips | 1517 (all unique transcripts) |
| Total duration | 2.87 h (median 6.46s, range 1.60–18.08s) |
| Sample rate | 44.1 kHz (100% consistent) |
| Channels | mono |
| Codec | MP3 128 kbps (lossy — quality ceiling; no lossless originals) |
| Peak amplitude (max) | 0.984 → **no clipping** |
| Loudness | RMS median −16.2 dBFS (consistent) |
| Clean clips (no flags) | 1501 / 1517 |

## Flags (16 clips)
- `too_long` (>15s): 9 clips (max 18.08s) — keep (XTTS/F5 handle) or split in Phase 4.
- `trail_silence` (>0.5s): 8 clips — auto-trimmed in Phase 4.

## Diacritization
- **1146 clips fully diacritized** (indices 1–1146; agent-script lines).
- **371 clips undiacritized** (indices 1147–1517; reflective/spiritual batch).
  → Audio is correct; only the *text* lacks tashkīl. Fix list:
  `data/manifests/diacritization_fixlist.json`. Auto-diacritize + human review in Phase 3.

## Decision
Train on all 1517 after Phase-3 diacritization of the 371. Reassess the 9 long clips in Phase 4.
