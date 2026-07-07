"""Audio preprocessing for TTS training.

Deliberately light-touch: the source is lossy 128 kbps MP3, so we avoid aggressive
processing that would amplify compression artifacts. Steps per clip:
  1. resample to target sr (mono),
  2. energy-based silence trim with small padding,
  3. peak-normalize to a fixed headroom (no loudness pumping),
  4. write 16-bit PCM WAV.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def process_clip(in_path: str, out_path: str, target_sr: int = 24000,
                 trim_db: int = 30, peak_dbfs: float = -1.0, pad_ms: int = 50) -> dict:
    y, _ = librosa.load(in_path, sr=target_sr, mono=True)
    if y.size == 0:
        return {"out": out_path, "duration": 0.0, "ok": False}

    yt, _ = librosa.effects.trim(y, top_db=trim_db)
    pad = int(target_sr * pad_ms / 1000)
    yt = np.concatenate([np.zeros(pad, dtype=yt.dtype), yt, np.zeros(pad, dtype=yt.dtype)])

    peak = float(np.max(np.abs(yt)))
    if peak > 0:
        yt = yt * (10 ** (peak_dbfs / 20) / peak)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, yt.astype(np.float32), target_sr, subtype="PCM_16")
    return {"out": out_path, "duration": round(len(yt) / target_sr, 3), "ok": True}
