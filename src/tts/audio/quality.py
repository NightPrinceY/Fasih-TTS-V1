"""Per-clip audio quality analysis for dataset QC.

Loads at native sample rate (mono) and reports the signals that matter for TTS
training data hygiene: duration, clipping, loudness, and leading/trailing silence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass
class ClipQC:
    path: str
    sr: int
    duration: float
    peak: float           # max |sample| in [0,1]
    rms_dbfs: float       # loudness proxy
    clipping: bool        # peak within 0.5 dB of full scale
    lead_silence: float   # seconds of silence at start
    trail_silence: float  # seconds of silence at end
    ok: bool
    flags: str


def analyze_clip(path: str, min_dur: float = 1.0, max_dur: float = 15.0,
                 trim_db: int = 30) -> dict:
    y, sr = librosa.load(path, sr=None, mono=True)
    dur = len(y) / sr if sr else 0.0
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    rms = float(np.sqrt(np.mean(y**2))) if y.size else 0.0
    rms_dbfs = 20 * np.log10(rms) if rms > 0 else -120.0
    clipping = peak >= 0.995

    # leading/trailing silence via energy-based trim
    if y.size:
        _, idx = librosa.effects.trim(y, top_db=trim_db)
        lead = idx[0] / sr
        trail = (len(y) - idx[1]) / sr
    else:
        lead = trail = 0.0

    flags = []
    if clipping:
        flags.append("clipping")
    if dur < min_dur:
        flags.append("too_short")
    if dur > max_dur:
        flags.append("too_long")
    if lead > 0.5:
        flags.append("lead_silence")
    if trail > 0.5:
        flags.append("trail_silence")
    if rms_dbfs < -40:
        flags.append("very_quiet")

    qc = ClipQC(
        path=str(Path(path).name), sr=sr, duration=round(dur, 3),
        peak=round(peak, 4), rms_dbfs=round(rms_dbfs, 1), clipping=clipping,
        lead_silence=round(lead, 3), trail_silence=round(trail, 3),
        ok=(len(flags) == 0), flags=";".join(flags),
    )
    return asdict(qc)
