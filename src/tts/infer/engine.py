"""Production XTTS inference engine for the Arabic 'Muslim' voice agent.

Loads the fine-tuned model once, caches the target-voice conditioning latents, runs the
full Arabic text front-end (normalize -> numbers -> diacritize-if-needed -> lexicon ->
chunk), and synthesizes — with a streaming mode for low time-to-first-audio.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

DEFAULT_REFS = [
    "data/processed/wav24/1.wav",
    "data/processed/wav24/5.wav",
    "data/processed/wav24/10.wav",
]


class XttsEngine:
    SR = 24000

    def __init__(self, model_dir: str = "models/xtts_ar_v1_best",
                 base_dir: str = "models/xtts_v2_base",
                 ref_wavs: list[str] | None = None, device: str | None = None,
                 use_diacritizer: bool = True, temperature: float = 0.65):
        from tts.text.pipeline import TextPipeline
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature

        cfg = XttsConfig()
        cfg.load_json(str(Path(base_dir) / "config.json"))
        self.model = Xtts.init_from_config(cfg)
        self.model.load_checkpoint(
            cfg, checkpoint_path=str(Path(model_dir) / "model.pth"),
            vocab_path=str(Path(base_dir) / "vocab.json"), use_deepspeed=False,
        )
        self.model.to(self.device).eval()

        refs = ref_wavs or DEFAULT_REFS
        self.gpt_cond, self.speaker = self.model.get_conditioning_latents(audio_path=refs)

        diac = None
        if use_diacritizer:
            from tts.text.diacritize import Diacritizer
            diac = Diacritizer(device=self.device)
        self.pipe = TextPipeline(diacritizer=diac)
        self._gap = np.zeros(int(self.SR * 0.12), dtype=np.float32)

    def _chunks(self, text: str) -> list[str]:
        return self.pipe.prepare_chunks(text)

    @torch.inference_mode()
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Full utterance as one array."""
        pieces = []
        for i, ch in enumerate(self._chunks(text)):
            out = self.model.inference(
                ch, "ar", self.gpt_cond, self.speaker,
                temperature=self.temperature, repetition_penalty=2.0,
                enable_text_splitting=False,
            )
            pieces.append(np.asarray(out["wav"], dtype=np.float32))
            if i:
                pieces.insert(-1, self._gap)
        return (np.concatenate(pieces) if pieces else np.zeros(0, np.float32)), self.SR

    @torch.inference_mode()
    def stream(self, text: str):
        """Yield audio chunks as they are generated (low time-to-first-audio)."""
        for ch in self._chunks(text):
            for piece in self.model.inference_stream(
                ch, "ar", self.gpt_cond, self.speaker,
                temperature=self.temperature, repetition_penalty=2.0,
                enable_text_splitting=False,
            ):
                yield piece.cpu().numpy().astype(np.float32)

    def time_to_first_audio(self, text: str) -> float:
        t0 = time.time()
        for _ in self.stream(text):
            return time.time() - t0
        return time.time() - t0
