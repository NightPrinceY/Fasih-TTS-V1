# Third-Party Notices

This project builds on external components. Their licenses govern their own code/weights.

| Component | Use | License | Distributed here? |
|-----------|-----|---------|-------------------|
| **Coqui XTTS v2** (`coqui-tts`) | Base model fine-tuned | Code MPL-2.0; **weights: Coqui Public Model License (non-commercial)** | ❌ No (downloaded by user) |
| **CATT** (github.com/abjadai/catt) | Arabic diacritization (vendored in `src/tts/text/catt/`) | MIT | ✅ Code only (with its LICENSE) |
| **Whisper large-v3** (via `faster-whisper`) | Evaluation ASR | MIT (model: OpenAI) | ❌ No (downloaded) |
| PyTorch, librosa, FastAPI, num2words, etc. | Libraries | respective OSS licenses | ❌ No (pip deps) |

## Important
- The **fine-tuned model weights are NOT published** in this repo. They are derived from XTTS v2,
  whose license is **non-commercial**; treat any derived weights accordingly.
- The **training dataset is private** and is not included.
- This repository contains **source code, configs, and documentation only**.
