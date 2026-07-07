#!/usr/bin/env bash
# Assemble + deploy the Fasih-TTS ZeroGPU demo Space (NightPrince/Fasih-TTS).
# Bundles the text front-end (src/tts) + CATT into space_build/ and uploads.
#
# Requires HF_TOKEN in .env. Speaker latents (speaker_latents.pt) must already be in the
# model repo (precomputed once from reference clips).
set -euo pipefail
cd "$(dirname "$0")/.."

rm -rf space_build/tts space_build/configs
mkdir -p space_build/configs/data
cp -r src/tts space_build/tts
cp configs/data/lexicon_ar.yaml space_build/configs/data/
find space_build -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

uv run python - <<'PY'
tok=[l.split('=',1)[1].strip() for l in open('.env') if l.startswith('HF_TOKEN=')][0]
from huggingface_hub import HfApi
api=HfApi(token=tok); rid="NightPrince/Fasih-TTS"
api.upload_folder(folder_path="space_build", repo_id=rid, repo_type="space",
                  commit_message="Deploy Fasih-TTS demo")
api.request_space_hardware(repo_id=rid, hardware="zero-a10g")
print("deployed ->", "https://huggingface.co/spaces/"+rid)
PY
