#!/usr/bin/env bash
# Build both Fasih-TTS images (CPU + GPU) and push to Docker Hub.
# Run from the build-context root (the dir containing fasih_tts_server/, src/, configs/).
#
#   DOCKER_HUB_TOKEN=dckr_pat_xxx ./fasih_tts_server/build_and_push.sh
#
# Only builds are needed here — no GPU required to build. The images bake in the
# model (public HF) + CATT checkpoint (GitHub) at build time, so they run offline.
set -euo pipefail

IMG=nightprincey/muslim-fasih-tts
USER_NAME=nightprincey

: "${DOCKER_HUB_TOKEN:?set DOCKER_HUB_TOKEN to your Docker Hub PAT}"

cd "$(dirname "$0")/.."   # -> build-context root

echo ">> docker login"
echo "$DOCKER_HUB_TOKEN" | docker login -u "$USER_NAME" --password-stdin

echo ">> building CPU image ($IMG:v1-cpu)"
docker build -f fasih_tts_server/Dockerfile.cpu -t "$IMG:v1-cpu" .

echo ">> building GPU image ($IMG:v1)"
docker build -f fasih_tts_server/Dockerfile -t "$IMG:v1" .

echo ">> pushing"
docker push "$IMG:v1-cpu"
docker push "$IMG:v1"

docker logout
echo ">> done: pushed $IMG:v1 and $IMG:v1-cpu"
