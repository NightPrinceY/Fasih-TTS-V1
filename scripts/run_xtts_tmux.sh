#!/usr/bin/env bash
# Launch/resume XTTS fine-tuning in a DETACHED tmux session named 'xtts'.
#
# Survives closing the IDE / terminal / SSH / this Claude session.
# Does NOT survive the machine sleeping/hibernating or WSL shutdown (CUDA context dies).
#
# Usage:
#   scripts/run_xtts_tmux.sh                       # fresh run
#   scripts/run_xtts_tmux.sh path/to/checkpoint.pth  # resume from a checkpoint
#
# Watch it:   tmux attach -t xtts     (detach again with Ctrl-b then d)
# Or tail:    tail -f logs/xtts_train.log
set -euo pipefail
cd "$(dirname "$0")/.."

RESTORE="${1:-}"
ARGS="--config configs/training/xtts_finetune.yaml"
if [ -n "$RESTORE" ]; then
  ARGS="$ARGS --restore $RESTORE"
  echo "resuming from: $RESTORE"
fi

if tmux has-session -t xtts 2>/dev/null; then
  echo "ERROR: a tmux session 'xtts' already exists. Kill it first: tmux kill-session -t xtts"
  exit 1
fi

tmux new-session -d -s xtts \
  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \
   uv run python scripts/train_xtts.py $ARGS 2>&1 | tee -a logs/xtts_train.log"

echo "started detached tmux session 'xtts' (GPU 1)"
echo "attach: tmux attach -t xtts   |   tail: tail -f logs/xtts_train.log"
