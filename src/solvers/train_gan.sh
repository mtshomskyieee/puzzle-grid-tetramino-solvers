#!/usr/bin/env bash
# Step 3 — train the Tetris move GAN (line-clear focused). Writes to models/gan_trained/
# (shipped models/gan_clean_v2/ is left untouched).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env.sh"
cd "$REPO_ROOT"

exec uv run python src/solvers/training_gan.py \
  --data-dir ./data/training_data_clean \
  --max-samples 20000 \
  --epochs 15 \
  --warmup-epochs 3 \
  --batch-size 32 \
  --model-dir ./models/gan_trained \
  "$@"
