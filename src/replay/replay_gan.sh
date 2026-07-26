#!/usr/bin/env bash
# Step 3 — watch the GAN play. Uses --inprocess (no API server needed).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env.sh"
cd "$REPO_ROOT"

exec uv run python src/replay/replay_gan.py \
  --model-dir ./models/gan_clean_v2 \
  --inprocess \
  --mode search \
  --num-samples 8 \
  "$@"
