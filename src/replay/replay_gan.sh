#!/usr/bin/env bash
# Step 3 — watch the GAN play. Requires the API server running (step 0),
# or use `uv run play-gan` which auto-starts it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env.sh"
cd "$REPO_ROOT"

exec uv run python src/replay/replay_gan.py \
  --model-dir ./models/gan_clean_v2 \
  --mode search \
  --num-samples 8 \
  "$@"
