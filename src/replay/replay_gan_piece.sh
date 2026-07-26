#!/usr/bin/env bash
# Step 3 (single-piece demo) — GAN on a long L-piece sequence. Requires API server (step 0).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env.sh"
cd "$REPO_ROOT"

exec uv run python src/replay/replay_gan.py \
  --model-dir ./models/gan_clean_v2 \
  --mode search \
  --num-samples 128 \
  --line-clear-bonus 12 \
  --max-moves 300 \
  --piece-sequence $(printf 'L %.0s' {1..300}) \
  "$@"
