#!/usr/bin/env bash
# Step 2 (v1) — watch the original VAE play. Requires the API server running (step 0).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env.sh"
cd "$REPO_ROOT"

exec uv run python src/replay/replay_vae.py \
  --model ./models/tetris_vae_v1_best.keras \
  --num-samples 20 \
  "$@"
