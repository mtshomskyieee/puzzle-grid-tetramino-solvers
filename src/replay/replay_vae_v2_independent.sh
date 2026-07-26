#!/usr/bin/env bash
# Step 2 (v2 variant) — VAE v2 playing with each piece sampled independently.
# Requires the API server running (step 0).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env.sh"
cd "$REPO_ROOT"

exec uv run python src/replay/replay_vae_v2_independent.py \
  --model ./models/vae_clean/tetris_vae_v2_best.keras \
  "$@"
