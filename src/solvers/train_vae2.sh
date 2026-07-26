#!/usr/bin/env bash
# Step 2 (v2, canonical) — train the improved VAE. Writes to models/vae_v2_trained/
# (shipped models/vae_clean/ is left untouched). Point replay at your new dir to use it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env.sh"
cd "$REPO_ROOT"

exec uv run python src/solvers/training_vae_v2.py \
  --data-dir ./data/training_data_clean \
  --model-dir ./models/vae_v2_trained \
  --epochs 50 \
  --max-samples 400000 \
  --latent-dim 64 \
  --beta 0.05 \
  --beta-anneal-epochs 15 \
  "$@"
