#!/usr/bin/env bash
# Step 2 (v1) — train the original VAE. Writes to models/vae_v1_trained/ (shipped
# pretrained models are left untouched). CPU by default; CUDA_VISIBLE_DEVICES=0 for GPU.
# --max-samples caps RAM/epoch time on CPU (exit 137 is often OOM); raise or set 0 for all.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env.sh"
cd "$REPO_ROOT"

exec uv run python src/solvers/training_vae_v1.py \
  --data-dir ./data/training_data_clean \
  --model-dir ./models/vae_v1_trained \
  --epochs 50 \
  --max-samples 400000 \
  "$@"
