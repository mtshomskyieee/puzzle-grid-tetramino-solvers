#!/usr/bin/env bash
# Step 1 — generate heuristic self-play games into data/training_data_clean/.
# Requires the API server running (default http://localhost:8000; start it in step 0).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env.sh"
cd "$REPO_ROOT"

API_URL="${API_URL:-http://localhost:8000}"
OUT="${OUT:-./data/training_data_clean}"
GAMES="${GAMES:-200}"
MAX_LINES="${MAX_LINES:-80}"

mkdir -p "$OUT"

echo "Generating $GAMES all-piece games (max ${MAX_LINES} lines) -> $OUT"
uv run python src/data_gen/heuristic-data-generator.py \
  --pieces I O T S Z J L \
  --games "$GAMES" \
  --max-lines "$MAX_LINES" \
  --output-dir "$OUT" \
  --api-url "$API_URL"

echo "Done."
