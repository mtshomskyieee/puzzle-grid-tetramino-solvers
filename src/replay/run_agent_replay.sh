#!/usr/bin/env bash
# Step 5 — start the puzzlegrid API server, then run the LangGraph agent replay client.
# The agent uses the transformer scorer (step 4) to choose placements and an OpenAI LLM
# only as a recovery fallback; set OPENAI_API_KEY in .env to enable the fallback.
# Extra args are forwarded to replay_using_agents.py.
#
# Usage:
#   ./run_agent_replay.sh
#   ./run_agent_replay.sh --max-moves 100
#   PORT=8123 ./run_agent_replay.sh

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../_env.sh"
cd "$REPO_ROOT"

PORT="${PORT:-8000}"
API_URL="${API_URL:-http://localhost:${PORT}}"
MODEL="${MODEL:-./models/transformer_scorer/transformer_scorer_best.keras}"
LOG_FILE="${LOG_FILE:-/tmp/puzzlegrid_agent_replay_api.log}"

API_PID=""
STARTED_SERVER=0

cleanup() {
  if [[ "$STARTED_SERVER" -eq 1 && -n "$API_PID" ]]; then
    kill "$API_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
    echo "API server (PID $API_PID) stopped."
  fi
}
trap cleanup EXIT

server_ready() {
  curl -s -f "${API_URL}/" >/dev/null 2>&1
}

if server_ready; then
  echo "API server already running at ${API_URL}"
else
  echo "Starting API server on port ${PORT} (logs: ${LOG_FILE})..."
  uv run uvicorn puzzlegrid.api:app --host 127.0.0.1 --port "$PORT" >"$LOG_FILE" 2>&1 &
  API_PID=$!
  STARTED_SERVER=1

  for i in $(seq 1 30); do
    if server_ready; then
      echo "API server ready (PID $API_PID)."
      break
    fi
    if ! kill -0 "$API_PID" 2>/dev/null; then
      echo "API server exited early. Check ${LOG_FILE}" >&2
      exit 1
    fi
    if [[ "$i" -eq 30 ]]; then
      echo "API server did not become ready within 30s. Check ${LOG_FILE}" >&2
      exit 1
    fi
    sleep 1
  done
fi

echo "Running agent replay against ${API_URL}..."
uv run python src/replay/replay_using_agents.py \
  --model "$MODEL" \
  --api-url "$API_URL" \
  "$@"
