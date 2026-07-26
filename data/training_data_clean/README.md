# training_data_clean

Training data regenerated with the **reachability-fixed engine** (`puzzlegrid/placement.py`).

## How it was generated
- API server: `uv run uvicorn puzzlegrid.api:app --port 8123`
- `API_URL=http://localhost:8123 GAMES=200 MAX_LINES=80 ./generate_clean_data.sh`
  (all-piece games: `--pieces I O T S Z J L`)

## Guarantees
- Every recorded placement is **reachable** from spawn by legal moves (verified by
  `prune_clean_data.py` — 0 unreachable across the set). No enclosed-pocket / floating
  placements, unlike the pre-fix data in `../legacy/training_data_*`.

## Pruning
- `uv run python prune_clean_data.py --dir training_data_clean --min-lines 5 --apply`
- Games clearing < 5 lines moved to `_pruned/` (10 games). Kept: **191 games**.
- Line-clear distribution (pre-prune, 201 games): min 1, median ~17, mean ~20, max 76.
