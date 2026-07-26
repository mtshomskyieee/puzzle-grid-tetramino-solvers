# ADR 0002: Authoritative engine and BFS placement

- **Status:** Accepted
- **Date:** 2026-07-26

## Context

Early placement checks used a column-corridor heuristic: if cells above a target
were empty, the placement was allowed. That wrongly accepted resting positions
inside enclosed pockets reachable only by sliding under an overhang — effectively
teleportation. Those illegal placements contaminated training data and made
solver comparisons untrustworthy.

## Decision

1. **`puzzlegrid` is the sole rules authority.** Data generators, solvers, and
   replay clients never own game rules. They either import the library in-process
   or drive the FastAPI server (`src/engine/puzzlegrid/api.py`, `uv run engine`).

2. **`src/engine/puzzlegrid/placement.py` is the single source of truth** for
   where a piece may legally come to rest. Reachability is a BFS over
   `(row, col, rotation)` from spawn using the same move semantics as
   `game.py` (left, right, clockwise rotate, soft-drop). A placement is legal
   **iff** it is both **reachable** and **resting**.

3. **`filter_low_stack` is opt-in stack-shaping policy**, not part of
   reachability. The API may apply it after `get_valid_placements`; callers must
   not confuse the two.

4. **Two placement interfaces stay first-class:** human-style incremental moves
   (`POST .../move`) and direct placement (`POST .../place`) for RL/AI.

## Consequences

- Every solver and the heuristic generator enumerate candidates from the same
  BFS, so illegal pocket placements cannot enter clean training data by
  construction.
- Fixing or changing rules happens only in `puzzlegrid`; clients stay thin.
- In-process and HTTP play paths must both call into this module (directly or via
  the API) — no parallel “good enough” reachability helpers.
