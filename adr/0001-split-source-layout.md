# ADR 0001: Split source layout (`common`, `engine`, `solvers`, `replay`, …)

- **Status:** Accepted
- **Date:** 2026-07-26

## Context

This package is a guided study: a puzzle-grid engine plus several solvers that train
and play against it. Mixing training, play clients, shared helpers, and the rules
engine in one flat tree made ownership unclear and encouraged duplicated placement
logic. Readers (human and AI) need a predictable place for each concern.

## Decision

Keep source under `src/` with explicit ownership and a one-way dependency rule:
nothing imports *into* the engine from solvers or replay.

| Path | Ownership |
|------|-----------|
| `src/engine/puzzlegrid/` | Rules + FastAPI; sole authority for legality |
| `src/common/` | Shared train/play helpers only (`gan_friendly_game.py`, `transformer_features.py`) — no training loops, no HTTP clients |
| `src/solvers/` | Training scripts only |
| `src/replay/` | Play clients only |
| `src/data_gen/` | Heuristic self-play and data validation/analysis |
| `src/cli/` | `uv run` console entrypoints |
| `src/tools/` | `uv run test` → pytest |

Hatch packages only `puzzlegrid`, `cli`, and `tools` (see `pyproject.toml`).
`common`, `solvers`, and `replay` stay on `PYTHONPATH` via `src/cli/__init__.py`
and `src/_env.sh`.

Dependency direction: **`puzzlegrid` ← everyone**. `common` may use the engine.
Solvers and replay may use the engine and `common`. Replay may import solver
modules only for custom Keras layers needed at model load time. The engine never
imports solvers or replay.

## Consequences

- Training and playing the same model are read in separate folders by design.
- Shared feature or in-process game helpers belong in `common/`, not copied into
  both `solvers/` and `replay/`.
- New study steps add training under `solvers/` and a matching client under
  `replay/`, wired through `cli` console scripts.
