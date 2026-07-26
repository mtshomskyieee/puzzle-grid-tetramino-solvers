# ADR 0003: Replay clients as the play-side interface

- **Status:** Accepted
- **Date:** 2026-07-26

## Context

Each learned approach needs a way to *play* after training. Putting play loops
inside training scripts blurred the study steps and made it harder to compare
models under the same engine. Some clients need the HTTP API (browser view,
agent keypress navigation); others benefit from in-process speed.

## Decision

1. **Play clients live only under `src/replay/`**, paired with training under
   `src/solvers/`. Invoke them via `uv run play-*` (see `src/cli/__init__.py`);
   bash wrappers under `src/replay/*.sh` remain for manual use.

2. **Shared play loop:** load model → create game → enumerate legal placements
   (via `placement.get_valid_placements` or the API) → choose → place. Agents
   differ by navigating with incremental `/move` instead of `/place`.

3. **Transport:**
   - HTTP for VAE clients and agents (`replay_vae*.py`, `replay_using_agents.py`).
   - In-process (`--inprocess` / CLI default) for GAN and transformer clients
     where supported (`replay_gan.py`, `replay_transformers.py`,
     `replay_transformer_policy.py`), typically via `common.gan_friendly_game`
     or direct `PuzzleGridGame`.

4. **CLI auto-starts** the API when a play command needs the server, then stops
   it afterward — no separate terminal required for the guided path.

5. **Replay may import solver modules only** to register custom Keras layers at
   load time (for example `SamplingLayer`). Replay must not run training.

## Consequences

- Shipped models under `models/` can be exercised immediately without opening
  training scripts.
- Adding a new solver implies a matching `replay_*.py` (and usually a
  `play-*` console script).
- HTTP vs in-process differences are confined to how the game is created and
  how placements are applied; candidate legality still comes from the engine
  (ADR 0002).
