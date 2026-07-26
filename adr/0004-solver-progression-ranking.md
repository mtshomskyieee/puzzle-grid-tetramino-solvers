# ADR 0004: Solvers as a progressive study; ranking over generation

- **Status:** Accepted
- **Date:** 2026-07-26

## Context

The package is pedagogical as well as empirical: each solver teaches a lesson
(train/inference mismatch, explicit objectives, ranking vs generation, agents).
A single “best model” folder would hide that arc. Early generative approaches
(VAE, then GAN, then a distilled policy) underperformed a placement *scorer*
that only ranks legal candidates.

## Decision

1. **Training lives under `src/solvers/`** as sequential study steps. Heuristic
   self-play data comes from `src/data_gen/` (not a learned solver, but the
   teacher for all of them).

2. **Canonicity and strength (explicit):**
   - Among VAEs, **v2** is canonical (posterior ranking of legal placements);
     v1 remains as the train/inference-mismatch cautionary tale.
   - Overall play strength: the **transformer placement scorer** is the best
     player; agents wrap that scorer and use an LLM only for recovery.
   - GAN and the distilled transformer policy are intermediate / contrastive
     branches, not replacements for the scorer.

3. **Informal shared contract** (no formal ABC/`Protocol` required): enumerate
   legal placements from the engine, score or rank them, then place. Feature
   parity for transformer train/infer stays in `src/common/transformer_features.py`.

4. **Architectural lesson:** ranking legal moves beats generating a placement
   from scratch for this domain — keep that distinction visible in docs and
   folder names (`train_transformers.py` / `replay_transformers.py` vs
   generative and policy scripts).

## Consequences

- New approaches should state whether they generate or rank, and where they sit
  relative to the scorer baseline.
- Replay and docs must not treat VAE v1 or the distilled policy as the default
  recommendation.
- Tests under `tests/` stay one file per study step so regressions map to the
  same progression.
