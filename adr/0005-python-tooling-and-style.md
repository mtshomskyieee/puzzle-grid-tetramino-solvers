# ADR 0005: Python tooling and style

- **Status:** Accepted
- **Date:** 2026-07-26

## Context

The package is Python-first and driven by `uv`. Contributors and agents need a
short contract for how to run, test, and style code without inventing a linter
stack that is not configured in the repo.

## Decision

1. **Runtime:** Python `>=3.10` as declared in `pyproject.toml`. Manage and run
   with **uv** (`uv sync`, `uv run <command>`).

2. **Entry points:** Prefer console scripts (`engine`, `play-*`, `train-*`,
   `generate-data`, `test`, …) defined in `pyproject.toml` and implemented in
   `src/cli/` / `src/tools/`. For raw `python src/...` invocations, `source
   src/_env.sh` first so `PYTHONPATH` and `REPO_ROOT` are set.

3. **Tests:** pytest is the gate (`uv run test`). Config lives under
   `[tool.pytest.ini_options]` (`testpaths=tests`, `pythonpath` for
   engine/common/solvers/replay). Keep one test module per study step.

4. **Typing and structure:** Use type annotations on public helpers; clear
   module docstrings for non-obvious contracts (especially placement and
   feature builders). Keep the train/replay/common split from ADR 0001.

5. **Hardware default:** CPU by default (`CUDA_VISIBLE_DEVICES=-1` in cli /
   `_env.sh`). Opt into GPU explicitly when needed.

6. **Formatters / typecheckers:** No ruff, black, or mypy configuration in this
   package today. Do not introduce them casually; match the style of
   neighboring modules instead.

## Consequences

- Workflow docs and agents should recommend `uv run …` over ad-hoc venv
  activation.
- Style debates resolve by matching existing files, not by adding an undeclared
  tool.
- Expanding packaging (installable `common`/`solvers`/`replay` wheels) would be
  a new ADR; path-based imports remain intentional for the guided study.
