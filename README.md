# PuzzleGrid: a Tetris-style game engine and four AI solvers

This repository is a **guided study**. It builds a Tetris-style ("puzzle grid") game engine
and then teaches four increasingly capable ways to make an AI play it. Read the code and docs
in order — each step builds on the one before:

0. **Game engine** — the rules, the board, and legal-placement (reachability) logic.
1. **Heuristic data generator** — a hand-written player that produces clean self-play data.
2. **VAE solver** — a generative model that learns to place pieces.
3. **GAN solver** — a generator/discriminator pair trained with an explicit line-clear goal.
4. **Transformer solver** — a placement *scorer* that ranks legal moves (the best player here).
5. **Agents solver** — a LangGraph agent that navigates the board using the transformer scorer,
   with an LLM as a recovery fallback.

The headline arc (max lines cleared on mixed games): **VAE ≈ 0 → GAN ≈ 6 → transformer
ranking ≈ 150+**. See [`docs/VAE_to_GAN_to_TransformerRanking.md`](docs/VAE_to_GAN_to_TransformerRanking.md)
for the full story.

**Architecture decisions** live in [`adr/`](adr/) (layout, engine, replay, solvers,
Python style, agent `.env` usage, and MIT / human–AI-friendly norms). The package is
[MIT licensed](LICENSE).

---

## Repository layout

```
adr/      architecture decision records (see adr/README.md)
data/     training_data_clean/   heuristic self-play games (ships with the repo)
models/   vae_clean/  gan_clean_v2/  transformer_scorer/  transformer_policy_soft/
          tetris_vae_v1_*            pretrained models (ship so you can replay immediately)
docs/     design write-ups, one per stage (see docs/README.md)
src/
  _env.sh          # source this before running anything (sets PYTHONPATH + REPO_ROOT)
  engine/          # step 0: the puzzlegrid package + main.py (the API server)
  data_gen/        # step 1: heuristic player + data validation/analysis
  common/          # shared helpers (gan_friendly_game, transformer_features)
  solvers/         # steps 2-4: training scripts (train_*.sh / training_*.py / distill)
  replay/          # steps 2-5: clients that play the game (replay_*.sh / replay_*.py)
```

The "solvers" (training) and "replay" (playing) code are split into separate folders on
purpose, so you can read *how a model is trained* separately from *how it plays*.

---

## Setup (do this once)

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # create the venv and install dependencies + the uv commands below
```

The whole workflow is driven by **`uv run <command>`** (run from the repository root).
Pretrained models and clean data ship with the repo, so you can **play any solver
immediately** without training. Training commands write to fresh `models/*_trained/` dirs so
they never overwrite the shipped models. Solvers that need the API server auto-start it, run,
then stop it — no separate terminal required.

For the agents step only, copy the env template and add a key:

```bash
cp .env.example .env    # then edit .env and set OPENAI_API_KEY
```

## Command reference

| Command | What it does | Server |
|---|---|---|
| `uv run engine` | start the API server on :8000 (open <http://localhost:8000/>) | — |
| `uv run test` | run the pytest suite | — |
| `uv run generate-data` | step 1 — heuristic self-play → `data/training_data_clean/` | auto |
| `uv run validate-data` / `uv run analyze-data` | check / summarize the data | auto / — |
| `uv run train-vae` · `train-gan` · `train-transformer` | train a solver (also `train-vae1`, `train-policy`, `distill`) | — |
| `uv run play-vae` | step 2 — VAE v2 plays | auto |
| `uv run play-gan` | step 3 — GAN plays (in-process) | — |
| `uv run play-transformer` | step 4 — the scorer plays (best player, in-process) | — |
| `uv run play-agents` | step 5 — LangGraph agent plays | auto |

Extra args pass straight through, e.g. `uv run play-vae --max-moves 50` or
`uv run generate-data --games 5 --max-lines 20`.

Every command also has an equivalent `.sh` wrapper under `src/solvers/` and `src/replay/`
(e.g. `src/replay/replay_vae_v2.sh`) for bash users. For manual `uv run python src/...`
invocations, `source src/_env.sh` first to set `PYTHONPATH`.

The pytest suite (`uv run test`, 18 tests) is fast and offline — it lives in `tests/`, one
file per step (`test_engine.py`, `test_heuristic_data.py`, `test_vae_solver.py`,
`test_gan_solver.py`, `test_transformer_solver.py`, `test_agents_solver.py`).

---

## Step 0 — Game engine

The engine lives in `src/engine/puzzlegrid/`: `board.py`, `pieces.py`, `game.py`, the FastAPI
server `api.py`, and — the heart of the project — `placement.py`. `placement.py` decides
**where a piece can legally come to rest** by running a BFS from the spawn position over legal
moves (left, right, rotate, soft-drop): a placement is valid *iff* it is both reachable and
resting. Everything else (data generation, every solver) treats the engine as the single
source of truth for the rules, either importing it in-process or driving it over HTTP.

Start the API server (the play/generate commands auto-start it, but you can run it yourself):

```bash
uv run engine
```

Open <http://localhost:8000/> to watch games in the browser.

---

## Step 1 — Heuristic data generator

`src/data_gen/heuristic-data-generator.py` is a hand-written player that scores candidate
placements (favoring low, flat, hole-free stacks) and plays complete games over the HTTP API,
saving each as a JSON log. This clean data is what every learned model trains on.

```bash
uv run generate-data                       # 200 games -> data/training_data_clean/
uv run generate-data --games 5 --max-lines 20   # a quick smaller run
```

Inspect the result:

```bash
uv run validate-data    # replays each game, checks legality
uv run analyze-data     # placement / rotation stats
```

Read: [`docs/Heuristic_and_DataGeneration.md`](docs/Heuristic_and_DataGeneration.md),
[`docs/HEURISTIC_EXPLANATION.md`](docs/HEURISTIC_EXPLANATION.md).

---

## Step 2 — VAE solver

A conditional VAE that, given a board and a piece, generates a placement. **v2 is canonical.**
v1 is included as a cautionary tale: it sampled from the prior at play time while training on
the posterior — a train/inference mismatch that made it clear ~0 lines. v2 ranks legal
placements by the posterior instead and actually plays.

```bash
# train (optional — a pretrained model ships in models/vae_clean/)
uv run train-vae                   # v2 -> models/vae_v2_trained/

# play (auto-starts the engine)
uv run play-vae --max-moves 100
uv run play-vae1                   # the v1 cautionary tale
```

Read: [`docs/VAE_writeup.md`](docs/VAE_writeup.md),
[`docs/vae_explaination_and_improvement_plan.md`](docs/vae_explaination_and_improvement_plan.md),
[`docs/vae_followup.md`](docs/vae_followup.md).

---

## Step 3 — GAN solver

A generator/discriminator pair where the discriminator also predicts line clears, giving the
model an **explicit line-clearing objective** the VAE lacked. It clears a few lines reliably.

```bash
# train (optional — a pretrained model ships in models/gan_clean_v2/)
uv run train-gan                   # -> models/gan_trained/

# play (in-process; no server needed)
uv run play-gan
```

Read: [`docs/readme_gan.md`](docs/readme_gan.md),
[`docs/GAN_VAE_handoff.md`](docs/GAN_VAE_handoff.md),
[`docs/GAN_individual_piece_study.md`](docs/GAN_individual_piece_study.md).

---

## Step 4 — Transformer solver

The strongest player. Instead of *generating* a move, it **scores** every legal placement
(surface shape, hole-fit, resulting line clears) and picks the best — ranking beats generation.
This is the model the agents step relies on.

```bash
# train the scorer (optional — ships in models/transformer_scorer/)
uv run train-transformer           # -> models/transformer_scorer_trained/

# play (in-process; no server needed)
uv run play-transformer --max-moves 200
```

A secondary branch **distills** the scorer into a direct-policy student (predicts rotation +
column directly, like the VAE/GAN). It plateaus lower than the scorer — included to show why
ranking wins.

```bash
uv run distill             # scorer self-play -> data/policy_data/
uv run train-policy        # -> models/transformer_policy_trained/
uv run play-policy         # plays with the shipped models/transformer_policy_soft/
```

Read: [`docs/transformer_scorer_plan.md`](docs/transformer_scorer_plan.md),
[`docs/transformer_scorer_results.md`](docs/transformer_scorer_results.md),
[`docs/transformer_policy_distillation.md`](docs/transformer_policy_distillation.md).

---

## Step 5 — Agents solver

A LangGraph agent (game-state → placement → movement → validation) plays over the HTTP API.
The **transformer scorer chooses the target placement** on every move; an OpenAI LLM is called
only as a recovery fallback when a placement is missed or BFS navigation fails. It runs without
an OpenAI key (the fallback is simply skipped with a warning).

```bash
cp .env.example .env    # add OPENAI_API_KEY to enable the LLM fallback (optional)
uv run play-agents --max-moves 100   # auto-starts the engine, then plays
```

Read: [`docs/transformer_ranker_and_agent_pacement.md`](docs/transformer_ranker_and_agent_pacement.md).

---

## Notes

- Everything runs on **CPU by default** (`CUDA_VISIBLE_DEVICES=-1`). For a GPU, prefix a
  command with `CUDA_VISIBLE_DEVICES=0`.
- The API server defaults to **port 8000** everywhere. Override with `--api-url` /
  `PORT=` where supported.
- `.env` is gitignored and must never be committed — only `.env.example` ships.
