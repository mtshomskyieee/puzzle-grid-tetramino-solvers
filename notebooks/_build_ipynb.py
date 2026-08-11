#!/usr/bin/env python3
"""Generate the three research notebooks. Run from repo root or this directory."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

NB_META = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "pygments_lexer": "ipython3",
    },
}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _lines(text),
    }


def _lines(text: str) -> list:
    text = text.strip("\n") + "\n"
    return text.splitlines(keepends=True)


def write_nb(name: str, cells: list) -> None:
    path = HERE / name
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": NB_META,
        "cells": cells,
    }
    path.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {path} ({len(cells)} cells)")


# ---------------------------------------------------------------------------
# 01 VAE
# ---------------------------------------------------------------------------

def vae_cells() -> list:
    return [
        md(
            r"""# VAE solver — training, posterior ranking, and latent activations

A walkthrough of the **conditional VAE** that tries to place tetrominoes on the
puzzle grid. Use this notebook to present the method, run a tiny training loop,
and **probe the latent network**: CNN feature maps, $z$ posteriors, and how
ranking changes when you swap pieces or board scenarios.

**Canonical model:** VAE v2 (posterior ranking). Shipped weights live in
`models/vae_clean/`. Full training is `uv run train-vae` (writes
`models/vae_v2_trained/` and does not overwrite the shipped model).

| | |
|---|---|
| **Idea** | Encode `(board, action) → z`, decode `(board, z, piece) → action` |
| **Play** | Rank every legal placement by reconstruction loss under $q(z \mid board, action)$ |
| **Headline result** | Reconstructs demos well, **clears ~0 lines** in play (covariate shift) |

Kernel: the package `.venv` (run `uv sync --group notebooks` first)."""
        ),
        md(
            """## 0. Setup

Import `nb_setup` **before** TensorFlow so CUDA is pinned to CPU. Matplotlib
and the engine/solver modules are then on `sys.path`."""
        ),
        code(
            """%matplotlib inline
import sys
from pathlib import Path

nb_dir = Path.cwd()
if not (nb_dir / "nb_setup.py").exists():
    nb_dir = nb_dir / "notebooks"
sys.path.insert(0, str(nb_dir))

import nb_setup as N
N.setup_notebook()

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from puzzlegrid.pieces import Piece, PieceType
from puzzlegrid import placement
from training_vae_v2 import (
    extract_board_and_actions,
    load_training_data,
    subsample_training_data,
    build_tetris_vae,
    train_model,
)

print("TF", tf.__version__, "GPUs", tf.config.list_physical_devices("GPU"))"""
        ),
        md(
            r"""## 1. What the VAE is learning

The VAE is a **behavioral clone**, not a value function.

1. A CNN compresses the 20×10 occupancy grid to `board_feat` (256-d).
2. An action encoder maps `(board_feat, piece, rotation, position) → (z_mean, z_log_var)`.
3. A decoder reconstructs the action from `(board_feat, z, piece)`.
4. Training minimizes reconstruction loss + $\beta$-KL (with $\beta$ annealed).

**v1 vs v2.** v1 sampled $z \sim \mathcal{N}(0,I)$ at play time while training
used the posterior — a train/inference mismatch. v2 ranks *legal* placements by
how well the posterior reconstructs them. That is the right inference procedure.
It still fails at play because imitation has no notion of “this board is dead.”

Read: `docs/VAE_writeup.md`, `docs/vae_explaination_and_improvement_plan.md`."""
        ),
        md("## 2. Peek at the teaching data"),
        code(
            """game = N.load_game()
print("file keys:", sorted(game.keys())[:12])
print("size:", game.get("width"), "x", game.get("height"),
      "lines_cleared:", game.get("lines_cleared") or game.get("total_lines_cleared"))
places = [m for m in game["moves"] if m.get("move_type") == "place"]
print("place moves:", len(places), "first piece:", places[0].get("piece_type"))

board, piece = N.board_before_nth_place(game, n=12)
print("scenario piece at move 12:", piece)
N.show_board(board, title=f"Real game, before move 12 ({piece})")"""
        ),
        code(
            """# Small extract so we can see label shapes without loading all 191 games.
sample_games = load_training_data(str(N.DATA_DIR), max_files=3)
data = extract_board_and_actions(sample_games)
print({k: np.shape(v) for k, v in data.items()})
print("piece histogram:", dict(zip(*np.unique(data["piece_idx"], return_counts=True))))
print("rotation histogram:", dict(zip(*np.unique(data["rotation_idx"], return_counts=True))))
print("line-clearing fraction:", float(np.mean(data["line_clears"] > 0)))"""
        ),
        md(
            """## 3. Architecture (v2)

Board CNN keeps spatial layout (strided convs + flatten — not global average
pool). Latent is 64-d. Decoder has separate heads for piece, rotation, and
normalized `(row, col)`.

```
board (20×10×1)
  Conv 32 → 64 → 96 (stride 2,1) → 128 (stride 2,2) → flatten → Dense 256
                                                     board_feat
action = concat[piece(7), rotation(4), position(2)]     (13)
encoder MLP: concat(board_feat, action) → z_mean, z_log_var  (64)
decoder: concat(board_feat, z, piece_embed) → piece / rotation / position
```"""
        ),
        code(
            """# Fresh untrained model so we can print a summary without loading 46MB yet.
demo = build_tetris_vae(latent_dim=64, beta=0.05)
_ = demo.board_encoder(tf.zeros((1, 20, 10, 1)))
demo.board_encoder.summary()"""
        ),
        md(
            """## 4. Optional short training

Leave `RUN_SHORT_TRAINING = False` for a presentation that uses shipped weights
(seconds). Set it `True` to fit a **tiny** subset in-notebook so you can show
the training API. Full training: `uv run train-vae`."""
        ),
        code(
            """RUN_SHORT_TRAINING = False  # flip to True for a 1-epoch demo

if RUN_SHORT_TRAINING:
    games = load_training_data(str(N.DATA_DIR), max_files=4)
    data = subsample_training_data(extract_board_and_actions(games), max_samples=512)
    model = build_tetris_vae(latent_dim=64, beta=0.05, learning_rate=1e-3)
    history = train_model(
        model, data,
        epochs=1, batch_size=32,
        model_dir=str(N.REPO_ROOT / "models" / "vae_notebook_demo"),
        beta_anneal_epochs=1,
        early_stopping_patience=5,
    )
    print({k: v[-1] for k, v in history.history.items()})
else:
    print("Skipping in-notebook training. Loading shipped models/vae_clean/ next.")"""
        ),
        md(
            """## 5. Load the shipped VAE

The saved `TetrisVAEV2` class does not deserialize (encoder/decoder are not in
`get_config`). We rebuild the class and `load_weights` from
`tetris_vae_v2_best.keras` — same weights the replay client wants."""
        ),
        code(
            """vae = N.load_vae_v2()
print("latent_dim", vae.latent_dim, "beta", vae.beta)
print("board encoder", vae.board_encoder.name, "decoder", vae.decoder.name)

conv_names = ["conv2d", "conv2d_1", "conv2d_2", "conv2d_3"]
cnn_acts = N.conv_activation_model(vae.board_encoder, conv_names)"""
        ),
        md(
            """## 6. Experiment lab — pick a piece and a scenario

Edit `PIECE` and `SCENARIO` and re-run from here. The rest of the notebook
uses these. Try:

- `almost_line` + `I` (horizontal) — a free line clear the VAE often still misses
- `right_well` + `I` — the Tetris well
- `bumpy` + `T` — off-distribution surface
- `empty` + any piece — spawn-like board"""
        ),
        code(
            """PIECE = "I"          # I O T S Z J L
SCENARIO = "almost_line"  # empty almost_line right_well left_well bumpy flat_stack overhang_pocket

board = N.SCENARIOS[SCENARIO]()
piece_type = PieceType[PIECE]
candidates = placement.get_valid_placements(board, piece_type)
print(f"{SCENARIO} / {PIECE}: {len(candidates)} legal reachable placements")
N.show_board(board, title=f"{SCENARIO} — current piece {PIECE}")"""
        ),
        md(
            """## 7. CNN activations (the spatial latent)

Each conv layer is a stack of feature maps. Watch how early layers light up
occupied cells and later layers smear that into wells / holes / row-fill
patterns. Change the scenario above and re-run to see the difference."""
        ),
        code(
            """board_in = N.encode_board(board)
maps = cnn_acts.predict(board_in, verbose=0)
for name, m in zip(conv_names, maps):
    N.show_feature_maps(m, title=f"board_encoder.{name}  shape={tuple(m.shape)}", max_channels=16)"""
        ),
        md(
            r"""## 8. Posterior $z$ for every legal placement

For each candidate action we encode $q(z \mid board, action)$ and decode.
v2 **ranks by reconstruction loss** (piece CE + rotation CE + 10 × position MSE).
Lower is “more like a demonstrated placement.”"""
        ),
        code(
            """def vae_score_candidates(model, board, piece_name):
    piece = PieceType[piece_name]
    cands = placement.get_valid_placements(board, piece)
    if not cands:
        return [], None, None
    h, w = len(board), len(board[0])
    n = len(cands)
    feat = model.board_encoder(N.encode_board(board), training=False)
    feat_b = tf.repeat(feat, n, axis=0)
    piece_idx = np.full((n,), N.PIECE_NAMES.index(piece_name), np.int32)
    rot = np.array([p[2] for p in cands], np.int32)
    pos = np.array([[p[0] / h, p[1] / w] for p in cands], np.float32)
    action = model.build_action_vector(
        tf.constant(piece_idx), tf.constant(rot), tf.constant(pos)
    )
    z_mean, z_log_var, _ = model.encode(feat_b, action, training=False)
    piece_oh = tf.one_hot(piece_idx, 7, dtype=tf.float32)
    piece_logits, rot_logits, pos_pred = model.decode(feat_b, z_mean, piece_oh, training=False)
    piece_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=piece_idx, logits=piece_logits
    ).numpy()
    rot_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=rot, logits=rot_logits
    ).numpy()
    pos_loss = tf.reduce_mean(tf.square(pos_pred - pos), axis=-1).numpy()
    recon = (
        model.piece_loss_weight * piece_loss
        + model.rotation_loss_weight * rot_loss
        + model.position_loss_weight * pos_loss
    )
    lines = []
    for (r, c, rot_) in cands:
        _, lc = placement.simulate_placement(board, piece, r, c, rot_)
        lines.append(lc)
    rows = []
    for i, (r, c, rot_) in enumerate(cands):
        rows.append({
            "i": i, "row": r, "col": c, "rot": rot_,
            "recon": float(recon[i]),
            "piece_ce": float(piece_loss[i]),
            "rot_ce": float(rot_loss[i]),
            "pos_mse": float(pos_loss[i]),
            "lines": int(lines[i]),
            "z_norm": float(np.linalg.norm(z_mean[i])),
        })
    rows.sort(key=lambda d: d["recon"])
    return rows, z_mean.numpy(), z_log_var.numpy()

ranked, z_mean, z_log_var = vae_score_candidates(vae, board, PIECE)
print(f"{len(ranked)} candidates, best recon={ranked[0]['recon']:.4f}")
N.ranked_table(ranked, ["recon", "col", "row", "rot", "lines", "piece_ce", "rot_ce", "pos_mse", "z_norm"], n=15)"""
        ),
        code(
            """best = ranked[0]
worst = ranked[-1]
N.show_boards(
    [
        {"board": board, "piece": (best["row"], best["col"], best["rot"]),
         "piece_type": PIECE, "title": f"VAE pick  recon={best['recon']:.3f}  lines={best['lines']}"},
        {"board": board, "piece": (worst["row"], worst["col"], worst["rot"]),
         "piece_type": PIECE, "title": f"worst recon={worst['recon']:.3f}  lines={worst['lines']}"},
    ],
    cols=2,
)

# Reconstruction loss as a column heatmap (min over rotations that land in that column).
heat = np.full((20, 10), np.nan)
for row in ranked:
    r, c = row["row"], row["col"]
    if np.isnan(heat[r, c]) or row["recon"] < heat[r, c]:
        heat[r, c] = row["recon"]
fig, ax = plt.subplots(figsize=(4.2, 7.4))
N.show_board(board, heatmap=heat, title="posterior recon loss (lower = more VAE-like)", ax=ax, show=False)
plt.tight_layout(); plt.show()"""
        ),
        md(
            r"""## 9. Look at $z$ itself

`z_mean` is 64-d. We plot (a) the chosen placement’s latent as an 8×8 grid,
(b) per-dimension variance across candidates, (c) a PCA of all candidate
posteriors colored by column — so you can see whether the latent organizes
*where* the piece goes."""
        ),
        code(
            """order = [r["i"] for r in ranked]
z_ord = z_mean[order]
z_best = z_ord[0]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
axes[0].imshow(z_best.reshape(8, 8), cmap="coolwarm", origin="upper")
axes[0].set_title("z_mean of VAE pick (8×8)")
axes[0].set_xticks([]); axes[0].set_yticks([])

std = z_mean.std(axis=0)
axes[1].bar(np.arange(len(std)), std, color="#67e8f9")
axes[1].set_title("std(z) across candidates")
axes[1].set_xlabel("latent dim")

from numpy.linalg import svd
Xz = z_mean - z_mean.mean(axis=0, keepdims=True)
_, _, Vt = svd(Xz, full_matrices=False)
xy = Xz @ Vt[:2].T
cols = np.array([p[1] for p in candidates])
sc = axes[2].scatter(xy[:, 0], xy[:, 1], c=cols, cmap="viridis", s=28)
axes[2].set_title("PCA of z_mean (color = column)")
plt.colorbar(sc, ax=axes[2], fraction=0.046, label="col")
plt.tight_layout(); plt.show()

print("mean |z_mean|", float(np.mean(np.abs(z_mean))),
      "mean exp(0.5 log_var) ~ posterior std",
      float(np.mean(np.exp(0.5 * z_log_var))))"""
        ),
        md(
            """## 10. Latent-dimension sweep

Hold the VAE's chosen $z$ and walk **one coordinate** from -3 to +3. Decode
the resulting `(rotation, position)`. If a dimension is used, the decoded
column/rotation will move; if it is dead, nothing happens. Change `DIM`."""
        ),
        code(
            """DIM = int(np.argmax(std))  # most active dim by default; try 0..63
print("sweeping latent dim", DIM, "std", float(std[DIM]))

feat = vae.board_encoder(N.encode_board(board), training=False)
piece_oh = N.piece_onehot(PIECE)
base = z_best.copy()
vals = np.linspace(-3, 3, 9)
decoded = []
for v in vals:
    z = base.copy()
    z[DIM] = v
    z_in = z.reshape(1, -1).astype(np.float32)
    p_logits, r_logits, pos = vae.decode(feat, z_in, piece_oh, training=False)
    rot = int(np.argmax(r_logits.numpy()[0]))
    row = int(round(float(pos[0, 0]) * 20))
    col = int(round(float(pos[0, 1]) * 10))
    decoded.append((v, row, col, rot))

print("value  row  col  rot")
for v, r, c, rot in decoded:
    print(f"{v:6.2f}  {r:3d}  {c:3d}  {rot:3d}")

fig, ax = plt.subplots(figsize=(6, 3.2))
ax.plot([d[0] for d in decoded], [d[2] for d in decoded], "o-", label="decoded col")
ax.plot([d[0] for d in decoded], [d[3] for d in decoded], "s-", label="decoded rot")
ax.set_xlabel(f"z[{DIM}]"); ax.legend(); ax.set_title("decoder response to a latent walk")
plt.tight_layout(); plt.show()"""
        ),
        md(
            """## 11. Solver step (in-process)

Place the VAE’s ranked choice, show the resulting board, and compare against
the candidate that actually clears the most lines (oracle among legal moves).
This is the v2 replay rule, without the HTTP server."""
        ),
        code(
            """pick = (best["row"], best["col"], best["rot"])
oracle = max(ranked, key=lambda d: (d["lines"], -d["recon"]))
oracle_p = (oracle["row"], oracle["col"], oracle["rot"])
post_vae, lc_vae = placement.simulate_placement(board, piece_type, *pick)
post_or, lc_or = placement.simulate_placement(board, piece_type, *oracle_p)

N.show_boards(
    [
        {"board": post_vae, "title": f"after VAE pick  lines+={lc_vae}"},
        {"board": post_or, "title": f"after line-clear oracle  lines+={lc_or}"},
    ],
    cols=2,
)
print("VAE chose", pick, "oracle", oracle_p)
print("Same move?" , pick == oracle_p)"""
        ),
        md(
            """## 12. Takeaways for a presenter

- **Training works.** Piece/rotation accuracy on held-out *demonstrations* is
  excellent. That is necessary and not sufficient.
- **Inference is posterior ranking**, not prior sampling. That was the v1 bug.
- **Activations are spatially meaningful** — the CNN sees occupancy — but the
  latent is organized around *imitating a placement*, not *valuing a board*.
- On `almost_line`, check whether the lowest recon loss is actually the clear.
  Often it is not: the teacher’s typical I-placement (flat, centered) outvotes
  the rare-but-correct well/clear.
- That is why the GAN adds an explicit line-clear head, and why the transformer
  **scores every legal candidate** instead of generating one.

Next: `02_gan_training_and_solver.ipynb`."""
        ),
    ]


# ---------------------------------------------------------------------------
# 02 GAN
# ---------------------------------------------------------------------------

def gan_cells() -> list:
    return [
        md(
            """# GAN solver — training, discriminator features, and move selection

The GAN is the first model in this study with an **explicit line-clear
objective**. A generator proposes `(rotation, position)` from board + piece +
noise; a discriminator judges `(pre-board, post-board, action)` with a realism
logit **and** a 5-way line-clear head (0–4 lines).

This notebook steps through training, then visualizes **why a move is selected**:
generator proposals, discriminator features (CNN maps + action vector), and the
score terms that rank legal placements.

| | |
|---|---|
| **Idea** | Adversarial generator + discriminator with line-clear head |
| **Play** | Sample / enumerate legal placements, rank by realism − λ·lines |
| **Headline result** | Clears a few lines (~6) — works, high variance |

Shipped weights: `models/gan_clean_v2/`. Full training: `uv run train-gan`."""
        ),
        md("## 0. Setup"),
        code(
            """%matplotlib inline
import sys
from pathlib import Path

nb_dir = Path.cwd()
if not (nb_dir / "nb_setup.py").exists():
    nb_dir = nb_dir / "notebooks"
sys.path.insert(0, str(nb_dir))

import nb_setup as N
N.setup_notebook()

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from puzzlegrid.pieces import PieceType
from puzzlegrid import placement
from gan_friendly_game import (
    encode_board_tensor,
    simulate_placement_detailed,
    snap_to_valid,
    piece_type_to_idx,
    GANFriendlyGame,
)
from replay_gan import load_gan_bundle, build_action_batch, PuzzleGridGANPlayer
from training_gan import (
    build_tetris_gan,
    extract_board_and_actions,
    load_training_data,
    subsample_training_data,
)

print("TF", tf.__version__)"""
        ),
        md(
            """## 1. Why a GAN after the VAE

The VAE reconstructs demonstrated placements and has **no value signal**. The
discriminator adds two things the VAE lacked:

1. **Realism** — does this `(pre, post, action)` look like a teacher move?
2. **Line-clear classification** — 0 / 1 / 2 / 3 / 4 lines, trained on real
   clears and on fakes labeled 0.

The generator is warmed up with supervised rotation/position loss, then trained
adversarially with a line-clear *reward* term. See `docs/readme_gan.md` and
`docs/VAE_TO_GAN_TRANSITION_RESULTS.md`."""
        ),
        md("## 2. Features the GAN actually trains on"),
        code(
            """games = load_training_data(str(N.DATA_DIR), max_files=3)
data = extract_board_and_actions(games)
print("samples", len(data["board"]))
print("action vector layout: piece(7) + rotation(4) + pos(2) = 13")
print("line-clear histogram:", dict(zip(*np.unique(data["line_clears"].astype(int), return_counts=True))))

# Show a real pre-board and the post-board after the recorded placement.
from training_gan import precompute_post_boards
tiny = {k: v[:8] for k, v in data.items()}
post = precompute_post_boards(tiny)
i = int(np.argmax(tiny["line_clears"]))
pre = tiny["board"][i]
N.show_boards(
    [
        {"board": (pre > 0).astype(int).tolist(),
         "title": f"pre-board  piece={N.PIECE_NAMES[int(tiny['piece_idx'][i])]}"},
        {"board": (post[i, ..., 0] > 0).astype(int).tolist(),
         "title": f"post-board  lines={int(tiny['line_clears'][i])}"},
    ],
    cols=2,
)"""
        ),
        md(
            """## 3. Architecture

```
Generator
  board → CNN 256   piece → Dense 64   noise (32)
       concat → MLP 512 + 2× residual
       → rotation logits (4)  and  position sigmoid (2)

Discriminator
  pre-board → CNN 256
  post-board → same CNN 256
  action (13)
       concat → MLP 512 + 2× residual
       → realism logit (1)   line_clear logits (5)
```

Both CNNs reuse the VAE v2 board encoder recipe (`build_board_encoder`)."""
        ),
        code(
            """gan_fresh = build_tetris_gan()
print("generator inputs", [t.name for t in gan_fresh.generator.inputs])
print("generator outputs", [t.name for t in gan_fresh.generator.outputs])
print("disc inputs", [t.name for t in gan_fresh.discriminator.inputs])
print("disc outputs", [t.name for t in gan_fresh.discriminator.outputs])
gan_fresh.generator.summary()"""
        ),
        md(
            """## 4. Optional short training

Default uses shipped `models/gan_clean_v2/`. A 1-epoch warmup on a few hundred
rows is enough to *show* the API; it will not reproduce the published ~6-line
player. Full run: `uv run train-gan`."""
        ),
        code(
            """RUN_SHORT_TRAINING = False

if RUN_SHORT_TRAINING:
    from training_gan import supervised_generator_warmup, train_gan
    games = load_training_data(str(N.DATA_DIR), max_files=3)
    data = subsample_training_data(extract_board_and_actions(games), 256)
    gan = build_tetris_gan(line_clear_reward_weight=6.0)
    gan.compile()
    train_gan(
        gan, data,
        epochs=1, warmup_epochs=1, batch_size=32,
        model_dir=str(N.REPO_ROOT / "models" / "gan_notebook_demo"),
        max_samples_for_post=256,
    )
else:
    print("Skipping in-notebook training.")"""
        ),
        md("## 5. Load the shipped GAN"),
        code(
            """generator, discriminator, gan_cfg = load_gan_bundle(str(N.GAN_DIR))
print(gan_cfg)
noise_dim = int(gan_cfg.get("noise_dim", 32))
print("noise_dim", noise_dim, "line_clear_reward_weight", gan_cfg.get("line_clear_reward_weight"))

# Nested CNN inside generator / discriminator (VAE-v2 board encoder recipe).
gen_cnn = generator.get_layer("board_encoder_v2")
disc_cnn = discriminator.get_layer("board_encoder_v2")
conv_names = ["conv2d", "conv2d_1", "conv2d_2", "conv2d_3"]
gen_acts = N.conv_activation_model(gen_cnn, conv_names)
disc_acts = N.conv_activation_model(disc_cnn, conv_names)
print("gen cnn", gen_cnn.name, "disc cnn", disc_cnn.name)"""
        ),
        md(
            """## 6. Experiment lab — piece and scenario

Edit and re-run. Useful contrasts:

- `almost_line` + `I` or `O` — does the line-clear head fire?
- `right_well` + `I` — Tetris setup; generator vs discriminator disagreement
- `bumpy` + `T` — messy surface, where realism and clears fight"""
        ),
        code(
            """PIECE = "I"
SCENARIO = "almost_line"
LINE_CLEAR_BONUS = 1.5   # replay_gan default; try 0.0 vs 6.0

board = N.SCENARIOS[SCENARIO]()
piece_type = PieceType[PIECE]
candidates = placement.get_valid_placements(board, piece_type)
print(f"{SCENARIO}/{PIECE}: {len(candidates)} legal placements")
N.show_board(board, title=f"{SCENARIO} — piece {PIECE}")"""
        ),
        md(
            """## 7. Generator proposals under noise

The generator is stochastic in noise $z$. Sample many noises, snap each
decoded `(row, col, rot)` onto the legal set, and see the **proposal cloud**.
If proposals pile up on one column, the generator has collapsed; if they miss
the well, the discriminator has to save the play."""
        ),
        code(
            """board_in = encode_board_tensor(board)
piece_oh = N.piece_onehot(PIECE)
n_noise = 64
noise = np.random.randn(n_noise, noise_dim).astype(np.float32)
board_b = np.repeat(board_in, n_noise, axis=0)
piece_b = np.repeat(piece_oh, n_noise, axis=0)
rot_logits, pos = generator.predict([board_b, piece_b, noise], verbose=0)
rots = np.argmax(rot_logits, axis=-1)
rows = np.clip(np.rint(pos[:, 0] * 20).astype(int), 0, 19)
cols = np.clip(np.rint(pos[:, 1] * 10).astype(int), 0, 9)

snapped = []
for r, c, rot in zip(rows, cols, rots):
    s, dist = snap_to_valid(int(r), int(c), int(rot), candidates)
    snapped.append((s, dist, int(r), int(c), int(rot)))

from collections import Counter
legal_counts = Counter(s for s, *_ in snapped if s is not None)
print("unique snapped placements", len(legal_counts))
print("top proposals:", legal_counts.most_common(8))

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
axes[0].hist(cols, bins=np.arange(11) - 0.5, color="#67e8f9", edgecolor="#0b1220")
axes[0].set_title("raw generator columns")
axes[0].set_xlabel("col")
axes[1].hist(rots, bins=np.arange(5) - 0.5, color="#c084fc", edgecolor="#0b1220")
axes[1].set_title("raw generator rotations")
axes[1].set_xlabel("rot")
plt.tight_layout(); plt.show()

# Overlay proposal density on the board (count of snaps per cell of the piece).
density = np.zeros((20, 10), dtype=float)
for (s, dist, *_rest) in snapped:
    if s is None:
        continue
    for rr, cc in N.piece_cells(PIECE, *s):
        if 0 <= rr < 20 and 0 <= cc < 10:
            density[rr, cc] += 1.0
fig, ax = plt.subplots(figsize=(4.2, 7.4))
N.show_board(board, heatmap=density, title="generator proposal density (snapped)", ax=ax, show=False)
plt.tight_layout(); plt.show()"""
        ),
        md(
            """## 8. Discriminator scoring — the features that pick the move

For **every legal placement** we simulate the post-board and ask the
discriminator for:

- `realism` — raw logit (higher = more teacher-like)
- `p(lines)` — softmax over 0–4
- simulated `lines` from the engine (ground truth for this candidate)

Replay combines them as in `replay_gan.py`:

```
combined = realism - λ * simulated_lines - 0.3 * argmax(p(lines)) + 0.1 * snap_dist
```

then prefers more simulated lines, breaking ties with lower `combined`."""
        ),
        code(
            """def score_candidates(board, piece_name, line_clear_bonus=1.5):
    piece = PieceType[piece_name]
    cands = placement.get_valid_placements(board, piece)
    h, w = len(board), len(board[0])
    piece_idx = piece_type_to_idx(piece)
    pre = encode_board_tensor(board)
    posts, actions, lines, valids, post_boards = [], [], [], [], []
    for (r, c, rot) in cands:
        sim = simulate_placement_detailed(board, piece, r, c, rot, snap=False)
        action = build_action_batch(piece_idx, rot, r, c, h, w)
        post_b = sim.post_board if sim.valid else board
        posts.append(encode_board_tensor(post_b)[0])
        actions.append(action[0])
        lines.append(int(sim.lines_cleared) if sim.valid else 0)
        valids.append(sim.valid)
        post_boards.append(post_b)
    pre_b = np.repeat(pre, len(cands), axis=0)
    realism, lc_logits = discriminator.predict(
        [pre_b, np.asarray(posts), np.asarray(actions)], verbose=0
    )
    rows = []
    for i, (r, c, rot) in enumerate(cands):
        lc_prob = tf.nn.softmax(lc_logits[i]).numpy()
        lc_pred = int(np.argmax(lc_prob))
        realism_v = float(realism[i, 0])
        combined = realism_v - line_clear_bonus * lines[i] - 0.3 * lc_pred
        rows.append({
            "row": r, "col": c, "rot": rot,
            "realism": realism_v,
            "lc_pred": lc_pred,
            "p_clear": float(1.0 - lc_prob[0]),
            "lines": lines[i],
            "combined": combined,
            "valid": valids[i],
            "post": post_boards[i],
            "lc_prob": lc_prob,
            "action": np.asarray(actions[i]),
        })
    rows.sort(key=lambda d: (-d["lines"], d["combined"]))
    return rows

ranked = score_candidates(board, PIECE, line_clear_bonus=LINE_CLEAR_BONUS)
print(f"{len(ranked)} candidates")
N.ranked_table(
    ranked,
    ["combined", "lines", "realism", "lc_pred", "p_clear", "col", "row", "rot"],
    n=15,
)"""
        ),
        code(
            """pick, miss = ranked[0], ranked[-1]
N.show_boards(
    [
        {"board": board, "piece": (pick["row"], pick["col"], pick["rot"]),
         "piece_type": PIECE,
         "title": f"GAN pick  lines={pick['lines']}  real={pick['realism']:.2f}"},
        {"board": pick["post"],
         "title": f"post-board after pick  p(clear)={pick['p_clear']:.2f}"},
        {"board": board, "piece": (miss["row"], miss["col"], miss["rot"]),
         "piece_type": PIECE,
         "title": f"lowest-ranked  lines={miss['lines']}  real={miss['realism']:.2f}"},
    ],
    cols=3,
)"""
        ),
        md(
            """## 9. Which feature decided?

Decompose the ranking into the terms the player actually uses. If flipping
`LINE_CLEAR_BONUS` changes the pick, the line-clear head (plus simulated lines)
is doing the work; if not, realism is dominating — the VAE failure mode again."""
        ),
        code(
            """fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
xs = np.arange(min(20, len(ranked)))
sub = ranked[: len(xs)]
axes[0].bar(xs, [r["realism"] for r in sub], color="#60a5fa")
axes[0].set_title("realism logit")
axes[1].bar(xs, [r["lines"] for r in sub], color="#4ade80", label="sim lines")
axes[1].plot(xs, [r["lc_pred"] for r in sub], "o-", color="#facc15", label="pred lines")
axes[1].set_title("line-clear features"); axes[1].legend(fontsize=8)
axes[2].bar(xs, [r["combined"] for r in sub], color="#f87171")
axes[2].set_title("combined (lower/ties after lines)")
for ax in axes:
    ax.set_xlabel("candidate rank")
plt.tight_layout(); plt.show()

# Column heatmap of predicted P(clear) — does the head see the well / gap?
heat = np.full((20, 10), np.nan)
for r in ranked:
    rr, cc = r["row"], r["col"]
    val = r["p_clear"]
    if np.isnan(heat[rr, cc]) or val > heat[rr, cc]:
        heat[rr, cc] = val
fig, ax = plt.subplots(figsize=(4.2, 7.4))
N.show_board(board, heatmap=heat, title="max P(clear) at each resting cell", ax=ax, show=False)
plt.tight_layout(); plt.show()

# Ablate the line-clear bonus and see if the pick changes.
ranked_realism = score_candidates(board, PIECE, line_clear_bonus=0.0)
print("pick with λ=LINE_CLEAR_BONUS", (pick["row"], pick["col"], pick["rot"]), "lines", pick["lines"])
print("pick with λ=0 (realism only)",
      (ranked_realism[0]["row"], ranked_realism[0]["col"], ranked_realism[0]["rot"]),
      "lines", ranked_realism[0]["lines"])"""
        ),
        md(
            """## 10. CNN features of pre- vs post-board

The discriminator shares one CNN across pre- and post-boards. Comparing feature
maps for the **chosen** post-board vs a **no-clear** post-board shows whether
later conv channels respond to completed rows."""
        ),
        code(
            """pre_maps = disc_acts.predict(encode_board_tensor(board), verbose=0)
post_maps = disc_acts.predict(encode_board_tensor(pick["post"]), verbose=0)
N.show_feature_maps(pre_maps[0], "pre-board conv2d (32 maps, first 16)", max_channels=16)
N.show_feature_maps(post_maps[0], "post-board conv2d after GAN pick", max_channels=16)
# Deepest spatial map (5×5×128) — more semantic, less literal occupancy.
N.show_feature_maps(pre_maps[3], "pre-board conv2d_3 (5×5)", max_channels=16)
N.show_feature_maps(post_maps[3], "post-board conv2d_3 (5×5)", max_channels=16)"""
        ),
        md(
            """### Action-vector features

The 13-d action is `piece(7) | rotation(4) | (row/H, col/W)`. Plotting it for
the pick vs a rejected move makes the discriminator’s non-board inputs visible."""
        ),
        code(
            """labels = [f"P{n}" for n in N.PIECE_NAMES] + ["R0", "R1", "R2", "R3", "row", "col"]
fig, ax = plt.subplots(figsize=(10, 3.2))
x = np.arange(13)
ax.bar(x - 0.2, pick["action"], 0.4, label="GAN pick", color="#67e8f9")
ax.bar(x + 0.2, miss["action"], 0.4, label="low-ranked", color="#f87171")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_title("action vector (13-d) — what the discriminator sees besides the boards")
ax.legend()
plt.tight_layout(); plt.show()"""
        ),
        md(
            """## 11. Solver: a few in-process moves

`PuzzleGridGANPlayer` can run without the API (`use_inprocess=True`). Play a
short game on this scenario’s piece set and watch lines."""
        ),
        code(
            """player = PuzzleGridGANPlayer(
    str(N.GAN_DIR),
    use_inprocess=True,
    num_samples=8,
    line_clear_bonus=LINE_CLEAR_BONUS,
)
# Start from an empty game with a biased sequence so the scenario can appear.
seq = [PIECE, "O", "T", "I", "L", "J", "S", "Z"] * 8
player.create_game(piece_sequence=seq)
# Force the first board into our scenario by copying into the local game.
player._local_game._game.board.grid = [row[:] for row in board]
state = player.play_game(max_moves=12, mode="search", delay=0.0)
print("status", state.get("status"),
      "lines", state.get("lines_cleared"),
      "pieces", state.get("pieces_placed"),
      "score", state.get("score"))
N.show_board(state["board"], title=f"after 12 GAN moves  lines={state.get('lines_cleared')}")"""
        ),
        md(
            """## 12. Takeaways

- The GAN’s **selection rule is a linear mix of learned features** (realism,
  predicted clears) and an engine feature (simulated lines). The plots above
  show which term wins on the board you chose.
- Generator noise explores; the discriminator (plus snap-to-legal) **chooses**.
  Ranking legal candidates with a line-clear head is already halfway to the
  transformer — but the GAN still *generates* then patches, instead of scoring
  the full legal set as its primary policy.
- Feature maps of pre vs post are the right place to look for “did it notice
  the completed row?”
- Ceiling in this study: ~6 lines. Next notebook: score every candidate with a
  transformer.

Next: `03_transformer_training_and_solver.ipynb`."""
        ),
    ]


# ---------------------------------------------------------------------------
# 03 Transformer
# ---------------------------------------------------------------------------

def transformer_cells() -> list:
    return [
        md(
            """# Transformer solver — ranking, features, and attention

The transformer is not a move generator. It is a **placement scorer**: every
legal reachable candidate is tokenized (board cells + piece + position +
surface + hole-fit), run through a small encoder, and given a quality score.
Play = `argmax` over that list.

That one change — *evaluate* instead of *emit* — is why this model clears
**150+ lines** while VAE/GAN stall in the single digits.

This notebook is aimed at an AI researcher: feature design, the scoring heads,
and **whether attention goes to the right cells and tokens**.

| | |
|---|---|
| **Idea** | Contrastive scoring of legal candidates |
| **Play** | Enumerate BFS placements, pick highest `score` |
| **Shipped config** | `d_model=64`, 4 heads, 2 layers (not the CLI defaults of 128/8/4) |

Weights: `models/transformer_scorer/transformer_scorer_best.keras`.
Train: `uv run train-transformer`."""
        ),
        md("## 0. Setup"),
        code(
            """%matplotlib inline
import sys
from pathlib import Path

nb_dir = Path.cwd()
if not (nb_dir / "nb_setup.py").exists():
    nb_dir = nb_dir / "notebooks"
sys.path.insert(0, str(nb_dir))

import nb_setup as N
N.setup_notebook()

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from puzzlegrid.pieces import PieceType
from puzzlegrid import placement
import transformer_features as tf_feat
from train_transformers import (
    PositionIndicesLayer,
    build_scorer_model,
    load_games,
    build_examples,
)
from replay_transformers import TransformerScorerPlayer

print("TF", tf.__version__)
print("FEATURE_KEYS", tf_feat.FEATURE_KEYS)"""
        ),
        md(
            """## 1. Ranking beats generation

VAE and GAN emit a `(rotation, position)` and snap it onto the legal set. The
transformer **sees every legal option** and learns `evaluate_placement`.

Training labels (from `transformer_features.candidate_labels`) are a shaped
score: line clears dominate, holes are punished, near-full rows are rewarded.
Auxiliary heads predict `lines_cleared` and `holes_created`.

Token sequence (length 204):

| index | token | meaning |
|------:|-------|---------|
| 0–199 | board cells | row-major occupancy, + positional embedding |
| 200 | piece | concat(piece one-hot, rotation one-hot) |
| 201 | position | normalized (row, col) of this candidate |
| 202 | surface | 10 column heights of the *current* stack |
| 203 | holefit | 5-d descriptor of what *this* move does |

If attention is working, board-cell keys under the piece and along the surface
should light up, and the hole-fit / surface tokens should be used — not ignored."""
        ),
        md("## 2. Feature design — look at one candidate"),
        code(
            """PIECE = "I"
SCENARIO = "almost_line"
board = N.SCENARIOS[SCENARIO]()
piece_type = PieceType[PIECE]
cands = placement.get_valid_placements(board, piece_type)
print("legal candidates", len(cands))

# Two candidates: one that should clear, one that should not (if the scenario allows).
feats_all = []
labels_all = []
for (r, c, rot) in cands:
    feats_all.append(tf_feat.candidate_features(board, piece_type, r, c, rot))
    labels_all.append(tf_feat.candidate_labels(board, piece_type, r, c, rot))

scores = np.array([lab[0] for lab in labels_all])
lines = np.array([lab[1] for lab in labels_all])
holes = np.array([lab[2] for lab in labels_all])
best_i, worst_i = int(np.argmax(scores)), int(np.argmin(scores))
print("teacher-best", cands[best_i], "score", scores[best_i], "lines", lines[best_i], "holes", holes[best_i])
print("teacher-worst", cands[worst_i], "score", scores[worst_i], "lines", lines[worst_i], "holes", holes[worst_i])
print("holefit best ", feats_all[best_i]["holefit"],
      "(holes_created, holes_filled, bump_delta, max_h, clean)")
print("surface      ", np.round(feats_all[best_i]["surface"], 2))

N.show_boards(
    [
        {"board": board, "piece": cands[best_i], "piece_type": PIECE,
         "title": f"teacher-best  s={scores[best_i]:.2f}  L={lines[best_i]}"},
        {"board": board, "piece": cands[worst_i], "piece_type": PIECE,
         "title": f"teacher-worst s={scores[worst_i]:.2f}  L={lines[worst_i]}"},
    ],
    cols=2,
)"""
        ),
        md(
            """## 3. Architecture

Shipped model: **2 layers × 4 heads**, `d_model=64`, `key_dim=16`. CLI defaults
in `build_scorer_model` are larger; the 150+ line run used this smaller net.

```
cells (200×64) + piece + position + surface + holefit   →  (B, 204, 64)
  ×2  MultiHeadAttention + residual LN + FFN (256→64)
  GAP → MLP → score / lines_cleared / holes_created
```"""
        ),
        code(
            """# Untrained twin with the SHIPPED sizes, for a readable summary.
tiny = build_scorer_model(d_model=64, num_heads=4, num_layers=2)
tiny.summary()"""
        ),
        md(
            """## 4. Optional short training

Building contrastive examples enumerates up to `max_candidates` legal placements
per move, so even one game is a few hundred rows. Full training:
`uv run train-transformer`."""
        ),
        code(
            """RUN_SHORT_TRAINING = False

if RUN_SHORT_TRAINING:
    games = load_games(str(N.DATA_DIR), max_files=1)
    X, y, w = build_examples(games, max_candidates=12)
    model = build_scorer_model(d_model=64, num_heads=4, num_layers=2)
    hist = model.fit(X, y, sample_weight=w, epochs=1, batch_size=32, verbose=2)
    print({k: v[-1] for k, v in hist.history.items()})
else:
    print("Skipping in-notebook training; loading shipped scorer.")"""
        ),
        md("## 5. Load the shipped scorer"),
        code(
            """scorer = tf.keras.models.load_model(
    str(N.TRANSFORMER_PATH),
    compile=False,
    custom_objects={"PositionIndicesLayer": PositionIndicesLayer},
)
print("inputs", [t.name for t in scorer.inputs])
print("attn layers", [l.name for l in scorer.layers if "attn" in l.name])
for layer in scorer.layers:
    if isinstance(layer, tf.keras.layers.MultiHeadAttention):
        print(layer.name, "heads", layer.num_heads, "key_dim", layer.key_dim)

TOKEN_SLICE = {
    "cells": slice(0, 200),
    "piece": 200,
    "position": 201,
    "surface": 202,
    "holefit": 203,
}
N_TOKENS = 204"""
        ),
        md(
            """## 6. Score every legal candidate

Compare the **model score** to the **teacher label** and to simulated line
clears. Agreement on the argmax is what matters at play time."""
        ),
        code(
            """def batch_from_board(board, piece_name):
    piece = PieceType[piece_name]
    cands = placement.get_valid_placements(board, piece)
    batch = {k: [] for k in tf_feat.FEATURE_KEYS}
    teacher = []
    sim_lines = []
    for (r, c, rot) in cands:
        fd = tf_feat.candidate_features(board, piece, r, c, rot)
        for k in tf_feat.FEATURE_KEYS:
            batch[k].append(fd[k])
        teacher.append(tf_feat.candidate_labels(board, piece, r, c, rot))
        _, lc = placement.simulate_placement(board, piece, r, c, rot)
        sim_lines.append(lc)
    X = {k: np.array(v) for k, v in batch.items()}
    return cands, X, teacher, np.array(sim_lines)

cands, X, teacher, sim_lines = batch_from_board(board, PIECE)
pred_score, pred_lines, pred_holes = scorer.predict(X, verbose=0)
pred_score = pred_score.reshape(-1)
pred_lines = pred_lines.reshape(-1)
pred_holes = pred_holes.reshape(-1)
teacher_score = np.array([t[0] for t in teacher])

order = np.argsort(-pred_score)
rows = []
for i in order:
    r, c, rot = cands[i]
    rows.append({
        "pred": float(pred_score[i]),
        "teacher": float(teacher_score[i]),
        "lines": int(sim_lines[i]),
        "pred_L": float(pred_lines[i]),
        "pred_H": float(pred_holes[i]),
        "col": c, "row": r, "rot": rot, "i": int(i),
    })
N.ranked_table(rows, ["pred", "teacher", "lines", "pred_L", "pred_H", "col", "row", "rot"], n=15)
print("model argmax", cands[int(np.argmax(pred_score))],
      "teacher argmax", cands[int(np.argmax(teacher_score))])"""
        ),
        code(
            """pick_i = int(np.argmax(pred_score))
worst_i = int(np.argmin(pred_score))
heat = np.full((20, 10), np.nan)
for (r, c, rot), s in zip(cands, pred_score):
    if np.isnan(heat[r, c]) or s > heat[r, c]:
        heat[r, c] = s
N.show_boards(
    [
        {"board": board, "piece": cands[pick_i], "piece_type": PIECE,
         "title": f"model pick  score={pred_score[pick_i]:.3f}  L={sim_lines[pick_i]}"},
        {"board": board, "heatmap": heat,
         "title": "model score at resting cells"},
        {"board": board, "piece": cands[worst_i], "piece_type": PIECE,
         "title": f"model worst score={pred_score[worst_i]:.3f}"},
    ],
    cols=3,
)"""
        ),
        md(
            """## 7. Extract attention

The saved model was not compiled with `return_attention_scores=True`, but the
`MultiHeadAttention` layers still accept that flag when called directly.

Scores shape: `(batch, heads, query, key)` over 204 tokens. We look at:

1. **Board ← special tokens** — does piece / surface / holefit attend to cells?
2. **Board ← board** (mean query) — occupancy self-attention
3. **Head specialization** — 4 heads × 2 layers
4. **Best vs worst candidate** — did attention move to the gap / well?"""
        ),
        code(
            """def attention_for(X_one):
    \"\"\"X_one: feature dict with batch dimension 1. Returns list of (1,H,Q,K) arrays.\"\"\"
    seq_model = getattr(scorer, "_seq_model", None)
    if seq_model is None:
        scorer._seq_model = tf.keras.Model(
            scorer.inputs, scorer.get_layer("concatenate_1").output
        )
        seq_model = scorer._seq_model
    x = seq_model(X_one, training=False)
    scores = []
    for i in range(2):
        attn_layer = scorer.get_layer(f"attn_{i}")
        attn_out, attn_scores = attn_layer(x, x, return_attention_scores=True, training=False)
        scores.append(attn_scores.numpy())
        x = scorer.get_layer(f"add1_{i}")([x, attn_out])
        x = scorer.get_layer(f"norm1_{i}")(x)
        ffn_out = scorer.get_layer(f"ffn_{i}")(x, training=False)
        x = scorer.get_layer(f"add2_{i}")([x, ffn_out])
        x = scorer.get_layer(f"norm2_{i}")(x)
    return scores  # [layer0, layer1], each (1, 4, 204, 204)


def one_index(X, i):
    return {k: v[i : i + 1] for k, v in X.items()}


def cells_to_board(vec200):
    return np.asarray(vec200).reshape(20, 10)


attn_pick = attention_for(one_index(X, pick_i))
attn_worst = attention_for(one_index(X, worst_i))
print("layer0", attn_pick[0].shape, "layer1", attn_pick[1].shape)"""
        ),
        md(
            """### 7a. Special-token → board (layer 0, mean over heads)

Each special token’s attention over the 200 cells, reshaped to 20×10. You want
`holefit` and `piece` to concentrate on the landing site and any gap the move
fills; `surface` should trace the top of the stack."""
        ),
        code(
            """def show_special_to_board(attn_layers, title, piece_placement=None):
    # mean heads, layer 0 and layer 1 stacked
    fig, axes = plt.subplots(2, 4, figsize=(12, 10))
    names = ["piece", "position", "surface", "holefit"]
    idxs = [200, 201, 202, 203]
    for li, attn in enumerate(attn_layers):
        # attn: (1, heads, Q, K)
        mean_heads = attn[0].mean(axis=0)  # (Q, K)
        for j, (name, qi) in enumerate(zip(names, idxs)):
            ax = axes[li, j]
            board_attn = cells_to_board(mean_heads[qi, :200])
            N.show_board(
                board,
                piece=piece_placement,
                piece_type=PIECE if piece_placement else None,
                heatmap=board_attn,
                title=f"L{li} {name}→board",
                ax=ax,
                show=False,
            )
    fig.suptitle(title, fontsize=13)
    plt.tight_layout(); plt.show()

show_special_to_board(attn_pick, "Attention on MODEL PICK", cands[pick_i])
show_special_to_board(attn_worst, "Attention on MODEL WORST", cands[worst_i])"""
        ),
        md(
            """### 7b. Per-head board attention (layer 1)

Layer 1 is closer to the score. Mean over queries restricted to cell tokens,
keys also cells — a spatial self-attention map per head. Heads that are “doing
something” focus on the surface, the hole, or the piece footprint; a collapsed
head is nearly uniform."""
        ),
        code(
            """def show_heads(attn, title):
    # attn (1, 4, 204, 204) — cell queries attending to cell keys
    heads = attn[0, :, :200, :200]  # (4, 200, 200)
    fig, axes = plt.subplots(1, 4, figsize=(14, 6.2))
    for h in range(4):
        # average query attention to each key cell
        key_mass = heads[h].mean(axis=0).reshape(20, 10)
        entropy = -np.sum(heads[h].mean(axis=0) * np.log(heads[h].mean(axis=0) + 1e-12))
        N.show_board(
            board,
            piece=cands[pick_i],
            piece_type=PIECE,
            heatmap=key_mass,
            title=f"head {h}  H={entropy:.2f}",
            ax=axes[h],
            show=False,
        )
    fig.suptitle(title)
    plt.tight_layout(); plt.show()

show_heads(attn_pick[1], "Layer 1 cell→cell key-mass (model pick)")

# Special-token attention mass: how much does each head use holefit / surface?
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for li, ax in enumerate(axes):
    mass = attn_pick[li][0, :, :, 200:].mean(axis=1)  # (heads, 4 special keys)
    im = ax.imshow(mass, cmap="magma", aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(["piece", "pos", "surface", "holefit"], rotation=30)
    ax.set_yticks(range(4)); ax.set_yticklabels([f"h{h}" for h in range(4)])
    ax.set_title(f"layer {li}: head → special-token key mass")
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout(); plt.show()"""
        ),
        md(
            """## 8. Is attention on the right things?

Two quantitative checks a researcher can re-run on any scenario:

1. **Landing-site mass** — fraction of `piece`/`holefit` attention that lands
   on cells the candidate piece occupies (plus 4-neighborhood). Higher on the
   pick than on a random legal move is a good sign.
2. **Feature ablation** — zero `holefit` or `surface` at inference and watch
   the argmax flip. If the pick is unchanged, that token is unused."""
        ),
        code(
            """def landing_mass(attn_layers, placement, token="holefit"):
    qi = TOKEN_SLICE[token]
    cells = N.piece_cells(PIECE, *placement)
    mask = np.zeros(200, dtype=bool)
    for r, c in cells:
        if 0 <= r < 20 and 0 <= c < 10:
            mask[r * 10 + c] = True
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < 20 and 0 <= cc < 10:
                    mask[rr * 10 + cc] = True
    # last layer, mean heads, that query → cells
    a = attn_layers[-1][0, :, qi, :200].mean(axis=0)
    return float(a[mask].sum() / (a.sum() + 1e-9))

print("holefit mass on pick footprint ", landing_mass(attn_pick, cands[pick_i], "holefit"))
print("holefit mass on worst footprint", landing_mass(attn_worst, cands[worst_i], "holefit"))
print("piece  mass on pick footprint  ", landing_mass(attn_pick, cands[pick_i], "piece"))
print("surface mass on pick footprint ", landing_mass(attn_pick, cands[pick_i], "surface"))


def ablate(X, key, fill=0.0):
    Y = {k: v.copy() for k, v in X.items()}
    Y[key] = np.full_like(Y[key], fill)
    return Y

def argmax_of(Xin):
    s, _, _ = scorer.predict(Xin, verbose=0)
    return int(np.argmax(s.reshape(-1)))

base = argmax_of(X)
print("base pick", cands[base])
for key in ("holefit", "surface", "position", "piece_type", "rotation"):
    j = argmax_of(ablate(X, key, 0.0))
    print(f"ablate {key:12s} → {cands[j]}   changed={j != base}")"""
        ),
        md(
            """## 9. Researcher experiments worth running

Change `PIECE` / `SCENARIO` at the top and re-run from §2. Suggested probes:

| Probe | What you hope to see |
|---|---|
| `right_well` + `I` | holefit/piece attention down the well; pick is vertical I in col 9 |
| `almost_line` + `O`/`I` | score head and `lines_cleared` head agree on the completing move |
| `bumpy` + `T` | holefit token used; ablate `holefit` should change the pick |
| `overhang_pocket` | BFS candidates only; attention should not “want” unreachable cells |
| Real game board (`N.board_before_nth_place`) | attention on the actual surface, not the empty sky |

Attention entropy per head (printed on the head maps) is a quick collapse check:
very high entropy ≈ uniform ≈ unused head."""
        ),
        md("## 10. Solver: in-process play"),
        code(
            """player = TransformerScorerPlayer(
    str(N.TRANSFORMER_PATH),
    use_inprocess=True,
    line_clear_bonus=0.0,
)
bag = ["I", "O", "T", "S", "Z", "J", "L"]
seq = [PIECE] + [p for p in bag if p != PIECE]
player.create_game(piece_sequence=seq * 20)
# Drop in our scenario as the starting stack.
player._local._game.board.grid = [row[:] for row in board]
final = player.play(max_moves=40, delay=0.0)
print("lines", final.get("lines_cleared"),
      "pieces", final.get("pieces_placed"),
      "score", final.get("score"),
      "status", final.get("status"))
N.show_board(final["board"], title=f"transformer after 40 moves  lines={final.get('lines_cleared')}")"""
        ),
        md(
            """## 11. Takeaways

- The transformer **attends over a designed token mix**, not raw pixels only.
  Surface and hole-fit are inductive biases that make “paying attention to the
  right things” checkable.
- Use **special-token → board maps**, **per-head entropy**, **landing-site
  mass**, and **ablation of holefit/surface** as a small eval suite whenever
  you retrain.
- Generation (VAE/GAN/distilled policy) and ranking are not interchangeable.
  The distilled policy in `models/transformer_policy_soft/` is the control:
  same family, worse interface, ~5 lines.
- Agents (`uv run play-agents`) wrap this scorer with BFS navigation; the LLM
  is only a recovery fallback.

That is the study arc: imitate (VAE) → add a value-ish head (GAN) → evaluate
every legal move (transformer)."""
        ),
    ]


def main():
    write_nb("01_vae_training_and_solver.ipynb", vae_cells())
    write_nb("02_gan_training_and_solver.ipynb", gan_cells())
    write_nb("03_transformer_training_and_solver.ipynb", transformer_cells())


if __name__ == "__main__":
    main()
