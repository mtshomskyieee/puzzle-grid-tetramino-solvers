#!/usr/bin/env python3
"""Train a transformer placement-SCORER for puzzle grid.

For each board state in the (clean) training data we enumerate every legal
reachable placement of the current piece, simulate it, and label all of them
(contrastive: good vs bad on the same board). The transformer scores a
(board, piece, candidate) and is trained to predict a shaped quality score plus
auxiliary lines-cleared and holes-created heads. See docs/transformer_scorer_plan.md.

At play time, replay_transformers.py enumerates candidates, scores them, and
picks the best. The model attends to: the whole board + its surface (column
heights), the piece type/rotation, and a hole-fit descriptor of each candidate.
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from puzzlegrid.pieces import PieceType
from puzzlegrid import placement
import transformer_features as tf_feat


class PositionIndicesLayer(tf.keras.layers.Layer):
    """Generate position indices [0..num_cells-1] per batch (serializable)."""

    def __init__(self, num_cells: int, **kwargs):
        super().__init__(**kwargs)
        self.num_cells = num_cells

    def call(self, inputs):
        batch = tf.shape(inputs)[0]
        pos = tf.range(self.num_cells, dtype=tf.int32)
        return tf.tile(tf.expand_dims(pos, 0), [batch, 1])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_cells": self.num_cells})
        return cfg


# ---------------------------------------------------------------- data
def load_games(data_dir: str, max_files: int = None) -> List[Dict]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if max_files:
        files = files[:max_files]
    games = []
    for fp in files:
        try:
            with open(fp) as f:
                games.append(json.load(f))
        except Exception as e:
            print(f"  skip {fp}: {e}")
    print(f"Loaded {len(games)} games from {data_dir}")
    return games


def build_examples(games: List[Dict], max_candidates: int = 20, seed: int = 42):
    """Contrastive examples: enumerate candidates per board state."""
    rng = np.random.RandomState(seed)
    feats = {k: [] for k in tf_feat.FEATURE_KEYS}
    score_y, lines_y, holes_y, weights = [], [], [], []

    for g in games:
        w = g.get("width", 10)
        h = g.get("height", 20)
        prev_after = None
        for mv in g.get("moves", []):
            if mv.get("move_type") != "place":
                continue
            pstr = (mv.get("piece_type") or "").upper()
            if pstr not in tf_feat.PIECE_TYPE_MAP:
                prev_after = mv.get("board_state") or prev_after
                continue
            piece_type = PieceType[pstr]
            board_before = tf_feat.reconstruct_board_before(prev_after, w, h)

            candidates = placement.get_valid_placements(board_before, piece_type)
            if not candidates:
                prev_after = mv.get("board_state") or prev_after
                continue
            if len(candidates) > max_candidates:
                idx = rng.choice(len(candidates), max_candidates, replace=False)
                candidates = [candidates[i] for i in idx]

            for (r, c, rot) in candidates:
                fdict = tf_feat.candidate_features(board_before, piece_type, r, c, rot)
                s, lc, hc, wt = tf_feat.candidate_labels(board_before, piece_type, r, c, rot)
                for k in tf_feat.FEATURE_KEYS:
                    feats[k].append(fdict[k])
                score_y.append(s); lines_y.append(lc); holes_y.append(hc); weights.append(wt)

            prev_after = mv.get("board_state") or prev_after

    if not score_y:
        raise ValueError("No training examples produced.")
    X = {k: np.array(v) for k, v in feats.items()}
    # List order must match model outputs: [score, lines_cleared, holes_created].
    y = [
        np.array(score_y, dtype=np.float32),
        np.array(lines_y, dtype=np.float32),
        np.array(holes_y, dtype=np.float32),
    ]
    weights = np.array(weights, dtype=np.float32)
    print(f"Built {len(score_y):,} candidate examples "
          f"(line-clearing: {int((y[1] > 0).sum()):,})")
    return X, y, weights


# ---------------------------------------------------------------- model
def build_scorer_model(board_height=20, board_width=10, d_model=128,
                       num_heads=8, num_layers=4) -> tf.keras.Model:
    num_cells = board_height * board_width

    board_in = tf.keras.layers.Input((board_height, board_width, 1), name="board")
    piece_in = tf.keras.layers.Input((7,), name="piece_type")
    rot_in = tf.keras.layers.Input((4,), name="rotation")
    pos_in = tf.keras.layers.Input((2,), name="position")
    surf_in = tf.keras.layers.Input((board_width,), name="surface")
    holefit_in = tf.keras.layers.Input((5,), name="holefit")

    cells = tf.keras.layers.Reshape((num_cells, 1))(board_in)
    cells = tf.keras.layers.Dense(d_model, name="cell_embed")(cells)
    pos_idx = PositionIndicesLayer(num_cells, name="pos_idx")(board_in)
    pos_enc = tf.keras.layers.Embedding(num_cells, d_model, name="pos_encoding")(pos_idx)
    cells = tf.keras.layers.Add(name="add_pos")([cells, pos_enc])

    def token(x, name):
        x = tf.keras.layers.Dense(d_model, activation="relu", name=name)(x)
        return tf.keras.layers.Reshape((1, d_model))(x)

    piece_tok = token(tf.keras.layers.Concatenate()([piece_in, rot_in]), "piece_embed")
    pos_tok = token(pos_in, "position_embed")
    surf_tok = token(surf_in, "surface_embed")     # top-of-stack profile
    hole_tok = token(holefit_in, "holefit_embed")  # hole-fit of candidate

    x = tf.keras.layers.Concatenate(axis=1)([cells, piece_tok, pos_tok, surf_tok, hole_tok])

    for i in range(num_layers):
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, name=f"attn_{i}")(x, x)
        x = tf.keras.layers.LayerNormalization(name=f"norm1_{i}")(
            tf.keras.layers.Add(name=f"add1_{i}")([x, attn]))
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model * 4, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(d_model),
        ], name=f"ffn_{i}")(x)
        x = tf.keras.layers.LayerNormalization(name=f"norm2_{i}")(
            tf.keras.layers.Add(name=f"add2_{i}")([x, ffn]))

    pooled = tf.keras.layers.GlobalAveragePooling1D(name="pool")(x)
    d = tf.keras.layers.Dense(256, activation="relu")(pooled)
    d = tf.keras.layers.Dropout(0.3)(d)
    d = tf.keras.layers.Dense(128, activation="relu")(d)
    d = tf.keras.layers.Dense(64, activation="relu")(d)

    score = tf.keras.layers.Dense(1, name="score")(d)
    lines = tf.keras.layers.Dense(1, name="lines_cleared")(d)
    holes = tf.keras.layers.Dense(1, name="holes_created")(d)

    model = tf.keras.Model(
        inputs=[board_in, piece_in, rot_in, pos_in, surf_in, holefit_in],
        outputs=[score, lines, holes],
    )
    # List-based loss/weights (output order: score, lines_cleared, holes_created).
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=["mse", "mse", "mse"],
        loss_weights=[1.0, 0.3, 0.3],
    )
    return model


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Train transformer placement-scorer")
    ap.add_argument("--data-dir", default="./data/training_data_clean")
    ap.add_argument("--model-dir", default="./models/transformer_scorer_trained")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--max-candidates", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--resume-from", default=None)
    args = ap.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    print("=" * 60)
    print("PuzzleGrid Transformer Placement-Scorer Training")
    print("=" * 60)

    games = load_games(args.data_dir, args.max_files)
    if not games:
        print("ERROR: no training data")
        return
    X, y, weights = build_examples(games, max_candidates=args.max_candidates)

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from {args.resume_from}")
        model = tf.keras.models.load_model(
            args.resume_from, compile=False,
            custom_objects={"PositionIndicesLayer": PositionIndicesLayer})
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=["mse", "mse", "mse"],
            loss_weights=[1.0, 0.3, 0.3])
    else:
        model = build_scorer_model(d_model=args.d_model, num_heads=args.num_heads,
                                   num_layers=args.num_layers)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.model_dir, "transformer_scorer_best.keras"),
            monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=4, min_lr=1e-6),
    ]

    model.fit(X, y, sample_weight=weights, validation_split=0.1,
              epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks,
              verbose=2)

    final = os.path.join(args.model_dir, "transformer_scorer_final.keras")
    model.save(final)
    print(f"\nSaved: {final}")


if __name__ == "__main__":
    main()
