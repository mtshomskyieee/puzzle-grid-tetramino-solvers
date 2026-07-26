#!/usr/bin/env python3
"""Train a direct-policy transformer distilled from the scorer's self-play.

The student takes a STATE (board + surface + current piece) and directly predicts
where to put the piece — a rotation (4-way) and a column (width-way) — exactly the
kind of choose/rotate/move decision the VAE and GAN generators made, but learned by
imitating the strong scorer teacher (see distill_policy_data.py).

At play time (replay_transformer_policy.py) the predicted (rotation, col) is snapped
to the nearest legal placement, like the VAE/GAN snap step.
"""

import argparse
import glob
import json
import os

import numpy as np
import tensorflow as tf

from puzzlegrid.pieces import PieceType
import transformer_features as tf_feat
from train_transformers import PositionIndicesLayer  # reuse serializable layer


def load_records(data_dir: str, max_files: int = None):
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if max_files:
        files = files[:max_files]
    records = []
    for fp in files:
        try:
            with open(fp) as f:
                records.extend(json.load(f).get("records", []))
        except Exception as e:
            print(f"  skip {fp}: {e}")
    print(f"Loaded {len(records):,} records from {len(files)} files")
    return records


def build_examples(records, board_width=10):
    """Build (X, y, soft). If records carry teacher soft distributions (col_probs /
    rot_probs), use them as soft distillation targets; else fall back to hard argmax."""
    soft = bool(records) and ("col_probs" in records[0])
    feats = {k: [] for k in tf_feat.STATE_FEATURE_KEYS}
    rot_y, col_y = [], []
    for r in records:
        pstr = r["piece_type"].upper()
        if pstr not in tf_feat.PIECE_TYPE_MAP:
            continue
        piece_type = PieceType[pstr]
        fd = tf_feat.state_features(r["board_before"], piece_type)
        for k in tf_feat.STATE_FEATURE_KEYS:
            feats[k].append(fd[k])
        if soft:
            rot_y.append(np.array(r["rot_probs"], dtype=np.float32))
            col_y.append(np.array(r["col_probs"], dtype=np.float32))
        else:
            rot_y.append(int(r["rotation"]) % 4)
            col_y.append(int(np.clip(r["col"], 0, board_width - 1)))
    if not rot_y:
        raise ValueError("No policy examples produced.")
    X = {k: np.array(v) for k, v in feats.items()}
    if soft:
        y = [np.array(rot_y, dtype=np.float32), np.array(col_y, dtype=np.float32)]
    else:
        y = [np.array(rot_y, dtype=np.int32), np.array(col_y, dtype=np.int32)]
    print(f"Built {len(rot_y):,} policy examples ({'soft' if soft else 'hard'} labels)")
    return X, y, soft


def build_policy_model(board_height=20, board_width=10, d_model=64,
                       num_heads=4, num_layers=2, soft=False) -> tf.keras.Model:
    num_cells = board_height * board_width

    board_in = tf.keras.layers.Input((board_height, board_width, 1), name="board")
    piece_in = tf.keras.layers.Input((7,), name="piece_type")
    surf_in = tf.keras.layers.Input((board_width,), name="surface")

    cells = tf.keras.layers.Reshape((num_cells, 1))(board_in)
    cells = tf.keras.layers.Dense(d_model, name="cell_embed")(cells)
    pos_idx = PositionIndicesLayer(num_cells, name="pos_idx")(board_in)
    pos_enc = tf.keras.layers.Embedding(num_cells, d_model, name="pos_encoding")(pos_idx)
    cells = tf.keras.layers.Add(name="add_pos")([cells, pos_enc])

    def token(x, name):
        x = tf.keras.layers.Dense(d_model, activation="relu", name=name)(x)
        return tf.keras.layers.Reshape((1, d_model))(x)

    piece_tok = token(piece_in, "piece_embed")
    surf_tok = token(surf_in, "surface_embed")
    x = tf.keras.layers.Concatenate(axis=1)([cells, piece_tok, surf_tok])

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

    rotation = tf.keras.layers.Dense(4, name="rotation")(d)          # logits
    column = tf.keras.layers.Dense(board_width, name="column")(d)    # logits

    model = tf.keras.Model(inputs=[board_in, piece_in, surf_in],
                           outputs=[rotation, column])
    if soft:
        # Soft (Hinton) distillation: KL / soft cross-entropy against teacher dists.
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    else:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=[loss_fn, loss_fn],
        metrics=[["accuracy"], ["accuracy"]],
    )
    return model


def main():
    ap = argparse.ArgumentParser(description="Train distilled direct-policy transformer")
    ap.add_argument("--data-dir", default="./data/policy_data")
    ap.add_argument("--model-dir", default="./models/transformer_policy_trained")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    print("=" * 60)
    print("PuzzleGrid Distilled Direct-Policy Transformer Training")
    print("=" * 60)

    records = load_records(args.data_dir, args.max_files)
    if not records:
        print("ERROR: no distillation data (run distill_policy_data.py first)")
        return
    X, y, soft = build_examples(records)

    model = build_policy_model(d_model=args.d_model, num_heads=args.num_heads,
                               num_layers=args.num_layers, soft=soft)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.model_dir, "transformer_policy_best.keras"),
            monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=3, min_lr=1e-6),
    ]
    model.fit(X, y, validation_split=0.1, epochs=args.epochs,
              batch_size=args.batch_size, callbacks=callbacks, verbose=2)

    final = os.path.join(args.model_dir, "transformer_policy_final.keras")
    model.save(final)
    print(f"\nSaved: {final}")


if __name__ == "__main__":
    main()
