#!/usr/bin/env python3
"""Train a GAN for Tetris move generation (line-clear focused).

Generator: board + piece + noise -> rotation logits + normalized position.
Discriminator: board + action (+ optional post-board) -> real/fake + line-clear head.

Uses gan_friendly_game for snapping/simulation during training.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from puzzlegrid.pieces import PieceType

from gan_friendly_game import simulate_placement_detailed
from training_vae_v2 import (
    build_board_encoder,
    extract_board_and_actions,
    load_training_data,
    residual_dense_block,
    subsample_training_data,
)


def build_generator(
    board_height: int = 20,
    board_width: int = 10,
    noise_dim: int = 32,
    feature_dim: int = 256,
    hidden_dim: int = 512,
    dropout: float = 0.1,
) -> tf.keras.Model:
    """Generator: board + piece one-hot + noise -> rotation logits + position."""
    board_in = tf.keras.layers.Input(
        shape=(board_height, board_width, 1), name="board"
    )
    piece_in = tf.keras.layers.Input(shape=(7,), name="piece_type")
    noise_in = tf.keras.layers.Input(shape=(noise_dim,), name="noise")

    board_encoder = build_board_encoder(board_height, board_width, feature_dim)
    board_feat = board_encoder(board_in)

    piece_embed = tf.keras.layers.Dense(64, activation="relu")(piece_in)
    x = tf.keras.layers.Concatenate()([board_feat, piece_embed, noise_in])
    x = tf.keras.layers.Dense(hidden_dim, activation="relu")(x)
    x = residual_dense_block(x, hidden_dim, dropout=dropout, name="gen_res_1")
    x = residual_dense_block(x, hidden_dim, dropout=dropout, name="gen_res_2")

    rot_branch = tf.keras.layers.Dense(128, activation="relu")(x)
    rotation_logits = tf.keras.layers.Dense(4, name="rotation_logits")(rot_branch)

    pos_branch = tf.keras.layers.Dense(256, activation="relu")(x)
    pos_branch = tf.keras.layers.Dense(128, activation="relu")(pos_branch)
    position = tf.keras.layers.Dense(2, activation="sigmoid", name="position")(pos_branch)

    return tf.keras.Model(
        inputs=[board_in, piece_in, noise_in],
        outputs=[rotation_logits, position],
        name="tetris_gan_generator",
    )


def build_discriminator(
    board_height: int = 20,
    board_width: int = 10,
    feature_dim: int = 256,
    hidden_dim: int = 512,
    dropout: float = 0.1,
    use_post_board: bool = True,
) -> tf.keras.Model:
    """Discriminator: pre-board + action -> real/fake logit + line-clear logits."""
    board_in = tf.keras.layers.Input(
        shape=(board_height, board_width, 1), name="pre_board"
    )
    action_in = tf.keras.layers.Input(shape=(13,), name="action_vector")

    board_encoder = build_board_encoder(board_height, board_width, feature_dim)
    pre_feat = board_encoder(board_in)

    if use_post_board:
        post_in = tf.keras.layers.Input(
            shape=(board_height, board_width, 1), name="post_board"
        )
        post_feat = board_encoder(post_in)
        x = tf.keras.layers.Concatenate()([pre_feat, post_feat, action_in])
        inputs = [board_in, post_in, action_in]
    else:
        x = tf.keras.layers.Concatenate()([pre_feat, action_in])
        inputs = [board_in, action_in]

    x = tf.keras.layers.Dense(hidden_dim, activation="relu")(x)
    x = residual_dense_block(x, hidden_dim, dropout=dropout, name="disc_res_1")
    x = residual_dense_block(x, hidden_dim, dropout=dropout, name="disc_res_2")

    realism_logit = tf.keras.layers.Dense(1, name="realism_logit")(x)
    line_clear_logits = tf.keras.layers.Dense(5, name="line_clear_logits")(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=[realism_logit, line_clear_logits],
        name="tetris_gan_discriminator",
    )


class TetrisGAN(tf.keras.Model):
    """Combined GAN with alternating generator/discriminator training."""

    def __init__(
        self,
        generator: tf.keras.Model,
        discriminator: tf.keras.Model,
        board_height: int = 20,
        board_width: int = 10,
        noise_dim: int = 32,
        rotation_loss_weight: float = 1.0,
        position_loss_weight: float = 10.0,
        snap_penalty_weight: float = 0.5,
        line_clear_reward_weight: float = 2.0,
        supervised_mix: float = 0.3,
        use_post_board: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.board_height = board_height
        self.board_width = board_width
        self.noise_dim = noise_dim
        self.rotation_loss_weight = rotation_loss_weight
        self.position_loss_weight = position_loss_weight
        self.snap_penalty_weight = snap_penalty_weight
        self.line_clear_reward_weight = line_clear_reward_weight
        self.supervised_mix = supervised_mix
        self.use_post_board = use_post_board

        self.gen_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
        self.disc_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

        self.gen_loss_tracker = tf.keras.metrics.Mean(name="g_loss")
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.sup_loss_tracker = tf.keras.metrics.Mean(name="sup_loss")

    @property
    def metrics(self):
        return [
            self.gen_loss_tracker,
            self.disc_loss_tracker,
            self.sup_loss_tracker,
        ]

    def _action_from_labels(
        self, piece_idx, rotation_idx, position
    ) -> tf.Tensor:
        piece_onehot = tf.one_hot(piece_idx, 7, dtype=tf.float32)
        rot_onehot = tf.one_hot(rotation_idx, 4, dtype=tf.float32)
        return tf.concat([piece_onehot, rot_onehot, position], axis=-1)

    def _post_board_batch(
        self,
        boards_np: np.ndarray,
        piece_idx: np.ndarray,
        rotation_idx: np.ndarray,
        position: np.ndarray,
    ) -> np.ndarray:
        """Build post-placement boards for a batch (numpy)."""
        n = boards_np.shape[0]
        h, w = self.board_height, self.board_width
        post = np.zeros((n, h, w), dtype=np.float32)
        piece_types = [list(PieceType)[int(i)] for i in piece_idx]

        for i in range(n):
            grid = (boards_np[i, ..., 0] > 0).astype(np.int32)
            board_list = grid.tolist()
            pt = piece_types[i]
            row = int(round(float(position[i, 0]) * h))
            col = int(round(float(position[i, 1]) * w))
            rot = int(rotation_idx[i]) % 4
            row = max(0, min(h - 1, row))
            col = max(0, min(w - 1, col))
            sim = simulate_placement_detailed(
                board_list, pt, row, col, rot, snap=True
            )
            if sim.valid and sim.post_board:
                post[i] = (np.array(sim.post_board) > 0).astype(np.float32)
            else:
                post[i] = grid.astype(np.float32)
        return post[..., np.newaxis]

    def _post_board_from_generated(
        self,
        boards_np: np.ndarray,
        piece_idx: np.ndarray,
        gen_rot: np.ndarray,
        gen_pos: np.ndarray,
    ) -> np.ndarray:
        """Build fake post-boards using generated actions."""
        n = boards_np.shape[0]
        h, w = self.board_height, self.board_width
        post = np.zeros((n, h, w), dtype=np.float32)
        piece_types = [list(PieceType)[int(i)] for i in piece_idx]

        for i in range(n):
            grid = (boards_np[i, ..., 0] > 0).astype(np.int32)
            board_list = grid.tolist()
            pt = piece_types[i]
            row = int(round(float(gen_pos[i, 0]) * h))
            col = int(round(float(gen_pos[i, 1]) * w))
            rot = int(gen_rot[i]) % 4
            row = max(0, min(h - 1, row))
            col = max(0, min(w - 1, col))
            sim = simulate_placement_detailed(
                board_list, pt, row, col, rot, snap=True
            )
            if sim.valid and sim.post_board:
                post[i] = (np.array(sim.post_board) > 0).astype(np.float32)
            else:
                post[i] = grid.astype(np.float32)
        return post[..., np.newaxis]

    def train_discriminator_step(
        self,
        board,
        piece_idx,
        rotation_idx,
        position,
        line_clears,
        post_board,
    ):
        batch = tf.shape(board)[0]
        noise = tf.random.normal((batch, self.noise_dim))
        piece_onehot = tf.one_hot(piece_idx, 7, dtype=tf.float32)

        with tf.GradientTape() as tape:
            real_action = self._action_from_labels(piece_idx, rotation_idx, position)
            gen_rot_logits, gen_pos = self.generator(
                [board, piece_onehot, noise], training=True
            )
            gen_rot = tf.argmax(gen_rot_logits, axis=-1, output_type=tf.int32)
            fake_action = self._action_from_labels(piece_idx, gen_rot, gen_pos)

            if self.use_post_board:
                fake_post_board_np = self._post_board_from_generated(
                    board.numpy(),
                    piece_idx.numpy(),
                    gen_rot.numpy(),
                    gen_pos.numpy(),
                )
                fake_post_board = tf.convert_to_tensor(
                    fake_post_board_np, dtype=tf.float32
                )
                real_logits, real_lc = self.discriminator(
                    [board, post_board, real_action], training=True
                )
                fake_logits, fake_lc = self.discriminator(
                    [board, fake_post_board, fake_action], training=True
                )
            else:
                real_logits, real_lc = self.discriminator(
                    [board, real_action], training=True
                )
                fake_logits, fake_lc = self.discriminator(
                    [board, fake_action], training=True
                )

            d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(real_logits), logits=real_logits
                )
            )
            d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(fake_logits), logits=fake_logits
                )
            )
            d_adv = d_loss_real + d_loss_fake

            lc_labels = tf.cast(
                tf.minimum(line_clears, 4), tf.int32
            )
            lc_loss_real = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=lc_labels, logits=real_lc
                )
            )
            lc_loss_fake = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.zeros_like(lc_labels), logits=fake_lc
                )
            )
            d_loss = d_adv + 0.5 * (lc_loss_real + lc_loss_fake)

        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_variables)
        )
        self.disc_loss_tracker.update_state(d_loss)
        return d_loss

    def train_generator_step(
        self,
        board,
        piece_idx,
        rotation_idx,
        position,
        line_clears,
        post_board,
    ):
        batch = tf.shape(board)[0]
        noise = tf.random.normal((batch, self.noise_dim))
        piece_onehot = tf.one_hot(piece_idx, 7, dtype=tf.float32)

        with tf.GradientTape() as tape:
            gen_rot_logits, gen_pos = self.generator(
                [board, piece_onehot, noise], training=True
            )
            gen_rot = tf.argmax(gen_rot_logits, axis=-1, output_type=tf.int32)
            fake_action = self._action_from_labels(piece_idx, gen_rot, gen_pos)

            if self.use_post_board:
                fake_logits, fake_lc = self.discriminator(
                    [board, post_board, fake_action], training=True
                )
            else:
                fake_logits, fake_lc = self.discriminator(
                    [board, fake_action], training=True
                )

            g_adv = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(fake_logits), logits=fake_logits
                )
            )

            lc_pred = tf.cast(tf.argmax(fake_lc, axis=-1), tf.float32)
            g_lc_reward = -self.line_clear_reward_weight * tf.reduce_mean(lc_pred)

            rot_sup = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=rotation_idx, logits=gen_rot_logits
                )
            )
            pos_sup = tf.reduce_mean(tf.square(position - gen_pos))
            sup_loss = (
                self.rotation_loss_weight * rot_sup
                + self.position_loss_weight * pos_sup
            )

            snap_penalty = tf.reduce_mean(
                tf.abs(position - gen_pos)
            )

            g_loss = (
                g_adv
                + g_lc_reward
                + self.supervised_mix * sup_loss
                + self.snap_penalty_weight * snap_penalty
            )

            lc_bonus = self.line_clear_reward_weight * tf.reduce_mean(line_clears)
            g_loss = g_loss - lc_bonus

        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )
        self.gen_loss_tracker.update_state(g_loss)
        self.sup_loss_tracker.update_state(sup_loss)
        return g_loss

    def train_step(self, data):
        board, piece_idx, rotation_idx, position, line_clears, post_board = data

        self.train_discriminator_step(
            board, piece_idx, rotation_idx, position, line_clears, post_board
        )
        self.train_generator_step(
            board, piece_idx, rotation_idx, position, line_clears, post_board
        )
        return {m.name: m.result() for m in self.metrics}


def precompute_post_boards(data: Dict[str, np.ndarray]) -> np.ndarray:
    """Precompute post-placement binary boards for training."""
    boards = data["board"]
    n = len(boards)
    h = boards.shape[1]
    w = boards.shape[2]
    post = np.zeros((n, h, w), dtype=np.float32)
    piece_types = [list(PieceType)[int(i)] for i in data["piece_idx"]]

    for i in range(n):
        board_list = (boards[i] > 0).astype(np.int32).tolist()
        pt = piece_types[i]
        row = int(round(float(data["position"][i, 0]) * h))
        col = int(round(float(data["position"][i, 1]) * w))
        rot = int(data["rotation_idx"][i]) % 4
        row = max(0, min(h - 1, row))
        col = max(0, min(w - 1, col))
        sim = simulate_placement_detailed(board_list, pt, row, col, rot, snap=True)
        if sim.valid:
            post[i] = (np.array(sim.post_board) > 0).astype(np.float32)
        else:
            post[i] = boards[i]
    return post[..., np.newaxis]


def build_tetris_gan(
    board_height: int = 20,
    board_width: int = 10,
    noise_dim: int = 32,
    feature_dim: int = 256,
    **kwargs,
) -> TetrisGAN:
    gen = build_generator(
        board_height, board_width, noise_dim=noise_dim, feature_dim=feature_dim
    )
    disc = build_discriminator(
        board_height, board_width, feature_dim=feature_dim, use_post_board=True
    )
    return TetrisGAN(
        generator=gen,
        discriminator=disc,
        board_height=board_height,
        board_width=board_width,
        noise_dim=noise_dim,
        **kwargs,
    )


def supervised_generator_warmup(
    gan: TetrisGAN,
    data: Dict[str, np.ndarray],
    epochs: int = 5,
    batch_size: int = 32,
) -> None:
    """Warm up generator on real placement labels."""
    board_data = data["board"][..., np.newaxis]
    n = len(board_data)
    indices = np.arange(n)

    for epoch in range(epochs):
        np.random.shuffle(indices)
        losses = []
        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            b = tf.convert_to_tensor(board_data[idx], dtype=tf.float32)
            p = tf.convert_to_tensor(data["piece_idx"][idx], dtype=tf.int32)
            r = tf.convert_to_tensor(data["rotation_idx"][idx], dtype=tf.int32)
            pos = tf.convert_to_tensor(data["position"][idx], dtype=tf.float32)
            piece_oh = tf.one_hot(p, 7, dtype=tf.float32)
            noise = tf.random.normal((tf.shape(p)[0], gan.noise_dim))

            with tf.GradientTape() as tape:
                rot_logits, pred_pos = gan.generator(
                    [b, piece_oh, noise], training=True
                )
                rot_loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=r, logits=rot_logits
                    )
                )
                pos_loss = tf.reduce_mean(tf.square(pos - pred_pos))
                loss = rot_loss + gan.position_loss_weight * pos_loss

            grads = tape.gradient(loss, gan.generator.trainable_variables)
            gan.gen_optimizer.apply_gradients(
                zip(grads, gan.generator.trainable_variables)
            )
            losses.append(float(loss))

        print(f"  Warmup epoch {epoch + 1}/{epochs} loss={np.mean(losses):.4f}")


def train_gan(
    gan: TetrisGAN,
    data: Dict[str, np.ndarray],
    epochs: int = 20,
    batch_size: int = 32,
    model_dir: str = "./models/tetris_gan",
    warmup_epochs: int = 3,
    max_samples_for_post: int = 50000,
) -> None:
    """Full training: warmup + adversarial loop with precomputed post boards."""
    os.makedirs(model_dir, exist_ok=True)

    if warmup_epochs > 0:
        print(f"Generator warmup ({warmup_epochs} epochs)...")
        supervised_generator_warmup(gan, data, epochs=warmup_epochs, batch_size=batch_size)

    board_data = data["board"]
    if max_samples_for_post > 0 and len(board_data) > max_samples_for_post:
        sub = subsample_training_data(data, max_samples_for_post)
    else:
        sub = data

    print("Precomputing post-placement boards...")
    post_boards = precompute_post_boards(sub)
    board_ch = sub["board"][..., np.newaxis]

    n = len(board_ch)
    n_val = max(1, int(n * 0.2))
    perm = np.random.permutation(n)
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    train_ds = (
        tf.data.Dataset.from_tensor_slices(
            (
                board_ch[train_idx],
                sub["piece_idx"][train_idx],
                sub["rotation_idx"][train_idx],
                sub["position"][train_idx],
                sub["line_clears"][train_idx],
                post_boards[train_idx],
            )
        )
        .shuffle(min(8192, len(train_idx)))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    print(f"Adversarial training ({epochs} epochs, {len(train_idx)} samples)...")
    for epoch in range(epochs):
        gan.gen_loss_tracker.reset_state()
        gan.disc_loss_tracker.reset_state()
        gan.sup_loss_tracker.reset_state()

        for batch in train_ds:
            gan.train_step(batch)

        print(
            f"  Epoch {epoch + 1}/{epochs} "
            f"g_loss={gan.gen_loss_tracker.result():.4f} "
            f"d_loss={gan.disc_loss_tracker.result():.4f} "
            f"sup={gan.sup_loss_tracker.result():.4f}"
        )

    gen_path = os.path.join(model_dir, "tetris_gan_generator.keras")
    disc_path = os.path.join(model_dir, "tetris_gan_discriminator.keras")
    config_path = os.path.join(model_dir, "tetris_gan_config.json")

    gan.generator.save(gen_path)
    gan.discriminator.save(disc_path)

    config = {
        "version": 1,
        "board_height": gan.board_height,
        "board_width": gan.board_width,
        "noise_dim": gan.noise_dim,
        "rotation_loss_weight": gan.rotation_loss_weight,
        "position_loss_weight": gan.position_loss_weight,
        "line_clear_reward_weight": gan.line_clear_reward_weight,
        "generator_path": gen_path,
        "discriminator_path": disc_path,
        "use_post_board": gan.use_post_board,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved generator: {gen_path}")
    print(f"Saved discriminator: {disc_path}")
    print(f"Saved config: {config_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Tetris move GAN")
    parser.add_argument("--data-dir", type=str, default="./training_data")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-dir", type=str, default="./models/tetris_gan")
    parser.add_argument("--noise-dim", type=int, default=32)
    parser.add_argument("--line-clear-reward-weight", type=float, default=2.0)
    args = parser.parse_args()

    print("=" * 60)
    print("Tetris Move GAN Training")
    print("=" * 60)

    training_data = load_training_data(args.data_dir, args.max_files)
    if not training_data:
        print("ERROR: No training data found.")
        return

    data = extract_board_and_actions(training_data)
    if args.max_samples and args.max_samples > 0:
        data = subsample_training_data(data, args.max_samples)

    print(f"Samples: {len(data['board'])}")
    print(f"Line-clearing moves: {int(np.sum(data['line_clears'] > 0))}")

    gan = build_tetris_gan(
        noise_dim=args.noise_dim,
        line_clear_reward_weight=args.line_clear_reward_weight,
    )
    gan.compile()

    train_gan(
        gan,
        data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_dir=args.model_dir,
        warmup_epochs=args.warmup_epochs,
        max_samples_for_post=args.max_samples,
    )


if __name__ == "__main__":
    main()
