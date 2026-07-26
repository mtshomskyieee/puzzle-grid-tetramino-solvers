#!/usr/bin/env python3
"""Train a conditional VAE to predict Tetris piece placement from board state.

V2 keeps the data flow from ``training_vae_v1.py`` but fixes the classifier
losses and gives the decoder more capacity for piece / rotation prediction.

Key V2 changes:
- sparse categorical losses use ``from_logits=True`` for raw Dense outputs
- board encoder preserves more spatial information than global average pooling
- decoder uses residual MLP blocks and separate action heads
- piece and rotation accuracy are tracked during training
- saved artifacts use v2 filenames so v1 checkpoints are not overwritten
"""

import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np

# Hide GPUs before TensorFlow import unless user set CUDA_VISIBLE_DEVICES (e.g. to "0").
# Avoids broken/missing CUDA drivers logging cuInit errors; training uses CPU by default.
if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from puzzlegrid.pieces import PieceType


class SamplingLayer(tf.keras.layers.Layer):
    """Reparameterization trick: z = mu + exp(0.5 * log_var) * epsilon."""

    def call(self, inputs, training=None):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def get_config(self):
        return super().get_config()


class TrainBeginCallback(tf.keras.callbacks.Callback):
    """Log once when fit starts so long CPU first batches are not mistaken for a hang."""

    def on_train_begin(self, logs=None):
        print(
            "   Training loop started. On CPU the first epoch can take a long time "
            "with large datasets; watch for batch progress below."
        )


class BetaAnnealingCallback(tf.keras.callbacks.Callback):
    """Ramp beta from 0 to target over the first N epochs."""

    def __init__(self, target_beta: float, anneal_epochs: int = 10):
        super().__init__()
        self.target_beta = target_beta
        self.anneal_epochs = anneal_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if self.anneal_epochs <= 0:
            beta = self.target_beta
        else:
            progress = min(1.0, (epoch + 1) / self.anneal_epochs)
            beta = self.target_beta * progress
        self.model.beta = beta


def load_training_data(data_dir: str = "./training_data", max_files: int = None) -> List[Dict]:
    """Load training data JSON files."""
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    if max_files:
        json_files = json_files[:max_files]

    training_data = []
    for filepath in json_files:
        try:
            with open(filepath, "r") as f:
                training_data.append(json.load(f))
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    print(f"Loaded {len(training_data)} training files")
    return training_data


def extract_board_and_actions(training_data: List[Dict]) -> Dict[str, np.ndarray]:
    """Extract pre-placement boards and placement actions from training data."""
    boards = []
    piece_indices = []
    rotation_indices = []
    positions = []
    line_clears = []

    piece_type_map = {pt.name: idx for idx, pt in enumerate(PieceType)}

    for game_data in training_data:
        board_width = game_data.get("width", 10)
        board_height = game_data.get("height", 20)
        moves = game_data.get("moves", [])
        if not moves:
            continue

        allowed_pieces = game_data.get("allowed_pieces", [])
        if not allowed_pieces:
            allowed_pieces = [pt.name for pt in PieceType]

        prev_board_state = None

        for move in moves:
            if move.get("move_type") != "place":
                continue

            post_board_state = move.get("board_state")
            if not post_board_state:
                continue

            if prev_board_state is None:
                pre_board = np.zeros((board_height, board_width), dtype=np.float32)
            else:
                pre_board = np.array(prev_board_state, dtype=np.float32)
                pre_board = (pre_board > 0).astype(np.float32)

            piece_type_str = move.get("piece_type")
            if not piece_type_str:
                if len(allowed_pieces) == 1:
                    piece_type_str = allowed_pieces[0]
                elif allowed_pieces:
                    piece_type_str = allowed_pieces[0]
                else:
                    prev_board_state = post_board_state
                    continue

            piece_type_str = piece_type_str.upper()
            if piece_type_str not in piece_type_map:
                prev_board_state = post_board_state
                continue

            piece_position = move.get("piece_position", {})
            row = piece_position.get("row", 0)
            col = piece_position.get("col", 0)
            rotation = piece_position.get("rotation", 0) % 4

            boards.append(pre_board)
            piece_indices.append(piece_type_map[piece_type_str])
            rotation_indices.append(rotation)
            positions.append(
                np.array([row / board_height, col / board_width], dtype=np.float32)
            )
            line_clears.append(float(move.get("lines_cleared", 0)))

            prev_board_state = post_board_state

    if not boards:
        raise ValueError("No valid training examples found. Check training data format.")

    return {
        "board": np.array(boards, dtype=np.float32),
        "piece_idx": np.array(piece_indices, dtype=np.int32),
        "rotation_idx": np.array(rotation_indices, dtype=np.int32),
        "position": np.array(positions, dtype=np.float32),
        "line_clears": np.array(line_clears, dtype=np.float32),
    }


def subsample_training_data(
    data: Dict[str, np.ndarray], max_samples: int, seed: int = 42
) -> Dict[str, np.ndarray]:
    """Randomly subsample rows to reduce RAM and CPU time (max_samples <= 0 = no limit)."""
    n = len(data["board"])
    if max_samples is None or max_samples <= 0 or max_samples >= n:
        return data
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_samples, replace=False)
    return {key: value[idx].copy() for key, value in data.items()}


def residual_dense_block(
    x,
    units: int,
    dropout: float,
    name: str,
):
    """Small residual MLP block used by the V2 action encoder and decoder."""
    shortcut = x
    x = tf.keras.layers.Dense(units, activation="relu", name=f"{name}_dense_1")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn_1")(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")(x)
    x = tf.keras.layers.Dense(units, activation="relu", name=f"{name}_dense_2")(x)
    if shortcut.shape[-1] != units:
        shortcut = tf.keras.layers.Dense(units, name=f"{name}_projection")(shortcut)
    x = tf.keras.layers.Add(name=f"{name}_add")([shortcut, x])
    return tf.keras.layers.Activation("relu", name=f"{name}_out")(x)


def build_board_encoder(
    board_height: int = 20,
    board_width: int = 10,
    feature_dim: int = 256,
) -> tf.keras.Model:
    """CNN encoder for board state -> feature vector.

    V1 used GlobalAveragePooling2D, which discards much of the board layout.
    V2 uses strided convolutions followed by flattening so holes, wells, and
    near-complete rows remain easier for the action heads to distinguish.
    """
    board_input = tf.keras.layers.Input(
        shape=(board_height, board_width, 1), name="board"
    )

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(board_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(96, (3, 3), activation="relu", padding="same", strides=(2, 1))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(feature_dim, activation="relu")(x)
    board_feat = tf.keras.layers.LayerNormalization(name="board_feat_norm")(x)
    return tf.keras.Model(board_input, board_feat, name="board_encoder_v2")


def build_decoder(
    latent_dim: int = 64,
    feature_dim: int = 256,
    hidden_dim: int = 512,
    dropout: float = 0.1,
) -> tf.keras.Model:
    """Decoder: board features + latent z + piece one-hot -> action heads."""
    board_feat_input = tf.keras.layers.Input(shape=(feature_dim,), name="board_feat")
    z_input = tf.keras.layers.Input(shape=(latent_dim,), name="z")
    piece_input = tf.keras.layers.Input(shape=(7,), name="piece_type")

    piece_embed = tf.keras.layers.Dense(64, activation="relu", name="piece_embed")(piece_input)
    x = tf.keras.layers.Concatenate(name="decoder_concat")(
        [board_feat_input, z_input, piece_embed]
    )
    x = tf.keras.layers.Dense(hidden_dim, activation="relu", name="decoder_input_dense")(x)
    x = residual_dense_block(x, hidden_dim, dropout=dropout, name="decoder_res_1")
    x = residual_dense_block(x, hidden_dim, dropout=dropout, name="decoder_res_2")

    piece_branch = tf.keras.layers.Concatenate(name="piece_branch_concat")([x, piece_embed])
    piece_branch = tf.keras.layers.Dense(128, activation="relu", name="piece_branch_dense")(
        piece_branch
    )
    piece_logits = tf.keras.layers.Dense(7, name="piece_logits")(piece_branch)

    rotation_branch = tf.keras.layers.Concatenate(name="rotation_branch_concat")(
        [x, piece_embed]
    )
    rotation_branch = tf.keras.layers.Dense(
        256, activation="relu", name="rotation_branch_dense_1"
    )(rotation_branch)
    rotation_branch = tf.keras.layers.Dropout(dropout, name="rotation_branch_dropout")(
        rotation_branch
    )
    rotation_branch = tf.keras.layers.Dense(
        128, activation="relu", name="rotation_branch_dense_2"
    )(rotation_branch)
    rotation_logits = tf.keras.layers.Dense(4, name="rotation_logits")(rotation_branch)

    position_branch = tf.keras.layers.Concatenate(name="position_branch_concat")(
        [x, piece_embed]
    )
    position_branch = tf.keras.layers.Dense(
        256, activation="relu", name="position_branch_dense_1"
    )(position_branch)
    position_branch = tf.keras.layers.Dense(
        128, activation="relu", name="position_branch_dense_2"
    )(position_branch)
    position = tf.keras.layers.Dense(2, activation="sigmoid", name="position")(position_branch)

    return tf.keras.Model(
        inputs=[board_feat_input, z_input, piece_input],
        outputs=[piece_logits, rotation_logits, position],
        name="decoder_v2",
    )


class TetrisVAEV2(tf.keras.Model):
    """Conditional VAE for board -> placement prediction."""

    def __init__(
        self,
        board_encoder: tf.keras.Model,
        decoder: tf.keras.Model,
        latent_dim: int = 64,
        beta: float = 0.05,
        encoder_hidden_dim: int = 512,
        piece_loss_weight: float = 1.0,
        rotation_loss_weight: float = 1.0,
        position_loss_weight: float = 10.0,
        line_clear_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.board_encoder = board_encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder_hidden_dim = encoder_hidden_dim
        self.piece_loss_weight = piece_loss_weight
        self.rotation_loss_weight = rotation_loss_weight
        self.position_loss_weight = position_loss_weight
        self.line_clear_weight = line_clear_weight

        self.encoder_dense = tf.keras.layers.Dense(
            encoder_hidden_dim, activation="relu", name="encoder_dense"
        )
        self.encoder_res_1 = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(encoder_hidden_dim, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(encoder_hidden_dim, activation="relu"),
            ],
            name="encoder_res_1",
        )
        self.encoder_res_2 = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(encoder_hidden_dim, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(encoder_hidden_dim, activation="relu"),
            ],
            name="encoder_res_2",
        )
        self.z_mean_layer = tf.keras.layers.Dense(latent_dim, name="z_mean")
        self.z_log_var_layer = tf.keras.layers.Dense(latent_dim, name="z_log_var")
        self.sampling = SamplingLayer(name="sampling")

        self.piece_loss_tracker = tf.keras.metrics.Mean(name="piece_loss")
        self.rotation_loss_tracker = tf.keras.metrics.Mean(name="rotation_loss")
        self.position_loss_tracker = tf.keras.metrics.Mean(name="position_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.piece_accuracy_tracker = tf.keras.metrics.Mean(name="piece_accuracy")
        self.rotation_accuracy_tracker = tf.keras.metrics.Mean(name="rotation_accuracy")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.kl_loss_tracker,
            self.piece_loss_tracker,
            self.rotation_loss_tracker,
            self.position_loss_tracker,
            self.piece_accuracy_tracker,
            self.rotation_accuracy_tracker,
        ]

    def encode(self, board_feat, action_vector, training=False):
        x = self.encoder_dense(
            tf.concat([board_feat, action_vector], axis=-1), training=training
        )
        res = self.encoder_res_1(x, training=training)
        x = tf.nn.relu(x + res)
        res = self.encoder_res_2(x, training=training)
        x = tf.nn.relu(x + res)
        z_mean = self.z_mean_layer(x)
        z_log_var = self.z_log_var_layer(x)
        z = self.sampling([z_mean, z_log_var])
        return z_mean, z_log_var, z

    def decode(self, board_feat, z, piece_onehot, training=False):
        return self.decoder([board_feat, z, piece_onehot], training=training)

    def build_action_vector(self, piece_idx, rotation_idx, position):
        piece_onehot = tf.one_hot(piece_idx, 7, dtype=tf.float32)
        rotation_onehot = tf.one_hot(rotation_idx, 4, dtype=tf.float32)
        return tf.concat([piece_onehot, rotation_onehot, position], axis=-1)

    def _example_weights(self, line_clears):
        if self.line_clear_weight <= 0:
            return tf.ones_like(line_clears, dtype=tf.float32)
        return 1.0 + self.line_clear_weight * tf.cast(line_clears, tf.float32)

    def compute_losses(
        self,
        piece_idx,
        rotation_idx,
        position,
        line_clears,
        outputs,
        z_mean,
        z_log_var,
    ):
        piece_logits, rotation_logits, position_pred = outputs
        example_weights = self._example_weights(line_clears)

        # Dense heads emit raw logits. V1 used the default from_logits=False,
        # which treats logits as probabilities and keeps CE near random baselines.
        piece_loss_per_example = tf.keras.losses.sparse_categorical_crossentropy(
            piece_idx, piece_logits, from_logits=True
        )
        rotation_loss_per_example = tf.keras.losses.sparse_categorical_crossentropy(
            rotation_idx, rotation_logits, from_logits=True
        )
        position_loss_per_example = tf.reduce_mean(tf.square(position - position_pred), axis=-1)

        weight_denominator = tf.reduce_sum(example_weights) + tf.keras.backend.epsilon()
        piece_loss = (
            tf.reduce_sum(piece_loss_per_example * example_weights) / weight_denominator
        )
        rotation_loss = (
            tf.reduce_sum(rotation_loss_per_example * example_weights) / weight_denominator
        )
        position_loss = (
            tf.reduce_sum(position_loss_per_example * example_weights) / weight_denominator
        )

        recon_loss = (
            self.piece_loss_weight * piece_loss
            + self.rotation_loss_weight * rotation_loss
            + self.position_loss_weight * position_loss
        )
        kl_loss = -0.5 * tf.reduce_mean(
            1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        total_loss = recon_loss + self.beta * kl_loss

        piece_pred = tf.argmax(piece_logits, axis=-1, output_type=piece_idx.dtype)
        rotation_pred = tf.argmax(rotation_logits, axis=-1, output_type=rotation_idx.dtype)
        piece_accuracy = tf.reduce_mean(tf.cast(tf.equal(piece_pred, piece_idx), tf.float32))
        rotation_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(rotation_pred, rotation_idx), tf.float32)
        )

        return (
            total_loss,
            recon_loss,
            kl_loss,
            piece_loss,
            rotation_loss,
            position_loss,
            piece_accuracy,
            rotation_accuracy,
        )

    def train_step(self, data):
        board, piece_idx, rotation_idx, position, line_clears = data

        with tf.GradientTape() as tape:
            board_feat = self.board_encoder(board, training=True)
            action_vector = self.build_action_vector(piece_idx, rotation_idx, position)
            piece_onehot = tf.one_hot(piece_idx, 7, dtype=tf.float32)

            z_mean, z_log_var, z = self.encode(board_feat, action_vector, training=True)
            outputs = self.decode(board_feat, z, piece_onehot, training=True)
            losses = self.compute_losses(
                piece_idx,
                rotation_idx,
                position,
                line_clears,
                outputs,
                z_mean,
                z_log_var,
            )

        grads = tape.gradient(losses[0], self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(losses[0])
        self.recon_loss_tracker.update_state(losses[1])
        self.kl_loss_tracker.update_state(losses[2])
        self.piece_loss_tracker.update_state(losses[3])
        self.rotation_loss_tracker.update_state(losses[4])
        self.position_loss_tracker.update_state(losses[5])
        self.piece_accuracy_tracker.update_state(losses[6])
        self.rotation_accuracy_tracker.update_state(losses[7])
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        board, piece_idx, rotation_idx, position, line_clears = data

        board_feat = self.board_encoder(board, training=False)
        action_vector = self.build_action_vector(piece_idx, rotation_idx, position)
        piece_onehot = tf.one_hot(piece_idx, 7, dtype=tf.float32)

        z_mean, z_log_var, z = self.encode(board_feat, action_vector, training=False)
        outputs = self.decode(board_feat, z, piece_onehot, training=False)
        losses = self.compute_losses(
            piece_idx,
            rotation_idx,
            position,
            line_clears,
            outputs,
            z_mean,
            z_log_var,
        )

        self.total_loss_tracker.update_state(losses[0])
        self.recon_loss_tracker.update_state(losses[1])
        self.kl_loss_tracker.update_state(losses[2])
        self.piece_loss_tracker.update_state(losses[3])
        self.rotation_loss_tracker.update_state(losses[4])
        self.position_loss_tracker.update_state(losses[5])
        self.piece_accuracy_tracker.update_state(losses[6])
        self.rotation_accuracy_tracker.update_state(losses[7])
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "latent_dim": self.latent_dim,
                "beta": self.beta,
                "encoder_hidden_dim": self.encoder_hidden_dim,
                "piece_loss_weight": self.piece_loss_weight,
                "rotation_loss_weight": self.rotation_loss_weight,
                "position_loss_weight": self.position_loss_weight,
                "line_clear_weight": self.line_clear_weight,
            }
        )
        return config


def build_tetris_vae(
    board_height: int = 20,
    board_width: int = 10,
    latent_dim: int = 64,
    beta: float = 0.05,
    feature_dim: int = 256,
    encoder_hidden_dim: int = 512,
    decoder_hidden_dim: int = 512,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
    piece_loss_weight: float = 1.0,
    rotation_loss_weight: float = 1.0,
    position_loss_weight: float = 10.0,
    line_clear_weight: float = 0.0,
) -> TetrisVAEV2:
    """Build and compile the full TetrisVAEV2 model."""
    board_encoder = build_board_encoder(board_height, board_width, feature_dim=feature_dim)
    decoder = build_decoder(
        latent_dim=latent_dim,
        feature_dim=feature_dim,
        hidden_dim=decoder_hidden_dim,
        dropout=dropout,
    )

    model = TetrisVAEV2(
        board_encoder=board_encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        beta=beta,
        encoder_hidden_dim=encoder_hidden_dim,
        piece_loss_weight=piece_loss_weight,
        rotation_loss_weight=rotation_loss_weight,
        position_loss_weight=position_loss_weight,
        line_clear_weight=line_clear_weight,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return model


def save_inference_bundle(model: TetrisVAEV2, model_dir: str):
    """Save board_encoder, decoder, and config for lightweight inference loading."""
    encoder_path = os.path.join(model_dir, "tetris_vae_v2_board_encoder.keras")
    decoder_path = os.path.join(model_dir, "tetris_vae_v2_decoder.keras")
    config_path = os.path.join(model_dir, "tetris_vae_v2_config.json")

    model.board_encoder.save(encoder_path)
    model.decoder.save(decoder_path)

    config = {
        "version": 2,
        "latent_dim": model.latent_dim,
        "beta": model.beta,
        "piece_loss_weight": model.piece_loss_weight,
        "rotation_loss_weight": model.rotation_loss_weight,
        "position_loss_weight": model.position_loss_weight,
        "line_clear_weight": model.line_clear_weight,
        "board_encoder_path": encoder_path,
        "decoder_path": decoder_path,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return encoder_path, decoder_path, config_path


def train_model(
    model: TetrisVAEV2,
    data: Dict[str, np.ndarray],
    validation_split: float = 0.2,
    epochs: int = 50,
    batch_size: int = 32,
    model_dir: str = "./models",
    beta_anneal_epochs: int = 10,
    early_stopping_patience: int = 10,
) -> tf.keras.callbacks.History:
    """Train the VAE."""
    board_data = data["board"][..., np.newaxis]

    n_samples = len(board_data)
    n_val = int(n_samples * validation_split)
    indices = np.random.permutation(n_samples)
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    shuffle_buf = min(8192, max(256, len(train_idx)))

    train_ds = tf.data.Dataset.from_tensor_slices(
        (
            board_data[train_idx],
            data["piece_idx"][train_idx],
            data["rotation_idx"][train_idx],
            data["position"][train_idx],
            data["line_clears"][train_idx],
        )
    ).shuffle(shuffle_buf).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(
        (
            board_data[val_idx],
            data["piece_idx"][val_idx],
            data["rotation_idx"][val_idx],
            data["position"][val_idx],
            data["line_clears"][val_idx],
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    os.makedirs(model_dir, exist_ok=True)
    best_path = os.path.join(model_dir, "tetris_vae_v2_best.keras")

    callbacks = [
        TrainBeginCallback(),
        BetaAnnealingCallback(target_beta=model.beta, anneal_epochs=beta_anneal_epochs),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            best_path,
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, early_stopping_patience // 2),
            min_lr=1e-6,
        ),
    ]

    return model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Tetris conditional VAE (v2)")
    parser.add_argument("--data-dir", type=str, default="./data/training_data_clean")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-dir", type=str, default="./models/vae_v2_trained")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--encoder-hidden-dim", type=int, default=512)
    parser.add_argument("--decoder-hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--beta-anneal-epochs", type=int, default=15)
    parser.add_argument("--piece-loss-weight", type=float, default=1.0)
    parser.add_argument("--rotation-loss-weight", type=float, default=1.0)
    parser.add_argument("--position-loss-weight", type=float, default=10.0)
    parser.add_argument(
        "--line-clear-weight",
        type=float,
        default=0.0,
        help="Optional per-example multiplier: weight = 1 + line_clear_weight * lines_cleared.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Randomly use at most this many rows after extraction (0 = use all). "
        "Recommended on CPU or limited RAM when you have millions of moves.",
    )
    parser.add_argument(
        "--subsample-seed",
        type=int,
        default=42,
        help="RNG seed for --max-samples subsampling.",
    )
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    print("=" * 60)
    print("PuzzleGrid Conditional VAE Training (v2)")
    print("=" * 60)

    print("\n1. Loading training data...")
    training_data = load_training_data(args.data_dir, args.max_files)
    if not training_data:
        print("ERROR: No training data found!")
        return

    print("\n2. Extracting pre-placement boards and actions...")
    data = extract_board_and_actions(training_data)
    print(f"   Samples (before subsample): {len(data['board'])}")
    if args.max_samples and args.max_samples > 0:
        data = subsample_training_data(
            data, args.max_samples, seed=args.subsample_seed
        )
        print(f"   Samples (after --max-samples): {len(data['board'])}")

    print(f"   Board shape: {data['board'].shape}")
    print(f"   Unique pieces: {len(np.unique(data['piece_idx']))}")
    print(f"   Unique rotations: {len(np.unique(data['rotation_idx']))}")
    print(f"   Moves clearing lines: {int(np.sum(data['line_clears'] > 0))}")

    print("\n3. Building/loading model...")
    gpus = tf.config.list_physical_devices("GPU")
    print(
        f"   TensorFlow visible devices: "
        f"GPU={len(gpus)} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r})"
    )
    custom_objects = {"SamplingLayer": SamplingLayer, "TetrisVAEV2": TetrisVAEV2}
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"   Resuming from: {args.resume_from}")
        model = tf.keras.models.load_model(
            args.resume_from, custom_objects=custom_objects, compile=False
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        )
    else:
        if args.resume_from:
            print(f"   Warning: resume file not found: {args.resume_from}")
        model = build_tetris_vae(
            latent_dim=args.latent_dim,
            beta=args.beta,
            feature_dim=args.feature_dim,
            encoder_hidden_dim=args.encoder_hidden_dim,
            decoder_hidden_dim=args.decoder_hidden_dim,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            piece_loss_weight=args.piece_loss_weight,
            rotation_loss_weight=args.rotation_loss_weight,
            position_loss_weight=args.position_loss_weight,
            line_clear_weight=args.line_clear_weight,
        )

    model.summary()

    print("\n4. Training...")
    history = train_model(
        model,
        data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_dir=args.model_dir,
        beta_anneal_epochs=args.beta_anneal_epochs,
        early_stopping_patience=args.early_stopping_patience,
    )

    final_path = os.path.join(args.model_dir, "tetris_vae_v2_final.keras")
    model.save(final_path)
    encoder_path, decoder_path, config_path = save_inference_bundle(model, args.model_dir)

    print(f"\n5. Models saved:")
    print(f"   Best:  {os.path.join(args.model_dir, 'tetris_vae_v2_best.keras')}")
    print(f"   Final: {final_path}")
    print(f"   Encoder: {encoder_path}")
    print(f"   Decoder: {decoder_path}")
    print(f"   Config:  {config_path}")

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Final train loss: {history.history['loss'][-1]:.4f}")
    print(f"Final val loss:   {history.history['val_loss'][-1]:.4f}")
    print(f"Best val loss:    {min(history.history['val_loss']):.4f}")
    print(f"Final train piece accuracy:    {history.history['piece_accuracy'][-1]:.4f}")
    print(f"Final val piece accuracy:      {history.history['val_piece_accuracy'][-1]:.4f}")
    print(f"Final train rotation accuracy: {history.history['rotation_accuracy'][-1]:.4f}")
    print(f"Final val rotation accuracy:   {history.history['val_rotation_accuracy'][-1]:.4f}")
    print(f"Epochs trained:   {len(history.history['loss'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
