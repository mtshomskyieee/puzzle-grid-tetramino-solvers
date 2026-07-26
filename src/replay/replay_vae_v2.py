#!/usr/bin/env python3
"""AI player that uses a trained VAE v2 model to play puzzle grid games.

V2 replay keeps the API/gameplay helpers from ``replay_vae.py`` but changes
inference to match ``training_vae_v2.py``:

- load v2 artifact names by default
- load the full ``TetrisVAEV2`` model when available
- rank every valid placement with q(z | board, candidate action)
- use proper softmax cross entropy for raw logits

If the full model cannot be loaded, the player falls back to the old prior
sampling path using the v2 board encoder and decoder bundle.
"""

import json
import os
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from puzzlegrid.pieces import PieceType

from replay_vae import PuzzleGridVAEPlayer
from training_vae_v2 import SamplingLayer, TetrisVAEV2


def load_vae_v2_for_inference(
    model_path: str,
) -> Tuple[Optional[TetrisVAEV2], tf.keras.Model, tf.keras.Model, int]:
    """Load the full VAE v2 model when possible, plus its encoder/decoder bundle."""
    model_dir = os.path.dirname(os.path.abspath(model_path))
    config_path = os.path.join(model_dir, "tetris_vae_v2_config.json")
    custom_objects = {"SamplingLayer": SamplingLayer, "TetrisVAEV2": TetrisVAEV2}

    full_model = None
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects=custom_objects,
                safe_mode=False,
            )
            if isinstance(model, TetrisVAEV2):
                full_model = model
        except Exception:
            full_model = None

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        latent_dim = int(config["latent_dim"])
        encoder_path = config.get("board_encoder_path")
        decoder_path = config.get("decoder_path")
        if encoder_path and not os.path.isabs(encoder_path):
            encoder_path = os.path.join(model_dir, os.path.basename(encoder_path))
        if decoder_path and not os.path.isabs(decoder_path):
            decoder_path = os.path.join(model_dir, os.path.basename(decoder_path))
    else:
        latent_dim = getattr(full_model, "latent_dim", 64)
        encoder_path = os.path.join(model_dir, "tetris_vae_v2_board_encoder.keras")
        decoder_path = os.path.join(model_dir, "tetris_vae_v2_decoder.keras")

    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        if full_model is not None:
            return full_model, full_model.board_encoder, full_model.decoder, full_model.latent_dim
        raise ValueError(
            f"Could not load VAE v2 inference bundle from {model_dir}. "
            "Expected tetris_vae_v2_board_encoder.keras and tetris_vae_v2_decoder.keras."
        )

    board_encoder = tf.keras.models.load_model(encoder_path, compile=False)
    decoder = tf.keras.models.load_model(decoder_path, compile=False)
    return full_model, board_encoder, decoder, latent_dim


class PuzzleGridVAEV2Player(PuzzleGridVAEPlayer):
    """PuzzleGrid player using VAE v2 posterior ranking."""

    def __init__(self, model_path: str, api_url: str = "http://localhost:8000"):
        self.api_url = api_url

        try:
            (
                self.full_model,
                self.board_encoder,
                self.decoder,
                self.latent_dim,
            ) = load_vae_v2_for_inference(model_path)
        except Exception as e:
            raise ValueError(
                f"Failed to load model from {model_path}: {e}\n"
                "Make sure the v2 model file exists and v2 inference bundle files are present."
            ) from e

        self.game_id = None
        self.piece_type_map = {pt.name: idx for idx, pt in enumerate(PieceType)}
        self.piece_type_reverse = {idx: pt for pt, idx in self.piece_type_map.items()}

    @staticmethod
    def _sparse_ce_from_logits(logits: np.ndarray, target_idx: int) -> float:
        logits = np.asarray(logits, dtype=np.float64)
        max_logit = np.max(logits)
        logsumexp = max_logit + np.log(np.sum(np.exp(logits - max_logit)))
        return float(logsumexp - logits[target_idx])

    def _reconstruction_error(
        self,
        board_feat: np.ndarray,
        piece_type: PieceType,
        row: int,
        col: int,
        rotation: int,
        z: np.ndarray,
        board_height: int,
        board_width: int,
    ) -> float:
        piece_onehot = self._piece_onehot(piece_type)
        piece_logits, rotation_logits, position_pred = self.decoder.predict(
            [board_feat, z, piece_onehot], verbose=0
        )

        piece_idx = self.piece_type_map[piece_type.name]
        piece_ce = self._sparse_ce_from_logits(piece_logits[0], piece_idx)
        rotation_ce = self._sparse_ce_from_logits(rotation_logits[0], rotation)

        target_pos = np.array(
            [[row / board_height, col / board_width]], dtype=np.float32
        )
        position_mse = float(np.mean((position_pred - target_pos) ** 2))

        return float(piece_ce + rotation_ce + position_mse)

    def _posterior_rank_placements(
        self,
        board: List[List[int]],
        board_feat: np.ndarray,
        piece_type: PieceType,
        valid_placements: List[Tuple[int, int, int]],
        board_height: int,
        board_width: int,
        line_clear_bonus: float = 0.0,
    ) -> Optional[Tuple[int, int, int]]:
        """Rank valid placements using q(z | board, candidate action)."""
        if self.full_model is None:
            return None

        n = len(valid_placements)
        piece_idx = self.piece_type_map[piece_type.name]
        piece_idx_batch = np.full((n,), piece_idx, dtype=np.int32)
        rotation_batch = np.array([rot for _, _, rot in valid_placements], dtype=np.int32)
        position_batch = np.array(
            [[row / board_height, col / board_width] for row, col, _ in valid_placements],
            dtype=np.float32,
        )
        board_feat_batch = np.repeat(board_feat.astype(np.float32), n, axis=0)
        piece_onehot = tf.one_hot(piece_idx_batch, 7, dtype=tf.float32)

        board_feat_tensor = tf.convert_to_tensor(board_feat_batch, dtype=tf.float32)
        action_vector = self.full_model.build_action_vector(
            tf.convert_to_tensor(piece_idx_batch),
            tf.convert_to_tensor(rotation_batch),
            tf.convert_to_tensor(position_batch),
        )
        z_mean, _, _ = self.full_model.encode(
            board_feat_tensor, action_vector, training=False
        )
        piece_logits, rotation_logits, position_pred = self.full_model.decode(
            board_feat_tensor, z_mean, piece_onehot, training=False
        )

        piece_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=piece_idx_batch, logits=piece_logits
        ).numpy()
        rotation_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=rotation_batch, logits=rotation_logits
        ).numpy()
        position_loss = tf.reduce_mean(
            tf.square(position_pred - position_batch), axis=-1
        ).numpy()

        scores = (
            getattr(self.full_model, "piece_loss_weight", 1.0) * piece_loss
            + getattr(self.full_model, "rotation_loss_weight", 1.0) * rotation_loss
            + getattr(self.full_model, "position_loss_weight", 1.0) * position_loss
        )

        if line_clear_bonus > 0:
            line_clears = np.array(
                [
                    self.simulate_placement(board, piece_type, row, col, rot)[1]
                    for row, col, rot in valid_placements
                ],
                dtype=np.float32,
            )
            scores = scores - line_clear_bonus * line_clears

        return valid_placements[int(np.argmin(scores))]

    def choose_best_placement(
        self,
        board: List[List[int]],
        piece_type: PieceType,
        num_samples: int = 20,
        line_clear_bonus: float = 0.0,
    ) -> Optional[Tuple[int, int, int]]:
        board_height = len(board)
        board_width = len(board[0])

        valid_placements = self.get_valid_placements(board, piece_type)
        if not valid_placements:
            return None
        if len(valid_placements) == 1:
            return valid_placements[0]

        board_input = self._prepare_board(board)
        board_feat = self.board_encoder.predict(board_input, verbose=0)

        posterior_choice = self._posterior_rank_placements(
            board,
            board_feat,
            piece_type,
            valid_placements,
            board_height,
            board_width,
            line_clear_bonus=line_clear_bonus,
        )
        if posterior_choice is not None:
            return posterior_choice

        return super().choose_best_placement(
            board,
            piece_type,
            num_samples=num_samples,
        )

    def make_move(self) -> bool:
        state = self.get_game_state()

        if state["status"] != "playing":
            print(f"Game status is not 'playing': {state['status']}")
            return False

        current_piece = state.get("current_piece")
        if not current_piece:
            print("No current piece available")
            return False

        piece_type_str = current_piece["type"]
        piece_type = PieceType[piece_type_str]
        board = state["board"]

        num_samples = getattr(self, "_num_samples", 20)
        line_clear_bonus = getattr(self, "_line_clear_bonus", 0.0)
        placement = self.choose_best_placement(
            board,
            piece_type,
            num_samples=num_samples,
            line_clear_bonus=line_clear_bonus,
        )
        if not placement:
            print(f"No valid placement found for piece {piece_type_str}")
            return False

        row, col, rotation = placement
        return self._submit_move(board, piece_type, piece_type_str, row, col, rotation)

    def _submit_move(
        self,
        board: List[List[int]],
        piece_type: PieceType,
        piece_type_str: str,
        row: int,
        col: int,
        rotation: int,
    ) -> bool:
        # Reuse the same client-side validation and API behavior as V1 without
        # copying the whole replay loop.
        from puzzlegrid.pieces import Piece
        import requests

        test_piece = Piece(piece_type, row=row, col=col, rotation=rotation)
        for r, c in test_piece.get_cells():
            if r < 0 or r >= len(board) or c < 0 or c >= len(board[0]):
                print(f"Invalid placement: cell ({r}, {c}) out of bounds")
                return False
            if board[r][c] != 0:
                print(f"Invalid placement: cell ({r}, {c}) already occupied")
                return False

        try:
            response = requests.post(
                f"{self.api_url}/games/{self.game_id}/place",
                json={"row": row, "col": col, "rotation": rotation},
                timeout=5.0,
            )
            if response.status_code == 200:
                return True

            error_detail = response.text
            try:
                error_detail = response.json().get("detail", error_detail)
            except Exception:
                pass
            print(f"Placement failed (HTTP {response.status_code}): {error_detail}")
            print(
                f"  Attempted placement: row={row}, col={col}, "
                f"rotation={rotation}, piece={piece_type_str}"
            )
            return False
        except requests.exceptions.RequestException as e:
            print(f"Network error during placement: {e}")
            return False

    def play_game(
        self,
        max_moves: int = 10000,
        move_delay: float = 0.1,
        num_samples: int = 20,
        line_clear_bonus: float = 0.0,
    ):
        self._line_clear_bonus = line_clear_bonus
        return super().play_game(
            max_moves=max_moves,
            move_delay=move_delay,
            num_samples=num_samples,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AI puzzle grid player using VAE v2 model")
    parser.add_argument(
        "--model",
        type=str,
        default="./models/vae_clean/tetris_vae_v2_best.keras",
        help="Path to trained VAE v2 model file",
    )
    parser.add_argument("--api-url", type=str, default="http://localhost:8000")
    parser.add_argument("--game-id", type=str, default=None)
    parser.add_argument("--pieces", type=str, nargs="*", default=None)
    parser.add_argument("--max-moves", type=int, default=10000)
    parser.add_argument("--move-delay", type=float, default=0.1)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Fallback prior samples used only if the full v2 model is unavailable.",
    )
    parser.add_argument(
        "--line-clear-bonus",
        type=float,
        default=0.0,
        help="Optional score bonus per simulated line clear during posterior ranking.",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PuzzleGrid AI Player (VAE v2 Model)")
    print("=" * 60)

    print(f"\nLoading model from: {args.model}")
    try:
        player = PuzzleGridVAEV2Player(args.model, args.api_url)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return

    print(f"\nCreating game via API: {args.api_url}")
    try:
        game_id = player.create_game(allowed_pieces=args.pieces, game_id=args.game_id)
        print(f"Game created: {game_id}")
    except Exception as e:
        print(f"ERROR: Failed to create game: {e}")
        return

    print("\nStarting gameplay...")
    try:
        player.play_game(
            max_moves=args.max_moves,
            move_delay=args.move_delay,
            num_samples=args.num_samples,
            line_clear_bonus=args.line_clear_bonus,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nERROR during gameplay: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
