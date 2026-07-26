#!/usr/bin/env python3
"""AI player using a trained VAE v2 model with independent placement selection.

Independent VAE v2 replay — model-chosen placement, no engine candidate filtering.

Unlike ``replay_vae_v2.py``, this player does not enumerate legal placements via
the game engine. The decoder chooses row, col, and rotation directly from prior
samples (optionally refined through the v2 action-conditioned posterior). Invalid
decodes are skipped and the next sample is tried immediately. If all samples fail,
an optional engine fallback keeps the game moving.
"""

from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from puzzlegrid.pieces import Piece, PieceType

from replay_vae_v2 import PuzzleGridVAEV2Player


class PuzzleGridVAEV2IndependentPlayer(PuzzleGridVAEV2Player):
    """PuzzleGrid player using direct VAE v2 decoder output (no engine filtering)."""

    @staticmethod
    def _is_valid_placement(
        board: List[List[int]],
        piece_type: PieceType,
        row: int,
        col: int,
        rotation: int,
    ) -> bool:
        test_piece = Piece(piece_type, row=row, col=col, rotation=rotation)
        for r, c in test_piece.get_cells():
            if r < 0 or r >= len(board) or c < 0 or c >= len(board[0]):
                return False
            if board[r][c] != 0:
                return False
        return True

    def _decode_candidate(
        self,
        board_feat: np.ndarray,
        z: np.ndarray,
        piece_type: PieceType,
        board_height: int,
        board_width: int,
    ) -> Tuple[int, int, int]:
        row, col, rotation = self._decode_placement(
            board_feat, z, piece_type, board_height, board_width
        )
        if self.full_model is not None:
            row, col, rotation = self._refine_with_posterior(
                board_feat,
                piece_type,
                row,
                col,
                rotation,
                board_height,
                board_width,
            )
        return row, col, rotation

    def _refine_with_posterior(
        self,
        board_feat: np.ndarray,
        piece_type: PieceType,
        row: int,
        col: int,
        rotation: int,
        board_height: int,
        board_width: int,
    ) -> Tuple[int, int, int]:
        """Refine a prior decode using q(z | board, model-predicted action)."""
        piece_idx = self.piece_type_map[piece_type.name]
        position = np.array(
            [[row / board_height, col / board_width]], dtype=np.float32
        )
        piece_onehot = self._piece_onehot(piece_type)

        board_feat_tensor = tf.convert_to_tensor(
            board_feat.astype(np.float32), dtype=tf.float32
        )
        action_vector = self.full_model.build_action_vector(
            tf.convert_to_tensor([piece_idx], dtype=tf.int32),
            tf.convert_to_tensor([rotation], dtype=tf.int32),
            tf.convert_to_tensor(position, dtype=tf.float32),
        )
        z_mean, _, _ = self.full_model.encode(
            board_feat_tensor, action_vector, training=False
        )
        piece_logits, rotation_logits, position_pred = self.full_model.decode(
            board_feat_tensor,
            z_mean,
            tf.convert_to_tensor(piece_onehot, dtype=tf.float32),
            training=False,
        )

        rotation = int(np.argmax(rotation_logits[0].numpy()))
        row = int(round(float(position_pred[0, 0].numpy()) * board_height))
        col = int(round(float(position_pred[0, 1].numpy()) * board_width))
        row = max(0, min(board_height - 1, row))
        col = max(0, min(board_width - 1, col))
        return row, col, rotation

    def _try_model_placements(
        self,
        board: List[List[int]],
        board_feat: np.ndarray,
        piece_type: PieceType,
        piece_type_str: str,
        num_samples: int,
    ) -> bool:
        """Try up to num_samples model decodes; submit the first valid one."""
        board_height = len(board)
        board_width = len(board[0])

        z_zero = np.zeros((1, self.latent_dim), dtype=np.float32)
        for i in range(num_samples):
            z = z_zero if i == num_samples - 1 else np.random.randn(
                1, self.latent_dim
            ).astype(np.float32)
            row, col, rotation = self._decode_candidate(
                board_feat, z, piece_type, board_height, board_width
            )
            if not self._is_valid_placement(board, piece_type, row, col, rotation):
                continue
            if self._submit_move(
                board, piece_type, piece_type_str, row, col, rotation, quiet=True
            ):
                return True
        return False

    def choose_best_placement(
        self,
        board: List[List[int]],
        piece_type: PieceType,
        num_samples: int = 20,
        **kwargs,
    ) -> Optional[Tuple[int, int, int]]:
        board_height = len(board)
        board_width = len(board[0])
        board_feat = self.board_encoder.predict(
            self._prepare_board(board), verbose=0
        )

        best_placement = None
        best_error = float("inf")
        z_zero = np.zeros((1, self.latent_dim), dtype=np.float32)

        for i in range(num_samples):
            z = z_zero if i == num_samples - 1 else np.random.randn(
                1, self.latent_dim
            ).astype(np.float32)
            row, col, rotation = self._decode_candidate(
                board_feat, z, piece_type, board_height, board_width
            )
            if not self._is_valid_placement(board, piece_type, row, col, rotation):
                continue
            error = self._reconstruction_error(
                board_feat,
                piece_type,
                row,
                col,
                rotation,
                z,
                board_height,
                board_width,
            )
            if error < best_error:
                best_error = error
                best_placement = (row, col, rotation)

        return best_placement

    def _submit_move(
        self,
        board: List[List[int]],
        piece_type: PieceType,
        piece_type_str: str,
        row: int,
        col: int,
        rotation: int,
        quiet: bool = False,
    ) -> bool:
        import requests

        test_piece = Piece(piece_type, row=row, col=col, rotation=rotation)
        for r, c in test_piece.get_cells():
            if r < 0 or r >= len(board) or c < 0 or c >= len(board[0]):
                if not quiet:
                    print(f"Invalid placement: cell ({r}, {c}) out of bounds")
                return False
            if board[r][c] != 0:
                if not quiet:
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

            if not quiet:
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
            if not quiet:
                print(f"Network error during placement: {e}")
            return False

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
        num_samples = getattr(self, "_num_samples", 30)
        engine_fallback = getattr(self, "_engine_fallback", True)

        board_feat = self.board_encoder.predict(
            self._prepare_board(board), verbose=0
        )
        if self._try_model_placements(
            board, board_feat, piece_type, piece_type_str, num_samples
        ):
            return True

        if engine_fallback:
            print(
                f"No valid model placement after {num_samples} samples for "
                f"{piece_type_str}; using engine-ranked fallback"
            )
            placement = super().choose_best_placement(
                board, piece_type, num_samples=num_samples
            )
            if placement:
                row, col, rotation = placement
                return self._submit_move(
                    board, piece_type, piece_type_str, row, col, rotation
                )

        print(f"Could not place piece {piece_type_str}")
        return False

    def play_game(
        self,
        max_moves: int = 10000,
        move_delay: float = 0.1,
        num_samples: int = 30,
        engine_fallback: bool = True,
    ):
        self._engine_fallback = engine_fallback
        return super(PuzzleGridVAEV2Player, self).play_game(
            max_moves=max_moves,
            move_delay=move_delay,
            num_samples=num_samples,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AI puzzle grid player using independent VAE v2 model decode"
    )
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
        default=30,
        help="Model decode attempts per move before optional engine fallback.",
    )
    parser.add_argument(
        "--no-engine-fallback",
        action="store_true",
        help="Stop the move if the model cannot produce a valid placement.",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PuzzleGrid AI Player (Independent VAE v2 Model)")
    print("=" * 60)

    print(f"\nLoading model from: {args.model}")
    try:
        player = PuzzleGridVAEV2IndependentPlayer(args.model, args.api_url)
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

    fallback_msg = (
        "engine fallback enabled"
        if not args.no_engine_fallback
        else "no engine fallback"
    )
    print(f"\nStarting gameplay (model-chosen placements, {fallback_msg})...")
    try:
        player.play_game(
            max_moves=args.max_moves,
            move_delay=args.move_delay,
            num_samples=args.num_samples,
            engine_fallback=not args.no_engine_fallback,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nERROR during gameplay: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
