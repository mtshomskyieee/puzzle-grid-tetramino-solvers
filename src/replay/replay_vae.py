#!/usr/bin/env python3
"""AI player that uses a trained VAE model to play puzzle grid games via the API.

Hybrid inference: sample multiple latent vectors, decode placements, filter to
valid positions, and pick the candidate with the lowest reconstruction error.
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import tensorflow as tf
from puzzlegrid.board import Board
from puzzlegrid.pieces import Piece, PieceType
from puzzlegrid import placement

from training_vae_v1 import SamplingLayer, TetrisVAE


def load_vae_for_inference(model_path: str) -> Tuple[tf.keras.Model, tf.keras.Model, int]:
    """Load board encoder and decoder for VAE inference."""
    model_dir = os.path.dirname(os.path.abspath(model_path))
    config_path = os.path.join(model_dir, "tetris_vae_v1_config.json")

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
        latent_dim = 32
        encoder_path = os.path.join(model_dir, "tetris_vae_v1_board_encoder.keras")
        decoder_path = os.path.join(model_dir, "tetris_vae_v1_decoder.keras")

    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        custom_objects = {"SamplingLayer": SamplingLayer, "TetrisVAE": TetrisVAE}
        try:
            model = tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects=custom_objects,
                safe_mode=False,
            )
            if isinstance(model, TetrisVAE):
                return model.board_encoder, model.decoder, model.latent_dim
        except Exception:
            pass
        raise ValueError(
            f"Could not load VAE inference bundle from {model_dir}. "
            "Expected tetris_vae_v1_board_encoder.keras and tetris_vae_v1_decoder.keras."
        )

    board_encoder = tf.keras.models.load_model(encoder_path, compile=False)
    decoder = tf.keras.models.load_model(decoder_path, compile=False)
    return board_encoder, decoder, latent_dim


class PuzzleGridVAEPlayer:
    """AI player that uses a conditional VAE to play puzzle grid games."""

    def __init__(self, model_path: str, api_url: str = "http://localhost:8000"):
        self.api_url = api_url

        try:
            self.board_encoder, self.decoder, self.latent_dim = load_vae_for_inference(
                model_path
            )
        except Exception as e:
            raise ValueError(
                f"Failed to load model from {model_path}: {e}\n"
                "Make sure the model file exists and inference bundle files are present."
            ) from e

        self.game_id = None

        self.piece_type_map = {pt.name: idx for idx, pt in enumerate(PieceType)}
        self.piece_type_reverse = {idx: pt for pt, idx in self.piece_type_map.items()}

    def create_game(
        self,
        width: int = 10,
        height: int = 20,
        allowed_pieces: Optional[List[str]] = None,
        game_id: Optional[str] = None,
    ) -> str:
        payload = {"width": width, "height": height}
        if allowed_pieces:
            payload["allowed_pieces"] = allowed_pieces
        if game_id:
            payload["game_id"] = game_id

        response = requests.post(f"{self.api_url}/games", json=payload)
        response.raise_for_status()
        game_data = response.json()
        self.game_id = game_data["game_id"]
        return self.game_id

    def get_game_state(self) -> Dict:
        if not self.game_id:
            raise ValueError("No game created. Call create_game() first.")

        response = requests.get(f"{self.api_url}/games/{self.game_id}")
        response.raise_for_status()
        return response.json()

    def _can_reach_position(
        self,
        board: List[List[int]],
        piece_type: PieceType,
        row: int,
        col: int,
        rotation: int,
    ) -> bool:
        # Delegates to the canonical BFS reachability check.
        return placement.can_reach_placement(board, piece_type, row, col, rotation)

    def get_valid_placements(
        self, board: List[List[int]], piece_type: PieceType
    ) -> List[Tuple[int, int, int]]:
        # Canonical reachable placements, then the low-stack shaping policy this
        # replay path has always applied (prefer keeping the stack low).
        valid = placement.get_valid_placements(board, piece_type)
        return placement.filter_low_stack(board, piece_type, valid)

    def simulate_placement(
        self,
        board_state: List[List[int]],
        piece_type: PieceType,
        row: int,
        col: int,
        rotation: int,
    ) -> Tuple[List[List[int]], int]:
        return placement.simulate_placement(board_state, piece_type, row, col, rotation)

    def _prepare_board(self, board: List[List[int]]) -> np.ndarray:
        board_2d = np.array(board, dtype=np.float32)
        board_2d = (board_2d > 0).astype(np.float32)
        return board_2d[np.newaxis, ..., np.newaxis]

    def _piece_onehot(self, piece_type: PieceType) -> np.ndarray:
        onehot = np.zeros((1, 7), dtype=np.float32)
        onehot[0, self.piece_type_map[piece_type.name]] = 1.0
        return onehot

    def _decode_placement(
        self,
        board_feat: np.ndarray,
        z: np.ndarray,
        piece_type: PieceType,
        board_height: int,
        board_width: int,
    ) -> Tuple[int, int, int]:
        piece_onehot = self._piece_onehot(piece_type)
        piece_logits, rotation_logits, position_pred = self.decoder.predict(
            [board_feat, z, piece_onehot], verbose=0
        )

        rotation = int(np.argmax(rotation_logits[0]))
        row = int(round(position_pred[0, 0] * board_height))
        col = int(round(position_pred[0, 1] * board_width))

        row = max(0, min(board_height - 1, row))
        col = max(0, min(board_width - 1, col))
        return row, col, rotation

    def _snap_to_valid(
        self,
        row: int,
        col: int,
        rotation: int,
        valid_placements: List[Tuple[int, int, int]],
    ) -> Optional[Tuple[int, int, int]]:
        if (row, col, rotation) in valid_placements:
            return (row, col, rotation)

        if not valid_placements:
            return None

        best = None
        best_dist = float("inf")
        for vr, vc, vrot in valid_placements:
            dist = abs(vr - row) + abs(vc - col) + (0 if vrot == rotation else 2)
            if dist < best_dist:
                best_dist = dist
                best = (vr, vc, vrot)
        return best

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
        piece_ce = -piece_logits[0, piece_idx]
        rotation_ce = -rotation_logits[0, rotation]

        target_pos = np.array(
            [[row / board_height, col / board_width]], dtype=np.float32
        )
        position_mse = float(np.mean((position_pred - target_pos) ** 2))

        return float(piece_ce + rotation_ce + position_mse)

    def choose_best_placement(
        self,
        board: List[List[int]],
        piece_type: PieceType,
        num_samples: int = 20,
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

        best_placement = None
        best_error = float("inf")

        for _ in range(num_samples):
            z = np.random.randn(1, self.latent_dim).astype(np.float32)
            row, col, rotation = self._decode_placement(
                board_feat, z, piece_type, board_height, board_width
            )
            snapped = self._snap_to_valid(row, col, rotation, valid_placements)
            if snapped is None:
                continue

            sr, sc, srot = snapped
            error = self._reconstruction_error(
                board_feat, piece_type, sr, sc, srot, z, board_height, board_width
            )
            if error < best_error:
                best_error = error
                best_placement = snapped

        if best_placement is not None:
            return best_placement

        z_zero = np.zeros((1, self.latent_dim), dtype=np.float32)
        row, col, rotation = self._decode_placement(
            board_feat, z_zero, piece_type, board_height, board_width
        )
        return self._snap_to_valid(row, col, rotation, valid_placements)

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
        placement = self.choose_best_placement(
            board, piece_type, num_samples=num_samples
        )
        if not placement:
            print(f"No valid placement found for piece {piece_type_str}")
            return False

        row, col, rotation = placement

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

    def play_game(self, max_moves: int = 10000, move_delay: float = 0.1, num_samples: int = 20):
        if not self.game_id:
            raise ValueError("No game created. Call create_game() first.")

        self._num_samples = num_samples
        print(f"Starting game: {self.game_id}")
        print(f"View at: {self.api_url}/games/{self.game_id}/view")
        print("-" * 60)

        self._num_samples = num_samples
        move_count = 0
        while move_count < max_moves:
            state = self.get_game_state()

            if state["status"] != "playing":
                print(f"\nGame over! Status: {state['status']}")
                print(f"Final score: {state['score']}")
                print(f"Lines cleared: {state['lines_cleared']}")
                print(f"Pieces placed: {state['pieces_placed']}")
                break

            if not self.make_move():
                state = self.get_game_state()
                if state["status"] != "playing":
                    print(f"\nGame over! Status: {state['status']}")
                    print(f"Final score: {state['score']}")
                    print(f"Lines cleared: {state['lines_cleared']}")
                    print(f"Pieces placed: {state['pieces_placed']}")
                    break
                print("Move failed but game still playing")
                break

            move_count += 1
            if move_count % 10 == 0:
                print(
                    f"Moves: {move_count}, Score: {state['score']}, "
                    f"Lines: {state['lines_cleared']}"
                )
            time.sleep(move_delay)

        if move_count >= max_moves:
            print(f"\nReached maximum moves ({max_moves})")

        final_state = self.get_game_state()
        print("\nFinal stats:")
        print(f"  Score: {final_state['score']}")
        print(f"  Lines cleared: {final_state['lines_cleared']}")
        print(f"  Pieces placed: {final_state['pieces_placed']}")
        print(f"\nView game at: {self.api_url}/games/{self.game_id}/view")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AI puzzle grid player using VAE model")
    parser.add_argument(
        "--model",
        type=str,
        default="./models/tetris_vae_v1_best.keras",
        help="Path to trained VAE model file",
    )
    parser.add_argument("--api-url", type=str, default="http://localhost:8000")
    parser.add_argument("--game-id", type=str, default=None)
    parser.add_argument("--pieces", type=str, nargs="*", default=None)
    parser.add_argument("--max-moves", type=int, default=10000)
    parser.add_argument("--move-delay", type=float, default=0.1)
    parser.add_argument("--num-samples", type=int, default=20)

    args = parser.parse_args()

    print("=" * 60)
    print("PuzzleGrid AI Player (VAE Model)")
    print("=" * 60)

    print(f"\nLoading model from: {args.model}")
    try:
        player = PuzzleGridVAEPlayer(args.model, args.api_url)
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
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nERROR during gameplay: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
