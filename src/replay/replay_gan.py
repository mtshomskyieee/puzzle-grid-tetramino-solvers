#!/usr/bin/env python3
"""Replay player using trained Tetris move GAN.

Loads generator/discriminator, samples candidate moves, snaps to legal placements,
and picks the best by combined discriminator score and simulated line clears.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import tensorflow as tf

from gan_friendly_game import (
    GANFriendlyGame,
    encode_board_tensor,
    get_valid_placements,
    piece_type_to_idx,
    simulate_placement_detailed,
    snap_to_valid,
)
from puzzlegrid.pieces import PieceType


def load_gan_bundle(model_dir: str) -> Tuple[tf.keras.Model, tf.keras.Model, dict]:
    """Load generator, discriminator, and config from model directory."""
    model_dir = os.path.abspath(model_dir)
    config_path = os.path.join(model_dir, "tetris_gan_config.json")

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        gen_path = config.get("generator_path", "")
        disc_path = config.get("discriminator_path", "")
        if gen_path and not os.path.isabs(gen_path):
            gen_path = os.path.join(model_dir, os.path.basename(gen_path))
        if disc_path and not os.path.isabs(disc_path):
            disc_path = os.path.join(model_dir, os.path.basename(disc_path))
    else:
        config = {
            "board_height": 20,
            "board_width": 10,
            "noise_dim": 32,
        }
        gen_path = os.path.join(model_dir, "tetris_gan_generator.keras")
        disc_path = os.path.join(model_dir, "tetris_gan_discriminator.keras")

    if not os.path.exists(gen_path) or not os.path.exists(disc_path):
        raise FileNotFoundError(
            f"GAN models not found in {model_dir}. "
            "Run training_gan.py first."
        )

    generator = tf.keras.models.load_model(gen_path, compile=False)
    discriminator = tf.keras.models.load_model(disc_path, compile=False)
    return generator, discriminator, config


def build_action_batch(
    piece_idx: int,
    rotation: int,
    row: int,
    col: int,
    board_height: int,
    board_width: int,
) -> np.ndarray:
    piece_onehot = np.zeros((1, 7), dtype=np.float32)
    piece_onehot[0, piece_idx] = 1.0
    rot_onehot = np.zeros((1, 4), dtype=np.float32)
    rot_onehot[0, rotation % 4] = 1.0
    pos = np.array(
        [[row / board_height, col / board_width]], dtype=np.float32
    )
    return np.concatenate([piece_onehot, rot_onehot, pos], axis=-1)


class PuzzleGridGANPlayer:
    """Select placements via GAN generator + discriminator ranking."""

    def __init__(
        self,
        model_dir: str,
        api_url: str = "http://localhost:8000",
        num_samples: int = 8,
        line_clear_bonus: float = 1.5,
        use_inprocess: bool = False,
    ):
        self.generator, self.discriminator, self.config = load_gan_bundle(model_dir)
        self.api_url = api_url
        self.num_samples = num_samples
        self.line_clear_bonus = line_clear_bonus
        self.use_inprocess = use_inprocess
        self.game_id: Optional[str] = None
        self._local_game: Optional[GANFriendlyGame] = None

        self.board_height = int(self.config.get("board_height", 20))
        self.board_width = int(self.config.get("board_width", 10))
        self.noise_dim = int(self.config.get("noise_dim", 32))

    def create_game(
        self,
        width: int = 10,
        height: int = 20,
        allowed_pieces: Optional[List[str]] = None,
        piece_sequence: Optional[List[str]] = None,
    ) -> str:
        if self.use_inprocess:
            self._local_game = GANFriendlyGame(
                width=width,
                height=height,
                piece_sequence=piece_sequence,
            )
            self.game_id = self._local_game._game.game_id
            return self.game_id

        payload: Dict = {"width": width, "height": height}
        if allowed_pieces:
            payload["allowed_pieces"] = allowed_pieces
        if piece_sequence:
            payload["piece_sequence"] = piece_sequence

        r = requests.post(f"{self.api_url}/games", json=payload, timeout=30)
        r.raise_for_status()
        self.game_id = r.json()["game_id"]
        return self.game_id

    def get_game_state(self) -> dict:
        if self.use_inprocess and self._local_game:
            return self._local_game.get_state_dict()

        if not self.game_id:
            raise ValueError("No game created")
        r = requests.get(f"{self.api_url}/games/{self.game_id}", timeout=30)
        r.raise_for_status()
        return r.json()

    def _score_placement(
        self,
        board: List[List[int]],
        piece_type: PieceType,
        row: int,
        col: int,
        rotation: int,
    ) -> Tuple[float, int, Tuple[int, int, int]]:
        """Return (score, lines_cleared, snapped_placement). Lower score is better for CE; we use negative realism."""
        board_h = len(board)
        board_w = len(board[0])
        piece_idx = piece_type_to_idx(piece_type)

        valid = get_valid_placements(board, piece_type)
        snapped, snap_dist = snap_to_valid(row, col, rotation, valid)
        if snapped is None:
            return float("inf"), 0, (row, col, rotation)

        srow, scol, srot = snapped
        sim = simulate_placement_detailed(
            board, piece_type, srow, scol, srot, snap=False
        )
        lines = sim.lines_cleared if sim.valid else 0

        board_in = encode_board_tensor(board)
        piece_oh = np.zeros((1, 7), dtype=np.float32)
        piece_oh[0, piece_idx] = 1.0
        action = build_action_batch(piece_idx, srot, srow, scol, board_h, board_w)

        if sim.valid:
            post_in = encode_board_tensor(sim.post_board)
            realism, lc_logits = self.discriminator.predict(
                [board_in, post_in, action], verbose=0
            )
        else:
            realism, lc_logits = self.discriminator.predict(
                [board_in, board_in, action], verbose=0
            )

        realism_score = float(realism[0, 0])
        lc_pred = float(np.argmax(lc_logits[0]))

        # Higher is better -> negate for argmin, or use negative combined
        combined = realism_score - self.line_clear_bonus * lines - 0.3 * lc_pred
        combined += 0.1 * snap_dist
        return combined, lines, (srow, scol, srot)

    def choose_best_placement(
        self,
        board: List[List[int]],
        piece_type: PieceType,
        mode: str = "search",
    ) -> Optional[Tuple[int, int, int]]:
        valid = get_valid_placements(board, piece_type)
        if not valid:
            return None
        if len(valid) == 1:
            return valid[0]

        board_h = len(board)
        board_w = len(board[0])
        piece_idx = piece_type_to_idx(piece_type)
        board_in = encode_board_tensor(board)
        piece_oh = np.zeros((1, 7), dtype=np.float32)
        piece_oh[0, piece_idx] = 1.0

        candidates: List[Tuple[int, int, int]] = []

        if mode == "fast":
            noise = np.random.randn(1, self.noise_dim).astype(np.float32)
            rot_logits, pos = self.generator.predict(
                [board_in, piece_oh, noise], verbose=0
            )
            rotation = int(np.argmax(rot_logits[0]))
            row = int(round(pos[0, 0] * board_h))
            col = int(round(pos[0, 1] * board_w))
            row = max(0, min(board_h - 1, row))
            col = max(0, min(board_w - 1, col))
            snapped, _ = snap_to_valid(row, col, rotation, valid)
            if snapped:
                candidates.append(snapped)
        else:
            # Search mode should evaluate the full legal move set so line clears are
            # not missed due to low stochastic sample coverage.
            candidates.extend(valid)
            for _ in range(self.num_samples):
                noise = np.random.randn(1, self.noise_dim).astype(np.float32)
                rot_logits, pos = self.generator.predict(
                    [board_in, piece_oh, noise], verbose=0
                )
                rotation = int(np.argmax(rot_logits[0]))
                row = int(round(pos[0, 0] * board_h))
                col = int(round(pos[0, 1] * board_w))
                row = max(0, min(board_h - 1, row))
                col = max(0, min(board_w - 1, col))
                snapped, _ = snap_to_valid(row, col, rotation, valid)
                if snapped and snapped not in candidates:
                    candidates.append(snapped)

        if not candidates:
            candidates = valid[: min(len(valid), self.num_samples)]

        best_placement = None
        best_score = float("inf")
        best_lines = -1

        for row, col, rot in candidates:
            score, lines, placement = self._score_placement(
                board, piece_type, row, col, rot
            )
            if lines > best_lines or (lines == best_lines and score < best_score):
                best_lines = lines
                best_score = score
                best_placement = placement

        return best_placement

    def place_move(self, row: int, col: int, rotation: int) -> bool:
        if self.use_inprocess and self._local_game:
            _, _, done, info = self._local_game.step_raw(row, col, rotation)
            return info.get("placement_ok", False) and not info.get("error")

        if not self.game_id:
            return False
        r = requests.post(
            f"{self.api_url}/games/{self.game_id}/place",
            json={"row": row, "col": col, "rotation": rotation},
            timeout=30,
        )
        return r.status_code == 200

    def make_move(self, mode: str = "search") -> bool:
        state = self.get_game_state()
        if state["status"] != "playing":
            return False

        current = state.get("current_piece")
        if not current:
            return False

        piece_type = PieceType[current["type"]]
        board = state["board"]
        placement = self.choose_best_placement(board, piece_type, mode=mode)
        if placement is None:
            return False

        row, col, rotation = placement
        return self.place_move(row, col, rotation)

    def play_game(
        self,
        max_moves: int = 500,
        delay: float = 0.0,
        mode: str = "search",
    ) -> dict:
        moves = 0
        while moves < max_moves:
            state = self.get_game_state()
            if state["status"] != "playing":
                break
            if not self.make_move(mode=mode):
                break
            moves += 1
            if delay > 0:
                time.sleep(delay)

        return self.get_game_state()


def main():
    parser = argparse.ArgumentParser(description="Replay with Tetris GAN player")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models/gan_clean_v2",
        help="Directory with GAN artifacts",
    )
    parser.add_argument("--api-url", type=str, default="http://localhost:8000")
    parser.add_argument("--inprocess", action="store_true", help="Use GANFriendlyGame locally")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--line-clear-bonus", type=float, default=1.5)
    parser.add_argument("--mode", choices=["fast", "search"], default="search")
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.1, help="Seconds between moves (browser pace)")
    parser.add_argument("--piece-sequence", type=str, nargs="*", default=None)
    args = parser.parse_args()

    player = PuzzleGridGANPlayer(
        args.model_dir,
        api_url=args.api_url,
        num_samples=args.num_samples,
        line_clear_bonus=args.line_clear_bonus,
        use_inprocess=args.inprocess,
    )

    player.create_game(piece_sequence=args.piece_sequence)
    print(f"Game {player.game_id} started (mode={args.mode})")

    final = player.play_game(max_moves=args.max_moves, mode=args.mode, delay=args.delay)
    print(f"Final score: {final.get('score', 0)}")
    print(f"Lines cleared: {final.get('lines_cleared', 0)}")
    print(f"Status: {final.get('status')}")


if __name__ == "__main__":
    main()
