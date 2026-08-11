#!/usr/bin/env python3
"""Play puzzle grid with the transformer placement-SCORER.

Enumerates every legal reachable placement of the current piece, scores each
with the trained scorer (train_transformers.py), and plays the best one. Ranks
by the model's predicted `score` head, optionally plus a bonus for the actually
simulated line clears. Supports the live API or in-process play, and
--allowed-pieces (e.g. single-piece games).
"""

import argparse
import time
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf

from puzzlegrid.pieces import PieceType
from puzzlegrid import placement
import transformer_features as tf_feat
from train_transformers import PositionIndicesLayer


class TransformerScorerPlayer:
    def __init__(self, model_path: str, api_url: str = "http://localhost:8000",
                 use_inprocess: bool = False, line_clear_bonus: float = 0.0):
        self.model = tf.keras.models.load_model(
            model_path, compile=False,
            custom_objects={"PositionIndicesLayer": PositionIndicesLayer})
        self.api_url = api_url
        self.use_inprocess = use_inprocess
        self.line_clear_bonus = line_clear_bonus
        self.game_id: Optional[str] = None
        self._local = None

    # ------------------------------------------------ engine plumbing
    def create_game(self, allowed_pieces: Optional[List[str]] = None,
                    piece_sequence: Optional[List[str]] = None):
        if self.use_inprocess:
            from gan_friendly_game import GANFriendlyGame
            self._local = GANFriendlyGame(width=10, height=20,
                                          piece_sequence=piece_sequence)
            self.game_id = self._local._game.game_id
            return self.game_id
        import requests
        payload = {"width": 10, "height": 20}
        if allowed_pieces:
            payload["allowed_pieces"] = allowed_pieces
        if piece_sequence:
            payload["piece_sequence"] = piece_sequence
        r = requests.post(f"{self.api_url}/games", json=payload, timeout=30)
        r.raise_for_status()
        self.game_id = r.json()["game_id"]
        return self.game_id

    def get_state(self) -> dict:
        if self.use_inprocess:
            return self._local.get_state_dict()
        import requests
        r = requests.get(f"{self.api_url}/games/{self.game_id}", timeout=30)
        r.raise_for_status()
        return r.json()

    def place(self, row: int, col: int, rotation: int) -> bool:
        if self.use_inprocess:
            _, _, done, info = self._local.step_raw(row, col, rotation)
            return "error" not in info
        import requests
        r = requests.post(f"{self.api_url}/games/{self.game_id}/place",
                          json={"row": row, "col": col, "rotation": rotation}, timeout=10)
        return r.status_code == 200

    # ------------------------------------------------ policy
    def score_candidates(self, board: List[List[int]], piece_type: PieceType):
        """Return (candidates, scores) — the model's predicted score per legal candidate."""
        candidates = placement.get_valid_placements(board, piece_type)
        if not candidates:
            return [], np.array([])
        batch = {k: [] for k in tf_feat.FEATURE_KEYS}
        sim_lines = []
        for (r, c, rot) in candidates:
            fd = tf_feat.candidate_features(board, piece_type, r, c, rot)
            for k in tf_feat.FEATURE_KEYS:
                batch[k].append(fd[k])
            if self.line_clear_bonus:
                _, lc = placement.simulate_placement(board, piece_type, r, c, rot)
                sim_lines.append(lc)
        X = {k: np.array(v) for k, v in batch.items()}
        score, _lines_pred, _holes_pred = self.model.predict(X, verbose=0)
        score = score.reshape(-1)
        if self.line_clear_bonus:
            score = score + self.line_clear_bonus * np.array(sim_lines, dtype=np.float32)
        return candidates, score

    def choose(self, board: List[List[int]], piece_type: PieceType):
        candidates, score = self.score_candidates(board, piece_type)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        return candidates[int(np.argmax(score))]

    def play(self, max_moves: int = 400, delay: float = 0.0) -> dict:
        for _ in range(max_moves):
            st = self.get_state()
            if st.get("status") != "playing":
                break
            cur = st.get("current_piece")
            if not cur:
                break
            piece_type = PieceType[cur["type"]]
            choice = self.choose(st["board"], piece_type)
            if choice is None:
                break
            if not self.place(*choice):
                break
            if delay > 0:
                time.sleep(delay)
        return self.get_state()


def main():
    ap = argparse.ArgumentParser(description="Replay with transformer placement-scorer")
    ap.add_argument("--model", default="./models/transformer_scorer/transformer_scorer_best.keras")
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--inprocess", action="store_true")
    ap.add_argument("--allowed-pieces", type=str, nargs="*", default=None)
    ap.add_argument("--piece-sequence", type=str, nargs="*", default=None)
    ap.add_argument("--line-clear-bonus", type=float, default=0.0)
    ap.add_argument("--max-moves", type=int, default=400)
    ap.add_argument("--delay", type=float, default=0.1, help="Seconds between moves (browser pace)")
    args = ap.parse_args()

    player = TransformerScorerPlayer(
        args.model, api_url=args.api_url, use_inprocess=args.inprocess,
        line_clear_bonus=args.line_clear_bonus)
    player.create_game(allowed_pieces=args.allowed_pieces,
                       piece_sequence=args.piece_sequence)
    final = player.play(max_moves=args.max_moves, delay=args.delay)
    print(f"Lines cleared: {final.get('lines_cleared', 0)}")
    print(f"Pieces placed: {final.get('pieces_placed', 0)}")
    print(f"Score: {final.get('score', 0)}")
    print(f"Status: {final.get('status')}")


if __name__ == "__main__":
    main()
