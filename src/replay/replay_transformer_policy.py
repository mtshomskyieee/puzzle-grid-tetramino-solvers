#!/usr/bin/env python3
"""Play puzzle grid with the distilled direct-policy transformer.

Unlike the scorer (which ranks all candidates), this student DIRECTLY predicts a
rotation and a column from the board+piece — a generator, like the VAE/GAN. The
predicted (rotation, col) is snapped to the nearest legal placement and played.
Supports the live API, in-process play, and --allowed-pieces.
"""

import argparse
from typing import List, Optional

import numpy as np
import tensorflow as tf

from puzzlegrid.pieces import PieceType
from puzzlegrid import placement
import transformer_features as tf_feat
from train_transformers import PositionIndicesLayer


class TransformerPolicyPlayer:
    def __init__(self, model_path: str, api_url: str = "http://localhost:8000",
                 use_inprocess: bool = False):
        self.model = tf.keras.models.load_model(
            model_path, compile=False,
            custom_objects={"PositionIndicesLayer": PositionIndicesLayer})
        self.api_url = api_url
        self.use_inprocess = use_inprocess
        self.game_id: Optional[str] = None
        self._local = None

    # ------------------------------------------------ engine plumbing
    def create_game(self, allowed_pieces=None, piece_sequence=None):
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

    def place(self, row, col, rotation) -> bool:
        if self.use_inprocess:
            _, _, _done, info = self._local.step_raw(row, col, rotation)
            return "error" not in info
        import requests
        r = requests.post(f"{self.api_url}/games/{self.game_id}/place",
                          json={"row": row, "col": col, "rotation": rotation}, timeout=10)
        return r.status_code == 200

    # ------------------------------------------------ policy
    def choose(self, board: List[List[int]], piece_type: PieceType):
        """Directly predict (rotation, col), then snap to the nearest legal move."""
        fd = tf_feat.state_features(board, piece_type)
        X = {k: fd[k][np.newaxis, ...] for k in tf_feat.STATE_FEATURE_KEYS}
        rot_logits, col_logits = self.model.predict(X, verbose=0)
        rotation = int(np.argmax(rot_logits[0]))
        col = int(np.argmax(col_logits[0]))
        candidates = placement.get_valid_placements(board, piece_type)
        return tf_feat.snap_to_placement(candidates, rotation, col)

    def play(self, max_moves: int = 400) -> dict:
        for _ in range(max_moves):
            st = self.get_state()
            if st.get("status") != "playing":
                break
            cur = st.get("current_piece")
            if not cur:
                break
            choice = self.choose(st["board"], PieceType[cur["type"]])
            if choice is None:
                break
            if not self.place(*choice):
                break
        return self.get_state()


def main():
    ap = argparse.ArgumentParser(description="Replay with distilled direct-policy transformer")
    ap.add_argument("--model", default="./models/transformer_policy_soft/transformer_policy_best.keras")
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--inprocess", action="store_true")
    ap.add_argument("--allowed-pieces", type=str, nargs="*", default=None)
    ap.add_argument("--piece-sequence", type=str, nargs="*", default=None)
    ap.add_argument("--max-moves", type=int, default=400)
    args = ap.parse_args()

    player = TransformerPolicyPlayer(args.model, api_url=args.api_url,
                                     use_inprocess=args.inprocess)
    player.create_game(allowed_pieces=args.allowed_pieces,
                       piece_sequence=args.piece_sequence)
    final = player.play(max_moves=args.max_moves)
    print(f"Lines cleared: {final.get('lines_cleared', 0)}")
    print(f"Pieces placed: {final.get('pieces_placed', 0)}")
    print(f"Score: {final.get('score', 0)}")
    print(f"Status: {final.get('status')}")


if __name__ == "__main__":
    main()
