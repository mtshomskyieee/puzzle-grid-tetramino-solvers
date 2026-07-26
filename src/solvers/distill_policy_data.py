#!/usr/bin/env python3
"""Generate policy-distillation data by letting the transformer SCORER self-play.

The scorer (teacher) is a strong player (clears 100+ lines). We record, at every
step of its games, the board it saw, the current piece, and the placement it
chose. Training a direct-policy student on these (board, piece) -> placement pairs
is on-policy imitation: the student learns to mimic a strong policy on the board
states that policy actually reaches, which is what avoids the covariate shift that
sank the VAE/GAN generators.

Output: JSON files in --output-dir, each {"records": [{board_before, piece_type,
row, col, rotation}, ...]}.
"""

import argparse
import json
import os

import numpy as np

from puzzlegrid.pieces import PieceType
from replay_transformers import TransformerScorerPlayer


def _softmax_masked(values: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax over entries that have a candidate (others get prob 0)."""
    mask = values > -1e8
    if mask.sum() == 0:
        return np.ones_like(values) / len(values)
    v = values / temperature
    v = np.where(mask, v, -np.inf)
    v = v - np.max(v[mask])
    e = np.where(mask, np.exp(v), 0.0)
    return e / e.sum()


def soft_targets(candidates, scores, width=10, temperature=0.5):
    """Teacher's preference distribution over columns and rotations.

    For each column/rotation, take the best candidate score, then softmax. This is
    the soft (Hinton) distillation target — the teacher's relative judgment over all
    options, not just its top pick.
    """
    col_best = np.full(width, -1e9, dtype=np.float64)
    rot_best = np.full(4, -1e9, dtype=np.float64)
    for (r, c, rot), s in zip(candidates, scores):
        c = min(max(int(c), 0), width - 1)
        col_best[c] = max(col_best[c], float(s))
        rot_best[int(rot) % 4] = max(rot_best[int(rot) % 4], float(s))
    return (_softmax_masked(col_best, temperature).tolist(),
            _softmax_masked(rot_best, temperature).tolist())


def main():
    ap = argparse.ArgumentParser(description="Scorer self-play -> policy distillation data")
    ap.add_argument("--scorer", default="./models/transformer_scorer/transformer_scorer_best.keras")
    ap.add_argument("--output-dir", default="./data/policy_data")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--max-moves", type=int, default=200)
    ap.add_argument("--line-clear-bonus", type=float, default=0.0)
    ap.add_argument("--student", default=None,
                    help="DAgger: path to a policy model that DRIVES the trajectory. "
                         "The scorer teacher still provides the recorded label, so we "
                         "collect expert labels on states the STUDENT visits.")
    ap.add_argument("--tag", default="distill", help="filename prefix for output games")
    ap.add_argument("--temperature", type=float, default=0.5,
                    help="softmax temperature for soft distillation targets (lower = sharper)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    teacher = TransformerScorerPlayer(
        args.scorer, use_inprocess=True, line_clear_bonus=args.line_clear_bonus)

    student = None
    if args.student:
        from replay_transformer_policy import TransformerPolicyPlayer
        student = TransformerPolicyPlayer(args.student, use_inprocess=True)
        print(f"DAgger mode: student {args.student} drives, teacher labels.")

    total_records = 0
    for g in range(args.games):
        teacher.create_game()  # fresh in-process game (mixed pieces)
        records = []
        for _ in range(args.max_moves):
            st = teacher.get_state()
            if st.get("status") != "playing":
                break
            cur = st.get("current_piece")
            if not cur:
                break
            piece_type = PieceType[cur["type"]]
            board = st["board"]

            # Teacher scores every candidate -> hard argmax label + soft target dists.
            candidates, scores = teacher.score_candidates(board, piece_type)
            if len(candidates) == 0:
                break
            best_i = int(np.argmax(scores))
            label = candidates[best_i]
            row, col, rot = label
            col_probs, rot_probs = soft_targets(candidates, scores,
                                                width=len(board[0]),
                                                temperature=args.temperature)
            records.append({
                "board_before": board,
                "piece_type": piece_type.name,
                "row": int(row), "col": int(col), "rotation": int(rot),
                "col_probs": col_probs,
                "rot_probs": rot_probs,
            })

            # Trajectory is driven by the student in DAgger mode, else the teacher.
            if student is not None:
                move = student.choose(board, piece_type) or label
            else:
                move = label
            if not teacher.place(*move):
                break

        out = os.path.join(args.output_dir, f"{args.tag}_game_{g:04d}.json")
        with open(out, "w") as f:
            json.dump({"records": records}, f)
        total_records += len(records)
        print(f"game {g+1}/{args.games}: {len(records)} records "
              f"(lines={st.get('lines_cleared', 0)})")

    print(f"\nTotal records: {total_records:,} across {args.games} games -> {args.output_dir}")


if __name__ == "__main__":
    main()
