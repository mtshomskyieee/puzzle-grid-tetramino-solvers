#!/usr/bin/env python3
"""Shared feature/label logic for the transformer placement-scorer.

Used by both train_transformers.py and replay_transformers.py so the feature
extraction can never drift between training and inference.

A "candidate" is a (row, col, rotation) legal resting placement of the current
piece on the current board. For each candidate we build:
  - board       : (H, W, 1) binary occupancy of the board BEFORE placement
  - piece_type  : (7,) one-hot
  - rotation    : (4,) one-hot
  - position    : (2,) normalized (row, col)
  - surface     : (W,) normalized column heights of the current board (top of stack)
  - holefit     : (5,) hole-fit descriptor for this candidate (what the move does)

and labels (from simulating the candidate):
  - score        : shaped placement-quality (scaled)
  - lines_cleared: 0..4
  - holes_created: >=0
plus a sample weight (line-clearing / row-completing moves weigh more).
"""

from typing import Dict, List, Tuple

import numpy as np

from puzzlegrid.pieces import Piece, PieceType
from puzzlegrid import placement

PIECE_TYPE_MAP = {pt.name: i for i, pt in enumerate(PieceType)}
SCORE_SCALE = 100.0  # keep regression targets ~O(1)


# ---------------------------------------------------------------- board metrics
def count_holes(board: List[List[int]]) -> int:
    """Empty cells that have at least one filled cell above them in the column."""
    if not board or not board[0]:
        return 0
    h, w = len(board), len(board[0])
    holes = 0
    for c in range(w):
        seen = False
        for r in range(h):
            if board[r][c] != 0:
                seen = True
            elif seen:
                holes += 1
    return holes


def column_heights(board: List[List[int]]) -> List[int]:
    """Top-of-stack height per column (H - topmost filled row; 0 if empty)."""
    h, w = len(board), len(board[0])
    heights = [0] * w
    for c in range(w):
        for r in range(h):
            if board[r][c] != 0:
                heights[c] = h - r
                break
    return heights


def bumpiness(heights: List[int]) -> int:
    return sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))


def compute_row_completion(board_after: List[List[int]], piece_cells,
                           board_w: int, board_h: int) -> Tuple[float, int]:
    """Bonus for advancing near-full rows (esp. rows the piece contributes to).

    Adapted from legacy/training_transformers_nonpatch_tf.py.
    Returns (row_completion_bonus, rows_near_completion).
    """
    bonus = 0.0
    near = 0
    contributed = set(r for r, _ in piece_cells if 0 <= r < board_h)
    for r in range(board_h - 1, max(0, board_h - 15) - 1, -1):
        filled = sum(1 for cell in board_after[r] if cell != 0)
        if 0 < filled < board_w:
            row_weight = (board_h - r) / board_h
            mult = 2.0 if r in contributed else 1.0
            if filled >= 8:
                bonus += 500.0 * row_weight * mult; near += 1
            elif filled >= 7:
                bonus += 300.0 * row_weight * mult; near += 1
            elif filled >= 6:
                bonus += 150.0 * row_weight * mult
            elif filled >= 5:
                bonus += 80.0 * row_weight * mult
            else:
                bonus += (filled / board_w) * 30.0 * row_weight * mult
    if near > 1:
        bonus += near * 100.0
    return bonus, near


# ---------------------------------------------------------------- per-candidate
def candidate_features(board_before: List[List[int]], piece_type: PieceType,
                       row: int, col: int, rotation: int) -> Dict[str, np.ndarray]:
    """Feature dict for one candidate placement (no labels)."""
    board_h, board_w = len(board_before), len(board_before[0])
    board_after, _ = placement.simulate_placement(
        board_before, piece_type, row, col, rotation
    )

    heights_before = column_heights(board_before)
    holes_before = count_holes(board_before)
    holes_after = count_holes(board_after)
    holes_created = max(0, holes_after - holes_before)
    holes_filled = max(0, holes_before - holes_after)
    bump_before = bumpiness(heights_before)
    bump_after = bumpiness(column_heights(board_after))
    max_h_after = max(column_heights(board_after))

    board_2d = (np.array(board_before, dtype=np.float32) > 0).astype(np.float32)

    piece_onehot = np.zeros(7, dtype=np.float32)
    piece_onehot[PIECE_TYPE_MAP[piece_type.name]] = 1.0
    rot_onehot = np.zeros(4, dtype=np.float32)
    rot_onehot[rotation % 4] = 1.0

    surface = np.array([h / board_h for h in heights_before], dtype=np.float32)
    max_cells = board_w * board_h
    holefit = np.array([
        holes_created / max_cells,
        holes_filled / max_cells,
        (bump_after - bump_before) / max_cells,   # surface roughness delta
        max_h_after / board_h,                     # how high this leaves the stack
        1.0 if holes_created == 0 else 0.0,        # clean-landing flag
    ], dtype=np.float32)

    return {
        "board": board_2d[..., np.newaxis],
        "piece_type": piece_onehot,
        "rotation": rot_onehot,
        "position": np.array([row / board_h, col / board_w], dtype=np.float32),
        "surface": surface,
        "holefit": holefit,
    }


def candidate_labels(board_before: List[List[int]], piece_type: PieceType,
                     row: int, col: int, rotation: int) -> Tuple[float, int, int, float]:
    """Return (score_scaled, lines_cleared, holes_created, sample_weight)."""
    board_h, board_w = len(board_before), len(board_before[0])
    board_after, lines_cleared = placement.simulate_placement(
        board_before, piece_type, row, col, rotation
    )
    holes_before = count_holes(board_before)
    holes_after = count_holes(board_after)
    holes_created = max(0, holes_after - holes_before)
    holes_filled = max(0, holes_before - holes_after)

    piece_cells = Piece(piece_type, row=row, col=col, rotation=rotation).get_cells()
    rc_bonus, rows_near = compute_row_completion(board_after, piece_cells, board_w, board_h)

    base = -0.01
    gap_penalty = -holes_created * 10.0
    gap_bonus = holes_filled * 5.0
    rc_reward = rc_bonus * 0.1

    if lines_cleared > 0:
        line_reward = lines_cleared * 100.0
        tetris_bonus = {1: 0.0, 2: 20.0, 3: 50.0}.get(lines_cleared, 200.0)
        score = base + line_reward + tetris_bonus + gap_penalty + gap_bonus + rc_reward
        weight = 10.0 + lines_cleared * 5.0
    else:
        if holes_created > 0:
            gap_penalty *= 2.0
        score = base + gap_penalty + gap_bonus + rc_reward
        if rows_near > 0:
            weight = 2.0 + rows_near * 0.5
        elif holes_created == 0 and holes_filled > 0:
            weight = 2.0
        else:
            weight = 1.0

    return score / SCORE_SCALE, int(lines_cleared), int(holes_created), float(weight)


# ---------------------------------------------------------------- board replay
def reconstruct_board_before(prev_after: List[List[int]], width: int, height: int):
    """Board seen before a move = the previous move's after-state (or empty)."""
    if prev_after is None:
        return [[0] * width for _ in range(height)]
    return [r[:] for r in prev_after]


FEATURE_KEYS = ["board", "piece_type", "rotation", "position", "surface", "holefit"]


# ---------------------------------------------------------------- policy (generator) side
# The policy/student transformer chooses a placement directly from the STATE only
# (board + surface + piece), like the VAE/GAN generators — no per-candidate features.
STATE_FEATURE_KEYS = ["board", "piece_type", "surface"]


def state_features(board_before: List[List[int]], piece_type: PieceType) -> Dict[str, np.ndarray]:
    """State-level features for the direct policy: board + surface + piece (no candidate)."""
    board_h, board_w = len(board_before), len(board_before[0])
    heights = column_heights(board_before)
    board_2d = (np.array(board_before, dtype=np.float32) > 0).astype(np.float32)
    piece_onehot = np.zeros(7, dtype=np.float32)
    piece_onehot[PIECE_TYPE_MAP[piece_type.name]] = 1.0
    surface = np.array([h / board_h for h in heights], dtype=np.float32)
    return {
        "board": board_2d[..., np.newaxis],
        "piece_type": piece_onehot,
        "surface": surface,
    }


def snap_to_placement(candidates: List[Tuple[int, int, int]], rotation: int, col: int):
    """Map a predicted (rotation, col) to a legal placement, snapping if needed.

    Prefer an exact (col, rotation) match (deepest row); else the legal candidate
    closest in column, then rotation. Mirrors the VAE/GAN "snap to legal" step.
    """
    if not candidates:
        return None
    exact = [p for p in candidates if p[2] == rotation and p[1] == col]
    if exact:
        return max(exact, key=lambda p: p[0])  # deepest resting row
    return min(candidates, key=lambda p: (abs(p[1] - col),
                                          0 if p[2] == rotation else 1,
                                          -p[0]))
