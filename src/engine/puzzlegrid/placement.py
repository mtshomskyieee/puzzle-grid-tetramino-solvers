"""Canonical piece-placement logic: reachability, legal placements, simulation.

This module is the single source of truth for "where can this piece legally end up".
It replaces the per-file ``_can_reach_position`` / ``get_valid_placements`` heuristics
that only checked whether the column-corridor above a target was empty -- a check that
wrongly accepted placements inside enclosed pockets reachable only by sliding under an
overhang (the "teleport into a hole" bug).

Reachability here is a graph search over ``(row, col, rotation)`` states starting from
the spawn position, using exactly the move semantics of ``puzzlegrid.game`` (left, right,
clockwise rotate, soft-drop). A placement is legal iff the search can reach it AND it
rests (cannot move down further). Depends only on ``board`` and ``pieces`` to avoid a
circular import with ``game``.
"""

from collections import deque
from typing import List, Optional, Set, Tuple

from .board import Board
from .pieces import Piece, PieceType

# (row, col, rotation)
Placement = Tuple[int, int, int]


def piece_type_to_idx(piece_type: PieceType) -> int:
    """Index of a piece type (matches game._lock_piece piece_id = idx + 1)."""
    return list(PieceType).index(piece_type)


def get_spawn_state(piece_type: PieceType, width: int) -> Placement:
    """Spawn (row, col, rotation) for a piece, replicating game._spawn_next_piece.

    Mirrors ``game._calculate_spawn_row`` (spawn high enough to accommodate negative
    row offsets) and the horizontal spawn column ``width // 2 - 1``.
    """
    shape = Piece.PIECE_SHAPES[piece_type][0]
    min_row_offset = min(dr for dr, _ in shape)
    row = -min_row_offset if min_row_offset < 0 else 0
    col = width // 2 - 1
    return (row, col, 0)


def _board_from_state(board_state: List[List[int]]) -> Board:
    board = Board(width=len(board_state[0]), height=len(board_state))
    board.grid = [r[:] for r in board_state]
    return board


def _rests(board: Board, piece: Piece) -> bool:
    """True if the piece cannot move down further (matches game.py:240)."""
    return not board.is_valid_position(piece.move(1, 0).get_cells())


def _reachable_resting_states(
    board_state: List[List[int]], piece_type: PieceType
) -> Set[Placement]:
    """All resting placements reachable from spawn by legal moves.

    BFS over (row, col, rotation). Transitions mirror the engine's legal moves:
    left, right, clockwise rotate, soft-drop (down one). Each candidate state is
    kept only if it is in a valid (in-bounds, non-colliding) position. A state is
    recorded as a placement if it also rests.
    """
    board = _board_from_state(board_state)

    spawn = get_spawn_state(piece_type, board.width)
    spawn_piece = Piece(piece_type, row=spawn[0], col=spawn[1], rotation=spawn[2])
    # Spawn blocked => game over, nothing is placeable.
    if not board.is_valid_position(spawn_piece.get_cells()):
        return set()

    visited: Set[Placement] = {spawn}
    queue: deque = deque([spawn])
    resting: Set[Placement] = set()

    while queue:
        row, col, rot = queue.popleft()
        piece = Piece(piece_type, row=row, col=col, rotation=rot)

        if _rests(board, piece):
            resting.add((row, col, rot))

        for nxt in (
            piece.move(0, -1),  # left
            piece.move(0, 1),   # right
            piece.rotate(),     # clockwise
            piece.move(1, 0),   # soft-drop
        ):
            state = (nxt.row, nxt.col, nxt.rotation)
            if state not in visited and board.is_valid_position(nxt.get_cells()):
                visited.add(state)
                queue.append(state)

    return resting


def can_reach_placement(
    board_state: List[List[int]],
    piece_type: PieceType,
    row: int,
    col: int,
    rotation: int,
) -> bool:
    """True if (row, col, rotation) is a legal, reachable resting placement."""
    return (row, col, rotation % 4) in _reachable_resting_states(board_state, piece_type)


def get_valid_placements(
    board_state: List[List[int]], piece_type: PieceType
) -> List[Placement]:
    """All legal resting placements for the piece, sorted for determinism."""
    return sorted(_reachable_resting_states(board_state, piece_type))


def filter_low_stack(
    board_state: List[List[int]],
    piece_type: PieceType,
    placements: List[Placement],
) -> List[Placement]:
    """Stack-shaping policy (NOT reachability): prefer placements that keep the
    stack low. Returns placements leaving the top 5 rows empty; else those with a
    cell in rows < 6; else the input unchanged. Extracted from the old inline
    logic in gan_friendly_game / replay_vae so callers can opt in.
    """
    width = len(board_state[0])
    preferred: List[Placement] = []
    fallback: List[Placement] = []

    for row, col, rotation in placements:
        post_board, _ = simulate_placement(board_state, piece_type, row, col, rotation)

        has_blocks_in_top_5 = any(
            post_board[r][c] != 0 for r in range(5) for c in range(width)
        )

        if not has_blocks_in_top_5:
            preferred.append((row, col, rotation))
        else:
            piece = Piece(piece_type, row=row, col=col, rotation=rotation)
            if any(r < 6 for r, _ in piece.get_cells()):
                fallback.append((row, col, rotation))

    if preferred:
        return preferred
    return fallback if fallback else placements


def simulate_placement(
    board_state: List[List[int]],
    piece_type: PieceType,
    row: int,
    col: int,
    rotation: int,
    drop_to_rest: bool = False,
) -> Tuple[List[List[int]], int]:
    """Place piece at locked coords; return (post_board, lines_cleared).

    Matches gan_friendly_game.simulate_placement semantics: uses exact locked
    coordinates by default (drop_to_rest=False) to align with VAE training data.
    """
    board = _board_from_state(board_state)

    piece = Piece(piece_type, row=row, col=col, rotation=rotation % 4)
    if drop_to_rest:
        while board.is_valid_position(piece.move(1, 0).get_cells()):
            piece = piece.move(1, 0)

    if not board.is_valid_position(piece.get_cells()):
        return board.get_state(), 0

    piece_id = piece_type_to_idx(piece_type) + 1
    board.place_piece(piece.get_cells(), piece_id)
    lines_cleared = board.clear_lines()
    return board.get_state(), lines_cleared
