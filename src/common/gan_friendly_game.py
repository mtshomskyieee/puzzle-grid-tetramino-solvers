#!/usr/bin/env python3
"""GAN-friendly Tetris environment: snapshots, undo, simulation, legal placements.

Uses exact locked coordinates (no drop_to_rest) for placement simulation, matching
VAE training data. Wraps puzzlegrid game logic for in-process training/replay.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from puzzlegrid.board import Board
from puzzlegrid.game import GameStatus, PuzzleGridGame
from puzzlegrid.pieces import Piece, PieceType
from puzzlegrid import placement as _placement


Placement = Tuple[int, int, int]  # row, col, rotation


@dataclass
class GameSnapshot:
    """Full restorable game state."""

    board_grid: List[List[int]]
    status: str
    score: int
    lines_cleared_total: int
    pieces_placed: int
    piece_sequence_index: int
    spawned_pieces: List[str]
    current_piece: Optional[Dict[str, Any]]
    next_piece_type: Optional[str]
    move_history_len: int


@dataclass
class SimulationResult:
    """Outcome of simulating a placement on a board copy."""

    post_board: List[List[int]]
    lines_cleared: int
    score_delta: int
    valid: bool
    final_placement: Optional[Placement] = None
    snap_distance: float = 0.0


def encode_board_binary(board: List[List[int]]) -> np.ndarray:
    """Binary occupancy grid (H, W) float32."""
    arr = np.array(board, dtype=np.float32)
    return (arr > 0).astype(np.float32)


def encode_board_tensor(board: List[List[int]]) -> np.ndarray:
    """Board as (1, H, W, 1) for CNN input."""
    b = encode_board_binary(board)
    return b[np.newaxis, ..., np.newaxis]


def build_action_vector(
    piece_idx: int,
    rotation_idx: int,
    row: int,
    col: int,
    board_height: int,
    board_width: int,
) -> np.ndarray:
    """VAE-compatible 13-d action: piece(7) + rotation(4) + norm position(2)."""
    piece_onehot = np.zeros(7, dtype=np.float32)
    piece_onehot[piece_idx] = 1.0
    rot_onehot = np.zeros(4, dtype=np.float32)
    rot_onehot[rotation_idx % 4] = 1.0
    pos = np.array(
        [row / board_height, col / board_width], dtype=np.float32
    )
    return np.concatenate([piece_onehot, rot_onehot, pos])


def piece_type_to_idx(piece_type: PieceType) -> int:
    return list(PieceType).index(piece_type)


def idx_to_piece_type(idx: int) -> PieceType:
    return list(PieceType)[idx]


def snap_to_valid(
    row: int,
    col: int,
    rotation: int,
    valid_placements: List[Placement],
) -> Tuple[Optional[Placement], float]:
    """Snap raw (row, col, rotation) to nearest legal placement."""
    if not valid_placements:
        return None, float("inf")

    rotation = rotation % 4
    if (row, col, rotation) in valid_placements:
        return (row, col, rotation), 0.0

    best: Optional[Placement] = None
    best_dist = float("inf")
    for vr, vc, vrot in valid_placements:
        dist = abs(vr - row) + abs(vc - col) + (0.0 if vrot == rotation else 2.0)
        if dist < best_dist:
            best_dist = dist
            best = (vr, vc, vrot)
    return best, best_dist


def _can_reach_position(
    board: List[List[int]],
    piece_type: PieceType,
    row: int,
    col: int,
    rotation: int,
) -> bool:
    """Reachability check (delegates to the canonical BFS in puzzlegrid.placement)."""
    return _placement.can_reach_placement(board, piece_type, row, col, rotation)


def get_valid_placements(
    board: List[List[int]], piece_type: PieceType
) -> List[Placement]:
    """All legal resting placements for piece_type on board.

    Uses the canonical reachable-placement set, then applies the low-stack
    shaping policy this engine has always exposed to callers/UI.
    """
    valid = _placement.get_valid_placements(board, piece_type)
    return _placement.filter_low_stack(board, piece_type, valid)


def simulate_placement(
    board_state: List[List[int]],
    piece_type: PieceType,
    row: int,
    col: int,
    rotation: int,
    drop_to_rest: bool = False,
) -> Tuple[List[List[int]], int]:
    """Place piece at locked coords; return post-board and lines cleared."""
    return _placement.simulate_placement(
        board_state, piece_type, row, col, rotation, drop_to_rest=drop_to_rest
    )


def simulate_placement_detailed(
    board_state: List[List[int]],
    piece_type: PieceType,
    row: int,
    col: int,
    rotation: int,
    snap: bool = True,
) -> SimulationResult:
    """Simulate with optional snap to legal placement."""
    valid = get_valid_placements(board_state, piece_type)
    if not valid:
        return SimulationResult(
            post_board=[r[:] for r in board_state],
            lines_cleared=0,
            score_delta=0,
            valid=False,
        )

    final_row, final_col, final_rot = row, col, rotation % 4
    snap_dist = 0.0
    if snap:
        snapped, snap_dist = snap_to_valid(row, col, rotation, valid)
        if snapped is None:
            return SimulationResult(
                post_board=[r[:] for r in board_state],
                lines_cleared=0,
                score_delta=0,
                valid=False,
            )
        final_row, final_col, final_rot = snapped

    placement = (final_row, final_col, final_rot)
    if placement not in valid:
        return SimulationResult(
            post_board=[r[:] for r in board_state],
            lines_cleared=0,
            score_delta=0,
            valid=False,
            final_placement=placement,
            snap_distance=snap_dist,
        )

    post_board, lines = simulate_placement(
        board_state, piece_type, final_row, final_col, final_rot
    )
    score_delta = 100 * (lines ** 2) if lines > 0 else 0
    return SimulationResult(
        post_board=post_board,
        lines_cleared=lines,
        score_delta=score_delta,
        valid=True,
        final_placement=placement,
        snap_distance=snap_dist,
    )


class GANFriendlyGame:
    """In-process game with pause, snapshot, undo, and raw move stepping."""

    def __init__(
        self,
        width: int = 10,
        height: int = 20,
        allowed_pieces: Optional[List[PieceType]] = None,
        piece_sequence: Optional[List[str]] = None,
        game_id: Optional[str] = None,
    ):
        self._game = PuzzleGridGame(
            width=width,
            height=height,
            allowed_pieces=allowed_pieces,
            game_id=game_id,
            piece_sequence=piece_sequence,
        )
        self._paused = False
        self._undo_stack: List[GameSnapshot] = []
        self._max_undo = 256

    @property
    def width(self) -> int:
        return self._game.board.width

    @property
    def height(self) -> int:
        return self._game.board.height

    def pause(self) -> None:
        self._paused = True
        self._game.status = GameStatus.PAUSED

    def resume(self) -> None:
        self._paused = False
        if self._game.status == GameStatus.PAUSED:
            self._game.status = GameStatus.PLAYING

    @property
    def is_paused(self) -> bool:
        return self._paused

    def snapshot(self) -> GameSnapshot:
        state = self._game.get_state()
        cp = state.get("current_piece")
        current = copy.deepcopy(cp) if cp else None
        return GameSnapshot(
            board_grid=self._game.board.get_state(),
            status=self._game.status.value,
            score=self._game.score,
            lines_cleared_total=self._game.lines_cleared_total,
            pieces_placed=self._game.pieces_placed,
            piece_sequence_index=self._game.piece_sequence_index,
            spawned_pieces=list(self._game.spawned_pieces),
            current_piece=current,
            next_piece_type=state.get("next_piece"),
            move_history_len=len(self._game.move_history),
        )

    def restore(self, snap: GameSnapshot) -> None:
        self._game.board.grid = [row[:] for row in snap.board_grid]
        self._game.status = GameStatus(snap.status)
        self._game.score = snap.score
        self._game.lines_cleared_total = snap.lines_cleared_total
        self._game.pieces_placed = snap.pieces_placed
        self._game.piece_sequence_index = snap.piece_sequence_index
        self._game.spawned_pieces = list(snap.spawned_pieces)
        self._game.move_history = self._game.move_history[: snap.move_history_len]

        if snap.current_piece:
            self._game.current_piece = Piece(
                PieceType[snap.current_piece["type"]],
                row=snap.current_piece["row"],
                col=snap.current_piece["col"],
                rotation=snap.current_piece["rotation"],
            )
        else:
            self._game.current_piece = None

        if snap.next_piece_type:
            pt = PieceType[snap.next_piece_type]
            spawn_row = self._game._calculate_spawn_row(pt)
            self._game.next_piece = Piece(
                pt, row=spawn_row, col=self._game.board.width // 2 - 1
            )
        else:
            self._game.next_piece = None

        self._paused = self._game.status == GameStatus.PAUSED

    def push_undo_point(self) -> None:
        self._undo_stack.append(self.snapshot())
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self.restore(self._undo_stack.pop())
        return True

    def get_board(self) -> List[List[int]]:
        return self._game.board.get_state()

    def get_current_piece_type(self) -> Optional[PieceType]:
        if self._game.current_piece is None:
            return None
        return self._game.current_piece.piece_type

    def get_state_dict(self) -> dict:
        return self._game.get_state()

    def line_clear_reward(self, lines_cleared: int) -> float:
        """Training reward emphasizing line clears."""
        return float(lines_cleared)

    def step_raw(
        self,
        row: int,
        col: int,
        rotation: int,
        *,
        record_undo: bool = True,
        use_drop_to_rest: bool = False,
    ) -> Tuple[dict, float, bool, dict]:
        """
        Apply raw move: snap to legal, place without mutating during test.

        Returns (state_dict, reward, done, info).
        """
        if self._paused or self._game.status != GameStatus.PLAYING:
            return self.get_state_dict(), 0.0, True, {"error": "not_playing"}

        piece_type = self.get_current_piece_type()
        if piece_type is None:
            return self.get_state_dict(), 0.0, True, {"error": "no_piece"}

        board = self.get_board()
        valid = get_valid_placements(board, piece_type)
        snapped, snap_dist = snap_to_valid(row, col, rotation, valid)
        info: Dict[str, Any] = {
            "raw": (row, col, rotation % 4),
            "valid_placements_count": len(valid),
            "snap_distance": snap_dist,
        }

        if snapped is None:
            info["error"] = "no_legal_placement"
            done = self._game.status != GameStatus.PLAYING
            return self.get_state_dict(), 0.0, done, info

        srow, scol, srot = snapped
        info["snapped"] = (srow, scol, srot)

        if record_undo:
            self.push_undo_point()

        ok = self._game.place_piece_direct(
            srow, scol, srot, drop_to_rest=use_drop_to_rest
        )
        lines = 0
        if self._game.move_history:
            lines = self._game.move_history[-1].lines_cleared

        reward = self.line_clear_reward(lines)
        info["lines_cleared"] = lines
        info["placement_ok"] = ok

        done = self._game.status != GameStatus.PLAYING
        return self.get_state_dict(), reward, done, info

    def test_move(
        self,
        row: int,
        col: int,
        rotation: int,
    ) -> SimulationResult:
        """Simulate move without changing live game state."""
        piece_type = self.get_current_piece_type()
        if piece_type is None:
            return SimulationResult(
                post_board=self.get_board(),
                lines_cleared=0,
                score_delta=0,
                valid=False,
            )
        return simulate_placement_detailed(
            self.get_board(), piece_type, row, col, rotation, snap=True
        )

    @staticmethod
    def get_valid_placements_for_board(
        board: List[List[int]], piece_type: PieceType
    ) -> List[Placement]:
        return get_valid_placements(board, piece_type)
