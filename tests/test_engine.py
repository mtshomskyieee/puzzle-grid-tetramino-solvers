"""Step 0 — game engine: rules, board, and reachability/placement."""
from puzzlegrid.board import Board
from puzzlegrid.game import GameStatus, PuzzleGridGame
from puzzlegrid.pieces import PieceType
from puzzlegrid import placement


def test_new_game_starts_playing():
    game = PuzzleGridGame(width=10, height=20, piece_sequence=["O"])
    state = game.get_state()
    assert game.status == GameStatus.PLAYING
    assert state["lines_cleared"] == 0


def test_direct_placement_locks_a_piece():
    game = PuzzleGridGame(width=10, height=20, piece_sequence=["O", "O"])
    ok = game.place_piece_direct(row=18, col=0, rotation=0, drop_to_rest=True)
    assert ok is True
    # something is now resting on the board
    assert any(cell != 0 for row in game.board.get_state() for cell in row)


def test_full_row_clears():
    board = Board(width=4, height=6)
    bottom = board.height - 1
    for col in range(board.width):
        board.place_piece([(bottom, col)], piece_id=1)
    assert board.clear_lines() == 1


def test_valid_placements_are_reachable_and_resting(board):
    placements = placement.get_valid_placements(board, PieceType.T)
    assert placements, "an empty board should offer legal T placements"
    for row, col, rotation in placements:
        assert placement.can_reach_placement(board, PieceType.T, row, col, rotation)


def test_simulate_placement_is_deterministic(board):
    p = placement.get_valid_placements(board, PieceType.I)[0]
    a, cleared_a = placement.simulate_placement(board, PieceType.I, *p)
    b, cleared_b = placement.simulate_placement(board, PieceType.I, *p)
    assert a == b and cleared_a == cleared_b
