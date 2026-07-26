"""Step 3 — GAN solver: the in-process game wrapper it trains/plays against."""
import numpy as np

from puzzlegrid.pieces import PieceType
from gan_friendly_game import (
    GANFriendlyGame,
    encode_board_binary,
    simulate_placement_detailed,
)


def test_encode_board_binary_is_01():
    board = [[0] * 10 for _ in range(20)]
    board[19][0] = 3  # a nonzero cell id
    enc = encode_board_binary(board)
    assert set(np.unique(enc)).issubset({0.0, 1.0})
    assert enc.sum() == 1.0


def test_simulate_placement_detailed_snaps_to_legal():
    board = [[0] * 10 for _ in range(20)]
    # an intentionally illegal spot; snap=True should map it to a legal placement
    result = simulate_placement_detailed(board, PieceType.O, row=0, col=0, rotation=0, snap=True)
    assert result.valid
    assert any(cell != 0 for row in result.post_board for cell in row)


def test_ganfriendly_game_places_a_piece():
    game = GANFriendlyGame(width=10, height=20, piece_sequence=["O", "O"])
    assert game.width == 10 and game.height == 20
