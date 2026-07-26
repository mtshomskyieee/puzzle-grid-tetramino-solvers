"""Step 4 — transformer solver: the placement features it scores, and legal snapping."""
from puzzlegrid.pieces import PieceType
from puzzlegrid import placement
import transformer_features as tf_feat


def _empty():
    return [[0] * 10 for _ in range(20)]


def test_count_holes_detects_a_hole():
    board = _empty()
    board[19][0] = 0      # empty cell...
    board[18][0] = 1      # ...covered from above -> a hole
    assert tf_feat.count_holes(board) >= 1


def test_candidate_features_keys_and_board_shape():
    board = _empty()
    feats = tf_feat.candidate_features(board, PieceType.T, row=18, col=4, rotation=0)
    assert set(tf_feat.FEATURE_KEYS).issubset(feats.keys())
    assert feats["board"].shape == (20, 10, 1)


def test_snap_to_placement_picks_a_legal_move():
    board = _empty()
    candidates = placement.get_valid_placements(board, PieceType.L)
    snapped = tf_feat.snap_to_placement(candidates, rotation=0, col=3)
    assert snapped in candidates
