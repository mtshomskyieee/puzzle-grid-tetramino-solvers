"""Shared bootstrap and drawing helpers for the solver notebooks.

Import this *before* TensorFlow. It pins CPU, puts ``src/{engine,common,solvers,replay}``
on ``sys.path``, and exposes board scenarios + matplotlib drawing used by all three
notebooks.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

NOTEBOOK_DIR = Path(__file__).resolve().parent
REPO_ROOT = NOTEBOOK_DIR.parent

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

for _sub in ("engine", "common", "solvers", "replay"):
    _p = str(REPO_ROOT / "src" / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA_DIR = REPO_ROOT / "data" / "training_data_clean"
VAE_DIR = REPO_ROOT / "models" / "vae_clean"
GAN_DIR = REPO_ROOT / "models" / "gan_clean_v2"
TRANSFORMER_PATH = REPO_ROOT / "models" / "transformer_scorer" / "transformer_scorer_best.keras"

PIECE_NAMES = ["I", "O", "T", "S", "Z", "J", "L"]
PIECE_COLORS = {
    "I": "#67e8f9",
    "O": "#fde047",
    "T": "#c084fc",
    "S": "#4ade80",
    "Z": "#f87171",
    "J": "#60a5fa",
    "L": "#fb923c",
    "stack": "#64748b",
    "empty": "#0f172a",
    "highlight": "#facc15",
}

Placement = Tuple[int, int, int]


def setup_notebook():
    """Configure matplotlib defaults. Call once from the first code cell."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "figure.facecolor": "#0b1220",
            "axes.facecolor": "#0b1220",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#e2e8f0",
            "text.color": "#e2e8f0",
            "xtick.color": "#94a3b8",
            "ytick.color": "#94a3b8",
            "grid.color": "#1e293b",
            "savefig.facecolor": "#0b1220",
            "font.size": 11,
        }
    )
    plt.rcParams["image.cmap"] = "magma"
    print(f"REPO_ROOT = {REPO_ROOT}")
    print(f"DATA_DIR  = {DATA_DIR}  ({len(list(DATA_DIR.glob('*.json')))} games)")
    print(f"VAE_DIR   = {VAE_DIR}")
    print(f"GAN_DIR   = {GAN_DIR}")
    print(f"TRANSFORMER_PATH = {TRANSFORMER_PATH}")


def empty_board(height: int = 20, width: int = 10) -> List[List[int]]:
    return [[0] * width for _ in range(height)]


def clone_board(board: List[List[int]]) -> List[List[int]]:
    return [row[:] for row in board]


def scenario_almost_line(missing_col: int = 9, n_rows: int = 1) -> List[List[int]]:
    """Bottom row(s) filled except one column — a one-piece line-clear setup."""
    board = empty_board()
    h, w = len(board), len(board[0])
    for r in range(h - n_rows, h):
        for c in range(w):
            if c != missing_col:
                board[r][c] = 1
    return board


def scenario_right_well(height: int = 6) -> List[List[int]]:
    """Open well on column 9 — classic I-piece Tetris setup."""
    board = empty_board()
    h, w = len(board), len(board[0])
    for r in range(h - height, h):
        for c in range(w - 1):
            board[r][c] = 1
    return board


def scenario_left_well(height: int = 4) -> List[List[int]]:
    board = empty_board()
    h, w = len(board), len(board[0])
    for r in range(h - height, h):
        for c in range(1, w):
            board[r][c] = 1
    return board


def scenario_bumpy() -> List[List[int]]:
    """Uneven surface — a covariate-shift-ish board the VAE never liked."""
    board = empty_board()
    heights = [5, 1, 6, 2, 7, 1, 4, 3, 6, 2]
    h = len(board)
    for c, col_h in enumerate(heights):
        for r in range(h - col_h, h):
            board[r][c] = 1
    return board


def scenario_flat_stack(height: int = 4) -> List[List[int]]:
    board = empty_board()
    h, w = len(board), len(board[0])
    for r in range(h - height, h):
        for c in range(w):
            board[r][c] = 1
    return board


def scenario_overhang_pocket() -> List[List[int]]:
    """Enclosed empty cell under an overhang — legal only if BFS can slide in."""
    board = empty_board()
    h = len(board)
    for c in range(10):
        board[h - 1][c] = 1
    board[h - 2][0] = 0
    board[h - 2][1] = 1
    board[h - 2][2] = 1
    for c in range(10):
        board[h - 3][c] = 1
    board[h - 3][0] = 1
    return board


SCENARIOS = {
    "empty": empty_board,
    "almost_line": scenario_almost_line,
    "right_well": scenario_right_well,
    "left_well": scenario_left_well,
    "bumpy": scenario_bumpy,
    "flat_stack": scenario_flat_stack,
    "overhang_pocket": scenario_overhang_pocket,
}


def load_game(path: Optional[Path] = None) -> dict:
    files = sorted(DATA_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No training games in {DATA_DIR}")
    path = path or files[len(files) // 2]
    with open(path) as f:
        return json.load(f)


def board_after_move(game: dict, move_index: int) -> List[List[int]]:
    """Return the board *after* the Nth place-move (0-based among place moves)."""
    places = [m for m in game.get("moves", []) if m.get("move_type") == "place"]
    if not places:
        return empty_board(game.get("height", 20), game.get("width", 10))
    move_index = max(0, min(move_index, len(places) - 1))
    return places[move_index]["board_state"]


def board_before_nth_place(game: dict, n: int = 15) -> Tuple[List[List[int]], Optional[str]]:
    """Pre-placement board and piece type for the nth place-move."""
    w, h = game.get("width", 10), game.get("height", 20)
    prev = None
    k = 0
    for move in game.get("moves", []):
        if move.get("move_type") != "place":
            continue
        if k == n:
            if prev is None:
                return empty_board(h, w), move.get("piece_type")
            return [row[:] for row in prev], move.get("piece_type")
        prev = move.get("board_state")
        k += 1
    if prev is None:
        return empty_board(h, w), None
    return [row[:] for row in prev], None


def piece_cells(piece_type_name: str, row: int, col: int, rotation: int):
    from puzzlegrid.pieces import Piece, PieceType

    piece = Piece(PieceType[piece_type_name], row=row, col=col, rotation=rotation)
    return piece.get_cells()


def show_board(
    board: List[List[int]],
    piece: Optional[Placement] = None,
    piece_type: Optional[str] = None,
    heatmap=None,
    title: str = "",
    ax=None,
    show: bool = True,
):
    """Draw a 20×10 occupancy grid, optional falling/locked piece, optional heatmap."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    h, w = len(board), len(board[0])
    grid = np.array(board, dtype=float)
    occ = (grid > 0).astype(float)

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(4.2, 7.4))
    else:
        fig = ax.figure

    if heatmap is not None:
        hm = np.asarray(heatmap, dtype=float)
        ax.imshow(hm, cmap="magma", origin="upper", interpolation="nearest", alpha=0.85)
        ax.imshow(
            np.where(occ > 0, 1.0, np.nan),
            cmap=ListedColormap(["#94a3b8"]),
            origin="upper",
            interpolation="nearest",
            alpha=0.45,
        )
        if piece is not None and piece_type is not None:
            pr, pc, prot = piece
            ys, xs = [], []
            for rr, cc in piece_cells(piece_type, pr, pc, prot):
                if 0 <= rr < h and 0 <= cc < w:
                    ys.append(rr)
                    xs.append(cc)
            if xs:
                ax.scatter(xs, ys, s=42, facecolors="none", edgecolors="#facc15", linewidths=1.4)
    else:
        rgba = np.zeros((h, w, 4))
        rgba[..., :3] = 0.06
        rgba[..., 3] = 1.0
        rgba[occ > 0] = (0.39, 0.45, 0.55, 1.0)
        if piece is not None and piece_type is not None:
            pr, pc, prot = piece
            color = PIECE_COLORS.get(piece_type, PIECE_COLORS["highlight"])
            rgb = _hex_to_rgb(color)
            for rr, cc in piece_cells(piece_type, pr, pc, prot):
                if 0 <= rr < h and 0 <= cc < w:
                    rgba[rr, cc, :3] = rgb
                    rgba[rr, cc, 3] = 1.0
        ax.imshow(rgba, origin="upper", interpolation="nearest")

    ax.set_xticks(range(w))
    ax.set_yticks(range(h))
    ax.set_xticklabels(range(w), fontsize=8)
    ax.set_yticklabels(range(h), fontsize=7)
    ax.set_xticks([x - 0.5 for x in range(1, w)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, h)], minor=True)
    ax.grid(which="minor", color="#1e293b", linewidth=0.6)
    ax.tick_params(which="major", length=0)
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    if created and show:
        plt.tight_layout()
        plt.show()
    return ax


def show_boards(items: Sequence[dict], cols: int = 3, figsize=None):
    """items: list of dicts with keys board, piece, piece_type, heatmap, title."""
    import math
    import matplotlib.pyplot as plt

    n = len(items)
    cols = max(1, cols)
    rows = int(math.ceil(n / cols))
    figsize = figsize or (4.0 * cols, 6.6 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes_list = list(axes.ravel()) if n > 1 else [axes]
    for i, ax in enumerate(axes_list):
        if i >= n:
            ax.axis("off")
            continue
        it = items[i]
        show_board(
            it["board"],
            piece=it.get("piece"),
            piece_type=it.get("piece_type"),
            heatmap=it.get("heatmap"),
            title=it.get("title", ""),
            ax=ax,
            show=False,
        )
    plt.tight_layout()
    plt.show()


def _hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def encode_board(board: List[List[int]]):
    import numpy as np

    arr = (np.array(board, dtype=np.float32) > 0).astype(np.float32)
    return arr[np.newaxis, ..., np.newaxis]


def piece_onehot(piece_type_name: str):
    import numpy as np

    v = np.zeros((1, 7), dtype=np.float32)
    v[0, PIECE_NAMES.index(piece_type_name)] = 1.0
    return v


def rotation_onehot(rotation: int):
    import numpy as np

    v = np.zeros((1, 4), dtype=np.float32)
    v[0, rotation % 4] = 1.0
    return v


def load_vae_v2():
    """Rebuild TetrisVAEV2 and load shipped weights (full keras class does not deserialize)."""
    import numpy as np
    import tensorflow as tf
    from training_vae_v2 import build_tetris_vae

    model = build_tetris_vae(latent_dim=64, beta=0.05, line_clear_weight=2.0)
    board = tf.zeros((1, 20, 10, 1), dtype=tf.float32)
    piece_idx = tf.zeros((1,), dtype=tf.int32)
    rot_idx = tf.zeros((1,), dtype=tf.int32)
    pos = tf.zeros((1, 2), dtype=tf.float32)
    feat = model.board_encoder(board, training=False)
    action = model.build_action_vector(piece_idx, rot_idx, pos)
    z_mean, z_log_var, z = model.encode(feat, action, training=False)
    _ = model.decode(feat, z, tf.one_hot(piece_idx, 7), training=False)
    # TetrisVAEV2 has no call(); Keras still requires the container to be built.
    model.built = True
    weights = VAE_DIR / "tetris_vae_v2_best.keras"
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")
        model.load_weights(str(weights))
    return model


def conv_activation_model(keras_model, conv_names: Sequence[str]):
    """Return a model that emits named Conv2D feature maps from a CNN."""
    import tensorflow as tf

    outputs = [keras_model.get_layer(name).output for name in conv_names]
    return tf.keras.Model(keras_model.inputs, outputs)


def show_feature_maps(maps, title: str, max_channels: int = 16):
    """Plot the first ``max_channels`` channels of a (1,H,W,C) or (H,W,C) tensor."""
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    arr = np.asarray(maps)
    if arr.ndim == 4:
        arr = arr[0]
    h, w, c = arr.shape
    n = min(c, max_channels)
    cols = 8 if n > 8 else n
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.35 * cols, 1.55 * rows + 0.4))
    axes_list = list(np.atleast_1d(axes).ravel())
    vmax = np.percentile(np.abs(arr[..., :n]), 99) + 1e-6
    for i, ax in enumerate(axes_list):
        if i >= n:
            ax.axis("off")
            continue
        ax.imshow(arr[..., i], cmap="magma", origin="upper", vmin=-vmax, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(str(i), fontsize=8, color="#94a3b8")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def ranked_table(rows: List[dict], keys: Sequence[str], n: int = 12):
    """Pretty-print a list of dicts as a compact table."""
    header = "  ".join(f"{k:>14}" for k in keys)
    print(header)
    print("-" * len(header))
    for row in rows[:n]:
        print("  ".join(f"{_fmt(row.get(k)):>14}" for k in keys))


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)
