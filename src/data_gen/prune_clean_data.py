#!/usr/bin/env python3
"""Prune weak games from a training-data dir by final_lines_cleared.

Games clearing fewer than --min-lines are moved to a `<dir>/_pruned/` subdir
(not deleted) so the training set contains only strong demonstrations. Also
re-verifies every placement is reachable (paranoia check on the clean set).

Usage: uv run python prune_clean_data.py --dir training_data_clean --min-lines 5
"""

import argparse
import glob
import json
import os
import shutil

from puzzlegrid.pieces import PieceType
from puzzlegrid import placement


def verify_reachable(data) -> bool:
    grid = [[0] * data["width"] for _ in range(data["height"])]
    for m in data.get("moves", []):
        if m.get("move_type") != "place":
            continue
        pos = m["piece_position"]
        pt = PieceType[m["piece_type"]]
        r, c, rot = pos["row"], pos["col"], pos["rotation"]
        if not placement.can_reach_placement(grid, pt, r, c, rot):
            return False
        grid, _ = placement.simulate_placement(grid, pt, r, c, rot)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="training_data_clean")
    ap.add_argument("--min-lines", type=int, default=5)
    ap.add_argument("--apply", action="store_true", help="actually move files (else dry-run)")
    args = ap.parse_args()

    pruned_dir = os.path.join(args.dir, "_pruned")
    files = sorted(glob.glob(os.path.join(args.dir, "*.json")))
    kept = weak = unreachable = 0

    for f in files:
        data = json.load(open(f))
        lines = data.get("final_lines_cleared", 0)
        ok = verify_reachable(data)
        if not ok:
            unreachable += 1
        if lines < args.min_lines or not ok:
            weak += 1
            if args.apply:
                os.makedirs(pruned_dir, exist_ok=True)
                shutil.move(f, os.path.join(pruned_dir, os.path.basename(f)))
        else:
            kept += 1

    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"[{mode}] dir={args.dir} min_lines={args.min_lines}")
    print(f"  total={len(files)} kept={kept} pruned={weak} (unreachable-flagged={unreachable})")


if __name__ == "__main__":
    main()
