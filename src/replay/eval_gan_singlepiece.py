#!/usr/bin/env python3
"""Evaluate the GAN in single-piece games (allowed_pieces=[P]) for each of the 7 pieces.

Uses the running API server (supports allowed_pieces) rather than in-process mode.
Usage: uv run python eval_gan_singlepiece.py --model-dir ./models/gan_clean_v2 \
         --api-url http://localhost:8123 --games 3
"""
import argparse

from replay_gan import PuzzleGridGANPlayer

PIECES = ["I", "O", "T", "S", "Z", "J", "L"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="./models/gan_clean_v2")
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--line-clear-bonus", type=float, default=3.0)
    ap.add_argument("--max-moves", type=int, default=400)
    args = ap.parse_args()

    print(f"model={args.model_dir}  games/piece={args.games}  "
          f"num_samples={args.num_samples}  bonus={args.line_clear_bonus}")
    print(f"{'piece':<6}{'lines per game':<28}{'best':<6}{'mean':<6}")

    for piece in PIECES:
        results = []
        for _ in range(args.games):
            player = PuzzleGridGANPlayer(
                args.model_dir,
                api_url=args.api_url,
                num_samples=args.num_samples,
                line_clear_bonus=args.line_clear_bonus,
                use_inprocess=False,
            )
            player.create_game(allowed_pieces=[piece])
            final = player.play_game(max_moves=args.max_moves, mode="search")
            results.append(int(final.get("lines_cleared", 0)))
        best = max(results)
        mean = sum(results) / len(results)
        print(f"{piece:<6}{str(results):<28}{best:<6}{mean:<6.1f}")


if __name__ == "__main__":
    main()
