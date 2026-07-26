"""Dev tooling for the PuzzleGrid study (not part of the game engine).

Exposes the `test` console script so `uv run test` runs the pytest suite.
"""
import pathlib


def main():
    """Entry point for `uv run test` — run the pytest suite in tests/."""
    import pytest

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    raise SystemExit(pytest.main([str(repo_root / "tests")]))
