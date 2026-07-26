"""Shared test fixtures/helpers.

`pythonpath` in pyproject.toml already puts src/engine, src/common, src/solvers and
src/replay on sys.path, so tests import the engine and solver modules directly.
"""
from typing import List

import pytest


def empty_board(width: int = 10, height: int = 20) -> List[List[int]]:
    """A blank board as a height x width grid of zeros."""
    return [[0] * width for _ in range(height)]


@pytest.fixture
def board():
    return empty_board()
