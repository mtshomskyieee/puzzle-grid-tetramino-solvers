"""Step 1 — heuristic data generator: the shipped clean data is well-formed."""
import glob
import importlib.util
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "training_data_clean")


def test_clean_data_ships():
    files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    assert len(files) > 100, "expected the curated clean training set to ship"


def test_game_log_has_expected_shape():
    path = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))[0]
    with open(path) as f:
        game = json.load(f)
    for key in ("moves", "final_lines_cleared", "allowed_pieces"):
        assert key in game
    move = game["moves"][0]
    assert "piece_type" in move and "board_state" in move


def test_generator_module_imports():
    # the generator filename is hyphenated, so import it by path
    path = os.path.join(REPO_ROOT, "src", "data_gen", "heuristic-data-generator.py")
    spec = importlib.util.spec_from_file_location("heuristic_data_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "BASE_URL")
