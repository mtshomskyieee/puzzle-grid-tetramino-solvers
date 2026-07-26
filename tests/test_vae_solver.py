"""Step 2 — VAE solver: data extraction and the custom sampling layer."""
import glob
import json
import os

import numpy as np

from training_vae_v2 import SamplingLayer, extract_board_and_actions

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "training_data_clean")


def _one_real_game():
    path = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))[0]
    with open(path) as f:
        return json.load(f)


def test_extract_board_and_actions_shapes():
    data = extract_board_and_actions([_one_real_game()])
    assert "board" in data and len(data["board"]) >= 1
    # every extracted array agrees on the number of samples
    n = len(data["board"])
    for value in data.values():
        assert len(value) == n


def test_sampling_layer_matches_mean_shape():
    z_mean = np.zeros((4, 8), dtype=np.float32)
    z_log_var = np.zeros((4, 8), dtype=np.float32)
    z = SamplingLayer()([z_mean, z_log_var])
    assert tuple(z.shape) == (4, 8)
