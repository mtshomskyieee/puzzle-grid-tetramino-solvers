# Solver notebooks

Three presentation / experiment notebooks — one per learned solver. Each one
loads the **shipped** weights, walks through a short training API (optional),
and visualizes *why* a move was chosen.

| Notebook | Solver | What to look at |
|---|---|---|
| [`01_vae_training_and_solver.ipynb`](01_vae_training_and_solver.ipynb) | VAE v2 | CNN feature maps, posterior \(z\), ranking by reconstruction |
| [`02_gan_training_and_solver.ipynb`](02_gan_training_and_solver.ipynb) | GAN | Generator proposal cloud, discriminator realism + line-clear head |
| [`03_transformer_training_and_solver.ipynb`](03_transformer_training_and_solver.ipynb) | Transformer scorer | Token map, attention overlays, holefit/surface ablation |

Shared helpers live in [`nb_setup.py`](nb_setup.py) (repo paths, board scenarios, drawing).
To regenerate the `.ipynb` files after editing the cell sources, run
`python notebooks/_build_ipynb.py`.

## Setup

From the package root:

```bash
uv sync --group notebooks
```

That installs `matplotlib`, `ipykernel`, and `ipywidgets` on top of the study
dependencies. In Cursor / VS Code, select the package `.venv` as the notebook
kernel. Then open any of the three files and run all cells.

Full training is still the CLI (`uv run train-vae`, `train-gan`,
`train-transformer`). Each notebook has a `RUN_SHORT_TRAINING = False` cell;
flip it only if you want a 1-epoch in-notebook demo on a tiny subset.

## Experiment knobs

In every notebook, a **lab cell** sets `PIECE` and `SCENARIO`. Re-run from that
cell downward.

Scenarios (defined in `nb_setup.py`): `empty`, `almost_line`, `right_well`,
`left_well`, `bumpy`, `flat_stack`, `overhang_pocket`.

Pieces: `I O T S Z J L`.
