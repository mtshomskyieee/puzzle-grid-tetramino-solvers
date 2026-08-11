"""Console entry points so the whole workflow runs via ``uv run <command>``.

Each command reuses an existing script's argparse ``main()``/``__main__`` block unchanged:
the launcher puts the split module dirs on ``sys.path``, changes to the repo root (so the
scripts' ``./data`` and ``./models`` defaults resolve), then runs the target with runpy.
Server-dependent commands auto-start the API server, run, then stop it.

Extra CLI args pass straight through and override any injected defaults, e.g.
``uv run play-vae --max-moves 50`` or ``uv run generate-data --games 5``.
"""
import os
import pathlib
import runpy
import subprocess
import sys
import time
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
_MODULE_DIRS = [SRC / "engine", SRC / "common", SRC / "solvers", SRC / "replay"]
_PORT = 8000


def _setup_env():
    for d in _MODULE_DIRS:
        p = str(d)
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    os.chdir(REPO_ROOT)


def _run_script(rel, extra_args=()):
    """Run a script as ``__main__`` with injected defaults followed by user args."""
    _setup_env()
    script = str(REPO_ROOT / rel)
    sys.argv = [script, *extra_args, *sys.argv[1:]]
    runpy.run_path(script, run_name="__main__")


def _server_up(url):
    try:
        with urllib.request.urlopen(url, timeout=1):
            return True
    except Exception:
        return False


def _with_server(rel, extra_args=(), port=_PORT):
    """Start the API server if it isn't running, run the script, then stop it."""
    _setup_env()
    url = f"http://localhost:{port}/"
    proc = None
    if _server_up(url):
        print(f"Using API server already running at {url}")
    else:
        print(f"Starting API server on port {port} ...")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(d) for d in _MODULE_DIRS] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "puzzlegrid.api:app",
             "--host", "127.0.0.1", "--port", str(port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
        )
        for _ in range(30):
            if _server_up(url):
                break
            if proc.poll() is not None:
                raise SystemExit(
                    f"API server exited early (is port {port} already in use?)."
                )
            time.sleep(1)
        else:
            proc.terminate()
            raise SystemExit("API server did not become ready within 30s.")
    try:
        _run_script(rel, extra_args)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            print("API server stopped.")


# --- engine --------------------------------------------------------------------------
def engine():
    _setup_env()
    import uvicorn
    uvicorn.run("puzzlegrid.api:app", host="0.0.0.0", port=_PORT)


# --- step 1: data --------------------------------------------------------------------
def generate_data():
    _with_server(
        "src/data_gen/heuristic-data-generator.py",
        ["--pieces", "I", "O", "T", "S", "Z", "J", "L",
         "--games", "200", "--max-lines", "80",
         "--output-dir", "./data/training_data_clean"],
    )


def validate_data():
    _with_server("src/data_gen/validate_training_data.py")


def analyze_data():
    _run_script("src/data_gen/analyze_training_data.py")


# --- steps 2-4: train ----------------------------------------------------------------
def train_vae():
    _run_script("src/solvers/training_vae_v2.py")


def train_vae1():
    _run_script("src/solvers/training_vae_v1.py")


def train_gan():
    _run_script("src/solvers/training_gan.py")


def train_transformer():
    _run_script("src/solvers/train_transformers.py")


def train_policy():
    _run_script("src/solvers/train_transformer_policy.py")


def distill():
    _run_script("src/solvers/distill_policy_data.py")


# --- steps 2-5: play -----------------------------------------------------------------
def play_vae():
    _with_server("src/replay/replay_vae_v2.py")


def play_vae1():
    _with_server("src/replay/replay_vae.py")


def play_gan():
    _with_server("src/replay/replay_gan.py")


def play_transformer():
    _with_server("src/replay/replay_transformers.py")


def play_policy():
    _run_script("src/replay/replay_transformer_policy.py", ["--inprocess"])


def play_agents():
    _with_server("src/replay/replay_using_agents.py")
