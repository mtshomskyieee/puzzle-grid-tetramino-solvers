# Source this before running any script in this repo:
#   source src/_env.sh
#
# It puts the engine + shared helpers + solver/replay module dirs on PYTHONPATH so the
# cross-directory imports resolve, and exports REPO_ROOT for data/model paths.
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(dirname "$SRC")"
export PYTHONPATH="$SRC/engine:$SRC/common:$SRC/solvers:$SRC/replay${PYTHONPATH:+:$PYTHONPATH}"

# Default to CPU (no CUDA driver required). Override with CUDA_VISIBLE_DEVICES=0 for GPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:--1}"
# 1 = hide TF INFO logs (still shows WARNING+); 2 = hide WARNING too.
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
