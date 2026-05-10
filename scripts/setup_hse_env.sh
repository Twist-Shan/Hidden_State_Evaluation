#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

conda env update -f "$ROOT_DIR/environment.yml" --prune
conda run -n hse python -m pip install torch
conda install -n hse -c conda-forge -y cuda-nvcc
TORCH_CUDA_ARCH_LIST=8.9 conda run -n hse python -m pip install "causal-conv1d>=1.4.0" mamba-ssm --no-build-isolation
conda run -n hse python -m ipykernel install --user --name hse --display-name "Python (hse)"
