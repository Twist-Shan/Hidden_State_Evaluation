#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Server setup script for Hidden_State_Evaluation
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="hse"
PYTHON_VERSION="3.11"

echo ">>> Project root: ${PROJECT_ROOT}"

# 1. Install basic system packages
echo ">>> Installing basic system packages..."
sudo apt update
sudo apt install -y git curl wget unzip htop tmux build-essential pkg-config

# 2. Install Miniconda if conda not found
if ! command -v conda >/dev/null 2>&1; then
    echo ">>> Installing Miniconda..."
    cd "${HOME}"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "${HOME}/miniconda3"
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> "${HOME}/.bashrc"
    export PATH="${HOME}/miniconda3/bin:${PATH}"
    conda init bash || true
fi

# Load conda for non-interactive shell
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# 3. Create/update conda environment
cd "${PROJECT_ROOT}"

if [ -f "environment.yml" ]; then
    if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        echo ">>> Updating existing conda environment: ${ENV_NAME}"
        conda env update -n "${ENV_NAME}" -f environment.yml --prune
    else
        echo ">>> Creating conda environment from environment.yml."
        conda env create -f environment.yml
    fi
else
    echo ">>> No environment.yml found. Creating minimal environment: ${ENV_NAME}"
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"

# 4. Install project in editable mode
python -m pip install --upgrade pip setuptools wheel
if [ -f "pyproject.toml" ]; then
    python -m pip install -e .
fi

# 5. Register Jupyter kernel
python -m pip install ipykernel
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"

# 6. Create standard project directories
mkdir -p data checkpoints results paper_figs logs tmp

# 7. Optional: install Codex CLI
if ! command -v codex >/dev/null 2>&1; then
    echo ">>> Installing Codex CLI..."
    curl -fsSL https://chatgpt.com/codex/install.sh | sh || echo "Codex install failed, install manually later"
fi

# 8. Optional downloads: put wget/unzip commands here

echo ">>> Setup complete. Activate environment with:"
echo "    conda activate ${ENV_NAME}"