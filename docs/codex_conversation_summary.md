# Codex Conversation Summary

Date: 2026-05-10

## Project Goal

Build a Hidden State Evaluation project based on the structure of `Task Vector Geometry/mini-ICL-main`, focused on whether RNN, LSTM, Transformer, and Mamba models encode sufficient statistics and compress irrelevant historical information on Dyck-style tasks.

## Implemented Structure

The project now contains a runnable research scaffold:

- `src/hse/tasks/dyck`: Dyck sampler, labels, and metrics.
- `src/hse/models`: RNN, LSTM, Transformer, official-Mamba-aware builder, and `mamba_like` fallback.
- `src/hse/utils`: training, evaluation, hidden-state extraction, JSON/tensor/label IO.
- `src/hse/analysis/probes`: Ridge/logistic probes with torch/numpy fallback.
- `src/hse/analysis/geometry`: probe direction geometry, especially `w_height` vs `w_left - w_right`.
- `src/hse/analysis/compression`: relevant retention and irrelevant forgetting probes.
- `configs`: Dyck no-noise, Dyck 50% noise, and Shuffle Dyck config templates.
- `scripts`: train, extract hidden states, run probes, and pipeline entry points.
- `notebooks/Dyck_All_Models_Train_Probe_Compression.ipynb`: runnable end-to-end notebook.

## Dyck Experiments

The notebook supports:

- `dyck_no_noise`
  - `repeat_prob = 1.0`
  - `total_length = 48`
  - `seq_len = 48`
- `dyck_50_noise`
  - `repeat_prob = 0.5`
  - `total_length = 20`
  - `seq_len = 48`

It runs:

- RNN
- LSTM
- Transformer
- Official Mamba if `mamba-ssm` is available
- `mamba_like` fallback if official Mamba is unavailable

The default notebook settings are small smoke-test values:

```python
SEEDS = [0]
TRAINING_STEPS = 5
BATCH_SIZE = 16
EXTRACT_EXAMPLES = 64
```

For real experiments, use:

```python
SEEDS = [0, 1, 2]
TRAINING_STEPS = 10_000
BATCH_SIZE = 128
EXTRACT_EXAMPLES = 50_000
EXTRACT_BATCH_SIZE = 512
```

## Probe Design

Relevant sufficient-statistic probes:

- Ridge: `left`, `right`, `height`
- Logistic: `height_class`, `left_right_class`, `legal_next_class`

Geometry check:

- Compare `w_height` with `w_left - w_right`

Compression probes:

- Relevant retention: mean normalized score on sufficient-statistic targets.
- Irrelevant decodability: mean normalized score on irrelevant prefix details.
- Irrelevant forgetting: `1 - irrelevant_decodability`.

Irrelevant labels currently include:

- `last_noise_token_class`
- `distant_noise_token_class`
- `noise_pattern_hash_class`
- `random_marker_class`

## Mamba Status

Attempted to install official `mamba-ssm` on Windows native Anaconda Python:

```bash
python -m pip install causal-conv1d>=1.4.0 mamba-ssm --no-build-isolation
python -m pip install mamba-ssm --no-build-isolation
```

Both failed because the current environment is Windows native and has no `nvcc`.

Observed environment:

- Python: 3.12.7
- Platform: Windows 11
- Torch: 2.5.1+cu121
- CUDA available through PyTorch: yes
- `nvcc`: not found

Official `mamba-ssm` targets Linux + NVIDIA GPU + CUDA. The recommended path is to run the project in Linux/WSL2 with CUDA toolkit available, then install:

```bash
pip install causal-conv1d>=1.4.0 mamba-ssm --no-build-isolation
```

The code now distinguishes:

- `model_name = "mamba"`: official `mamba-ssm`, with `require_official_mamba=True`.
- `model_name = "mamba_like"`: local fallback, useful only for smoke tests.

## Verification Already Done

The following checks passed in the Windows environment:

```bash
pytest -q
```

Result:

```text
2 passed
```

The notebook was executed successfully with local Jupyter using project-local Jupyter config/runtime directories. It ran through the smoke experiment.

## WSL Migration Status

The user requested moving the project and this conversation context into WSL. At the time of this summary, `wsl -l -v` reported no installed WSL distributions. The next step is to install a WSL distro, preferably Ubuntu, and then copy this project into the Linux filesystem.
