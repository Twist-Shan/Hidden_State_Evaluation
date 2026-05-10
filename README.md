# Hidden State Evaluation

This project studies whether RNN, LSTM, Transformer, and Mamba hidden states encode task-relevant sufficient statistics, and whether they discard task-irrelevant prefix information through effective compression.

The project is now organized around the pipeline in [experiment_pipeline_plan.md](/home/hp_twist_shan/Research/Hidden%20State%20Evaluation/experiment_pipeline_plan.md):

`task -> 4 matched models -> train -> hidden-state extraction -> probes -> geometry/compression`

## Research Scope

The target comparison is:

- `rnn`
- `lstm`
- `transformer`
- `mamba`

The main task expansion order is:

1. Dyck
2. Shuffle Dyck
3. Markov / HMM / VoMC
4. Needle in a Haystack

At the moment, the first fully runnable task is `Dyck`.

## Task-First Notebook Layout

The notebook workflow is now task-based rather than one giant mixed notebook.

| Task | Notebook | Status |
|---|---|---|
| Dyck | [notebooks/Dyck_4Models_Probe.ipynb](/home/hp_twist_shan/Research/Hidden%20State%20Evaluation/notebooks/Dyck_4Models_Probe.ipynb) | Runnable |
| Shuffle Dyck | planned | Placeholder |
| Markov / HMM | planned | Placeholder |
| Needle in a Haystack | planned | Placeholder |

If you want to start immediately, open the Dyck notebook and run it cell by cell.

## Mamba Setup

For the formal 4-model comparison, the notebook expects the official `mamba-ssm` package.

Recommended environment:

- Linux
- or WSL2 Ubuntu
- NVIDIA GPU available to PyTorch
- CUDA toolkit available in the environment used to install `mamba-ssm`

Recommended install command:

```bash
pip install -e .
pip install "causal-conv1d>=1.4.0" mamba-ssm --no-build-isolation
```

The Dyck notebook includes a preflight check:

- if official `mamba-ssm` is available, it runs `mamba`
- if it is missing and `REQUIRE_OFFICIAL_MAMBA = True`, it fails early with an explicit installation message
- if you intentionally want a non-official smoke fallback, set `FALLBACK_TO_MAMBA_LIKE = True`

Important:

- `mamba_like` is only an engineering fallback
- it is not a substitute for official Mamba in the actual comparison

## Install

Base project install:

```bash
pip install -e .
```

Main dependencies are declared in [pyproject.toml](/home/hp_twist_shan/Research/Hidden%20State%20Evaluation/pyproject.toml).

## Scripts

The script pipeline is still available when you want non-notebook runs:

```bash
python scripts/run_pipeline.py --config configs/dyck_no_noise.yaml --model rnn
python scripts/train_model.py --config configs/dyck_no_noise.yaml --model rnn --seed 0
python scripts/extract_hidden_states.py --run results/dyck_no_noise/rnn_seed0
python scripts/run_probes.py --features results/dyck_no_noise/rnn_seed0/hidden_states.pt
```

## Project Layout

```text
src/hse/tasks/dyck/             Dyck sampler, labels, metrics
src/hse/models/                 RNN, LSTM, Transformer, Mamba wrappers
src/hse/analysis/probes/        Linear probes
src/hse/analysis/compression/   Relevant retention / irrelevant forgetting
src/hse/experiments/dyck.py     Notebook-friendly Dyck runner
configs/                        Task configs
notebooks/                      Task notebooks
scripts/                        CLI pipeline entry points
results/                        Checkpoints, hidden states, labels, probe summaries
docs/                           Notes and workflow docs
```

## Current Status

Implemented now:

- Dyck sampler and labels
- four-family model interface
- notebook-friendly Dyck runner
- task-based Dyck notebook
- hidden-state extraction
- sufficient-statistic probes
- compression summaries

Still to implement:

- Shuffle Dyck task module
- Markov / HMM task module
- Needle task module
- richer geometry visualizations
