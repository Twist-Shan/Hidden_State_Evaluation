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
| Shuffle Dyck | [notebooks/Shuffle_Dyck_4Models_Probe_No_Noise.ipynb](/home/hp_twist_shan/Research/Hidden%20State%20Evaluation/notebooks/Shuffle_Dyck_4Models_Probe_No_Noise.ipynb) | Runnable |
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

## Project Layout

```text
experiment_pipeline_plan.md     Pipeline design note that drives the project organization
environment.yml                 Conda environment definition for notebook and CLI runs
configs/                        YAML configs for Dyck variants and future task settings
docs/                           Workflow notes and conversation summaries
notebooks/                      Task-level notebooks; Dyck is the current runnable entrypoint
paper_figs/                     Saved figures for paper-facing plots and exports
results/                        Experiment outputs, checkpoints, extracted states, and probe summaries
scripts/                        CLI entry points for train / extract / probe / geometry / env setup
src/hse/                        Python package root
src/hse/models/simple.py        Current model implementations for RNN, LSTM, Transformer, and Mamba
src/hse/tasks/dyck/             Implemented Dyck task: config, sampler, labels, and metrics
src/hse/tasks/shuffle_dyck/     Implemented Shuffle Dyck task with interleaved bracket streams and multi-counter labels
src/hse/tasks/markov/           Placeholder package for future Markov / HMM experiments
src/hse/tasks/needle/           Placeholder package for future needle-style tasks
src/hse/analysis/probes/        Linear probe code and Dyck-specific probe helpers
src/hse/analysis/compression/   Relevant-retention / irrelevant-forgetting metrics
src/hse/analysis/geometry/      Direction and geometry analysis utilities
src/hse/analysis/visualization/ Reserved package for future plotting helpers
src/hse/experiments/dyck.py     Notebook-friendly Dyck orchestration over models, seeds, and probes
src/hse/utils/                  Shared training, extraction, config, IO, and label-loading helpers
tests/                          Lightweight scaffold tests for package structure
```

Practical reading of the layout:

- `notebooks/` is the human-facing entrypoint. If you want to run experiments interactively, start there.
- `src/hse/experiments/` is the notebook support layer. It packages the train-extract-probe flow into reusable helpers.
- `scripts/` is the CLI mirror of the same workflow. Use it when you want reproducible batch runs outside notebooks.
- `src/hse/tasks/` defines task generation and labels. Right now only `dyck/` is implemented; the other task directories are scaffolds.
- `src/hse/models/simple.py` currently contains all four model families in one file. So `src/hse/models/` exists, but it is not yet split into per-model modules.
- `results/` is expected to grow quickly. It holds both notebook outputs and script-generated runs, so it is part of the working tree rather than just a scratch folder.

## Current Status

Implemented now:

- Dyck sampler and labels
- Shuffle Dyck sampler and labels
- four-family model interface
- notebook-friendly Dyck runner
- task-based Dyck notebook
- task-based Shuffle Dyck notebook
- hidden-state extraction
- sufficient-statistic probes
- compression summaries

Still to implement:

- Markov / HMM task module
- Needle task module
- richer geometry visualizations
