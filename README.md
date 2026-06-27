# Hidden State Evaluation

Hidden State Evaluation is a research codebase for studying whether sequence models encode task-relevant sufficient statistics in their hidden states, and whether they discard task-irrelevant prefix information through effective compression.

The repository currently focuses on controlled synthetic sequence tasks, especially Dyck-style counting tasks. The main experimental question is not only whether a model predicts the next token correctly, but whether its internal state contains readable variables such as counts, stack depth, stack top, legal next-token classes, and other sufficient statistics.

## Research Motivation

The project is organized around two related claims:

1. A trained sequence model should internally represent the statistics that are sufficient for solving the task.
2. Good hidden-state compression should preserve task-relevant information while forgetting irrelevant prefix details such as noise tokens, distractors, or exact raw-token history.

The current pipeline is:

```text
task -> model -> train -> hidden-state extraction -> probes -> geometry/compression analysis
```

The broader plan is documented in:

- [experiment_pipeline_plan.md](experiment_pipeline_plan.md): the general sufficient-statistics and compression pipeline.
- [synthetic_transformer_experiment_plan.md](synthetic_transformer_experiment_plan.md): the Transformer-focused synthetic counting plan, including length/noise sweeps, emergence analysis, readout analysis, synthetic NIAH bridges, and reasoning-token experiments.

## Current Scope

The codebase supports matched experiments over these model families:

| Model | Status | Notes |
|---|---|---|
| RNN | Implemented | Basic recurrent baseline. |
| LSTM | Implemented | Gated recurrent baseline. |
| Transformer | Implemented | Main model for the synthetic Transformer counting experiments. |
| Mamba | Implemented when `mamba-ssm` is installed | Official Mamba is optional because it has extra CUDA/package constraints. |
| `mamba_like` | Implemented as fallback only | Engineering fallback for smoke checks, not a substitute for official Mamba results. |

Implemented task families:

| Task | Status | Main entry points | Main outputs |
|---|---|---|---|
| Dyck baseline | Implemented | `configs/dyck_no_noise.yaml`, `configs/dyck_noise.yaml` | `results/dyck_no_noise/`, `results/dyck_noise/` |
| Shuffle Dyck | Implemented | `configs/shuffle_dyck.yaml`, `configs/shuffle_dyck_noise.yaml` | `results/notebooks/shuffle_dyck_reference_style/` |
| Dyck-k | Implemented | `configs/dyck_k_no_noise.yaml`, `configs/dyck_k_50_noise.yaml`, `configs/dyck_k_long_no_noise.yaml` | `results/notebooks/dyck_k_reference_style/` |
| Dyck-(2,3) / Dyck-2 CFG next-token | Implemented | `configs/dyck23_cfg_next_token.yaml`, `scripts/run_dyck23.py` | `results/dyck23_cfg_next_token/`, `results/dyck2_cfg_transformer_next_token_*` |
| Task A: Transformer length/noise counting sweep | Implemented | `configs/generated_task_a/*.yaml`, `notebooks/Dyck_Syn_to_Rea/Task_A_Length_Noise.ipynb` | `results/dyck_counter_task_a_*`, `figures/dyck_counter_task_a*` |
| Task A extra probes, ablations, activation patching | Implemented | `scripts/task_a_extra_probes.py`, `scripts/task_a_height_ablation.py`, `scripts/task_a_layerwise_activation_patch.py` | `results/dyck_counter_task_a_extra_probes/`, `results/dyck_counter_task_a_ablation/`, `results/dyck_counter_task_a_activation_patch/` |
| Task B: checkpoint-level emergence analysis | Partially implemented | `configs/dyck_counter_emergence_transformer.yaml`, `scripts/run_pipeline.py` | `results/dyck_counter_emergence_transformer/` |
| Task C: direct final-count answer and readout alignment | Planned |  |  |
| Task D: synthetic Needle-in-a-Haystack counting bridge | Planned |  |  |
| Task E: CoT / reasoning-token counting | Planned |  |  |
| Markov / HMM / VoMC tasks | Planned |  |  |
| Needle-in-a-Haystack task module | Planned |  |  |

Blank entry/output cells mean the item is planned but not implemented in the current repository state.

## Installation

The recommended environment is the provided Conda environment:

```bash
conda env update -f environment.yml
conda activate hse
python -m ipykernel install --user --name hse
```

For a lighter editable install:

```bash
pip install -e .
```

Core dependencies are declared in [pyproject.toml](pyproject.toml) and [environment.yml](environment.yml). Typical runs require PyTorch, NumPy, pandas, scikit-learn, PyYAML, matplotlib, seaborn, tqdm, and pyarrow.

### Official Mamba

Formal four-model comparisons should use official `mamba-ssm` for the Mamba condition. On a Linux or WSL2 CUDA environment, install it separately:

```bash
pip install "causal-conv1d>=1.4.0" mamba-ssm --no-build-isolation
```

If official Mamba is unavailable, the CLI skips unavailable `mamba` runs unless a config explicitly requires them. The `mamba_like` fallback is useful for engineering smoke tests only.

## Quickstart

Run a small end-to-end Dyck pipeline:

```bash
python scripts/run_pipeline.py --config configs/dyck_no_noise.yaml --model rnn --seed 0 --steps 200 --num-examples 512
```

Run the default Dyck no-noise pipeline for one model and one seed:

```bash
python scripts/run_pipeline.py --config configs/dyck_no_noise.yaml --model rnn --seed 0
```

Run a Transformer setting from the Task A length/noise sweep:

```bash
python scripts/run_pipeline.py --config configs/generated_task_a/dyck_counter_task_a_sparse_long.yaml --model transformer --seed 0
```

Run the Dyck-(2,3) CFG next-token suite for selected models:

```bash
python scripts/run_dyck23.py --models rnn lstm transformer --seed 0
```

## Pipeline Stages

`scripts/run_pipeline.py` wraps the standard workflow:

1. Train a causal next-token model.
2. Extract hidden states from selected layers, positions, and checkpoints.
3. Fit linear probes for sufficient statistics.
4. Run geometry and compression summaries.

Stage-specific commands:

```bash
python scripts/run_pipeline.py --config configs/dyck_no_noise.yaml --model rnn --seed 0 --stage train
python scripts/run_pipeline.py --config configs/dyck_no_noise.yaml --model rnn --seed 0 --stage extract
python scripts/run_pipeline.py --config configs/dyck_no_noise.yaml --model rnn --seed 0 --stage probe
python scripts/run_pipeline.py --config configs/dyck_no_noise.yaml --model rnn --seed 0 --stage geometry
```

The individual scripts are also available:

| Script | Purpose |
|---|---|
| `scripts/train_model.py` | Train a model from a YAML config. |
| `scripts/extract_hidden_states.py` | Rebuild a trained run and save hidden-state tensors plus labels. |
| `scripts/run_probes.py` | Fit ridge/logistic probes and save probe summaries and directions. |
| `scripts/analyze_geometry.py` | Run lightweight probe-direction geometry checks. |
| `scripts/run_dyck23.py` | Run the Dyck-(2,3) CFG suite. |

## What the Probes Measure

For Dyck-style tasks, the most common labels are:

| Target | Type | Meaning |
|---|---|---|
| `left` | Regression | Number of opening brackets seen in the prefix. |
| `right` | Regression | Number of closing brackets seen in the prefix. |
| `height` | Regression | Current unmatched-count height, `left - right`. |
| `height_class` | Classification | Discrete height class. |
| `left_right_class` | Classification | Exact `(left, right)` prefix-count class. |
| `legal_next_class` | Classification | Legal next-token class under the task grammar. |

For Dyck-k, additional labels include stack depth, top-of-stack class, top-r stack summaries, and per-bracket-type counters. For Shuffle Dyck, labels include per-type heights and joint counter classes.

The main analysis asks whether these variables are linearly readable from hidden states, how the readable direction changes across layers/checkpoints, and whether irrelevant information is retained or forgotten.

## Task A: Transformer Length/Noise Counting Sweep

Task A is the most developed part of the synthetic Transformer plan. It tests how sequence length, bracket density, and noise affect both behavior and internal count readability.

Representative settings live in [configs/generated_task_a/](configs/generated_task_a/):

| Setting | Purpose |
|---|---|
| `clean_short` | Short clean Dyck counting baseline. |
| `noisy_short` | Short sequence with 50% generation probability and noise. |
| `sparse_medium` | Medium context with sparse bracket signal. |
| `sparse_long` | Long context with sparse bracket signal. |
| `extreme_long` | Very sparse long-context counting stress test. |
| `tiny_extreme_long` / `sparse_len2000_*` | Sparse-supervision and bracket-density controls. |

Task A helper scripts:

```bash
python scripts/summarize_task_a_results.py
python scripts/task_a_extra_probes.py
python scripts/task_a_height_ablation.py
python scripts/task_a_sparse_supervision_ablation.py
python scripts/task_a_layerwise_activation_patch.py --from-existing
```

These scripts write summary CSVs and figures under:

```text
results/dyck_counter_task_a_summary.csv
results/dyck_counter_task_a_extra_probes/
results/dyck_counter_task_a_ablation/
results/dyck_counter_sparse_supervision_ablation/
results/dyck_counter_task_a_activation_patch/
figures/dyck_counter_task_a*/
```

The main interactive entry point is [notebooks/Dyck_Syn_to_Rea/Task_A_Length_Noise.ipynb](notebooks/Dyck_Syn_to_Rea/Task_A_Length_Noise.ipynb).

## Repository Layout

```text
configs/                         YAML experiment configs
configs/generated_task_a/         Concrete Task A length/noise sweep configs
docs/                            Workflow notes and paper-related materials
figures/                         Script-generated analysis figures
notebooks/                       Interactive experiment notebooks
paper_figs/                      Paper-facing exported figures
results/                         Training outputs, hidden states, probes, summaries, HTML views
scripts/                         CLI entry points and Task A analysis scripts
src/hse/                         Python package
tests/                           Lightweight scaffold tests
experiment_pipeline_plan.md       General sufficient-statistics/compression plan
synthetic_transformer_experiment_plan.md  Synthetic Transformer counting plan
```

Important source modules:

```text
src/hse/models/simple.py          RNN, LSTM, Transformer, official/fallback Mamba wrappers
src/hse/tasks/registry.py         Task registry used by the CLI pipeline
src/hse/tasks/dyck/               Dyck sampler, labels, metrics
src/hse/tasks/shuffle_dyck/       Shuffle Dyck sampler and labels
src/hse/tasks/dyck_k/             Dyck-k stack task sampler and labels
src/hse/tasks/dyck23/             Dyck-(2,3) CFG sampler and labels
src/hse/tasks/markov/             Placeholder package
src/hse/tasks/needle/             Placeholder package
src/hse/analysis/probes/          Ridge/logistic probes and Dyck probe helpers
src/hse/analysis/compression/     Relevant-retention / irrelevant-forgetting probes
src/hse/analysis/geometry/        Probe-direction geometry checks
src/hse/experiments/              Notebook-friendly orchestration helpers
src/hse/utils/                    Config, IO, training, extraction utilities
```

## Output Layout

A standard run is written to:

```text
results/<experiment_name>/<model>_seed<seed>/
  config.json
  metrics.json
  checkpoints/
    model_final.pt
    model_best.pt
    model_step_<step>.pt
  hidden_states/
    final/
      layer_<layer>.pt
      labels.parquet
      metadata.json
  probes/
    layerwise_probe.csv
    checkpoint_probe.csv
    summary.json
    compression_probe_rows.csv
    directions/
```

Older notebook-oriented runs may also contain flat files such as `hidden_states.pt` and `labels.parquet` directly inside the run directory. The probe scripts handle both layouts where possible.

## Synthetic Transformer Plan Status

| Plan item | Current implementation | Missing pieces |
|---|---|---|
| A. Length and noise effects on counting | Implemented through Task A configs, Transformer runs, summary notebook, extra probes, height ablations, sparse-supervision ablations, and activation patching. | Systematic multi-seed main sweep still needs to be completed. |
| B. Emergence of internal counting features | Config and checkpoint extraction support exist in `dyck_counter_emergence_transformer.yaml` and `run_pipeline.py`. | Dedicated `analyze_emergence.py`, `probe_emergence_step`, `behavior_emergence_step`, and `verbalization_lag` summaries are not implemented yet. |
| C. From hidden count to output/readout | Task A extra probes include partial output-head-use analysis. | Direct final-count task, `NUM_k` answer tokens, unembedding alignment, and logit-lens count analysis are not implemented yet. |
| D. Synthetic NIAH bridge |  |  |
| E. CoT / reasoning-token counting |  |  |

## Notebook Guide

The notebook inventory is maintained in [notebooks/README.md](notebooks/README.md). The main groups are:

- `notebooks/Dyck_Syn_to_Rea/`: synthetic-to-real counting experiments, currently centered on Task A.
- `notebooks/Dyck_Synthetic/`: Dyck, Shuffle Dyck, Dyck-k, Dyck CFG, and Dyck-2 next-token notebooks.

For reproducible runs, prefer the scripts. For inspection, visualization, and writing analysis, use the notebooks.

## Development Notes

- Use `rg` or `rg --files` to inspect files quickly.
- Keep generated experiment artifacts under `results/`, `figures/`, or `paper_figs/`.
- Keep new tasks registered in `src/hse/tasks/registry.py` if they should work with `scripts/run_pipeline.py`.
- Add probe labels close to the task sampler so hidden-state extraction and probing stay reproducible.

## Tests

The repository currently has lightweight scaffold tests:

```bash
pytest
```

These tests check package structure rather than full experimental correctness. The primary validation for research results is the train/extract/probe workflow plus saved metrics and probe summaries.
