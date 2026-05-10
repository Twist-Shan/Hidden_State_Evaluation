# Hidden State Evaluation

This repo studies whether hidden states encode task-relevant sufficient statistics, and whether different architectures preserve useful information while forgetting irrelevant prefix details. The current runnable entry point starts from the Dyck task and follows a fixed pipeline:

`train -> extract hidden states -> run probes -> inspect geometry/compression`

The two reference documents are:

- [experiment_pipeline_plan.md](/home/hp_twist_shan/Research/Hidden%20State%20Evaluation/experiment_pipeline_plan.md)
- [docs/codex_conversation_summary.md](/home/hp_twist_shan/Research/Hidden%20State%20Evaluation/docs/codex_conversation_summary.md)

## Setup

Install the project dependencies before running the pipeline:

```bash
pip install -e .
```

## Start Here

If you want the smallest end-to-end experiment, start with Dyck and one model:

```bash
python scripts/run_pipeline.py --config configs/dyck_no_noise.yaml --model rnn --seed 0 --steps 200 --num-examples 512
```

That command will:

1. train `rnn_seed0`
2. save a checkpoint and metrics under `results/dyck_no_noise/rnn_seed0`
3. extract hidden states plus aligned Dyck prefix labels
4. run sufficient-statistic and compression probes
5. print the saved probe summary through the geometry stage

If you want the full configured model sweep for the same task:

```bash
python scripts/run_pipeline.py --config configs/dyck_no_noise.yaml
```

If `mamba-ssm` is not installed, `run_pipeline.py` will skip the official `mamba` model automatically so the Dyck pipeline can still start cleanly.

## Current Pipeline

The implemented runnable baseline is Dyck.

### 1. Train

- Script: `scripts/train_model.py`
- Input: a YAML config under `configs/`
- Output: `results/<experiment>/<model>_seed<seed>/checkpoints/model_final.pt`

### 2. Extract Hidden States

- Script: `scripts/extract_hidden_states.py`
- Output files:
  - `hidden_states.pt`
  - `labels.parquet` or `labels.csv`

The extraction uses the architecture-specific hidden state:

- RNN: last-layer `h`
- LSTM: last-layer `c`
- Transformer: last-layer `h`
- Mamba / `mamba_like`: last-layer `h`

### 3. Run Probes

- Script: `scripts/run_probes.py`
- Output directory: `results/<experiment>/<model>_seed<seed>/probes/`

Current Dyck probes include:

- Ridge targets: `left`, `right`, `height`
- Classification targets: `height_class`, `left_right_class`, `legal_next_class`
- Compression-style summary over relevant vs irrelevant labels

### 4. Geometry / Summary Check

- Script: `scripts/analyze_geometry.py`
- Current behavior: print `probes/summary.json`

The main geometry sanity check is whether the learned `height` direction aligns with `w_left - w_right`.

## Recommended Dyck Runs

### Dyck no noise

Config: `configs/dyck_no_noise.yaml`

- `dyck_pairs = 24`
- `total_length = 48`
- `seq_len = 48`
- `repeat_prob = 1.0`
- cleanest starting point for sufficient-statistic probing

### Dyck 50% noise

Config: `configs/dyck_noise.yaml`

- `dyck_pairs = 10`
- `total_length = 20`
- `seq_len = 48`
- `repeat_prob = 0.5`
- first compression-oriented setting

## Script Usage

Run the whole Dyck pipeline:

```bash
python scripts/run_pipeline.py --config configs/dyck_no_noise.yaml --model rnn
```

Run only training:

```bash
python scripts/run_pipeline.py --config configs/dyck_no_noise.yaml --model rnn --stage train
```

Resume from an existing run and only do extraction:

```bash
python scripts/run_pipeline.py --config configs/dyck_no_noise.yaml --model rnn --seed 0 --stage extract
```

Run probes on an existing extracted run:

```bash
python scripts/run_pipeline.py --config configs/dyck_no_noise.yaml --model rnn --seed 0 --stage probe
```

Run the stages manually if needed:

```bash
python scripts/train_model.py --config configs/dyck_no_noise.yaml --model rnn --seed 0
python scripts/extract_hidden_states.py --run results/dyck_no_noise/rnn_seed0
python scripts/run_probes.py --features results/dyck_no_noise/rnn_seed0/hidden_states.pt
python scripts/analyze_geometry.py --probe-dir results/dyck_no_noise/rnn_seed0/probes
```

## Repo Layout

```text
src/hse/tasks/dyck/           Dyck sampler, labels, metrics
src/hse/models/               RNN, LSTM, Transformer, Mamba wrappers
src/hse/analysis/probes/      Linear sufficient-statistic probes
src/hse/analysis/compression/ Relevant retention / irrelevant forgetting
scripts/                      Reproducible experiment entry points
configs/                      Dyck and Shuffle-Dyck configs
results/                      Checkpoints, states, labels, probe summaries
docs/                         Workflow notes and conversation summary
```

## What Is Implemented vs Planned

Implemented now:

- Dyck training
- hidden-state extraction
- sufficient-statistic probes
- compression summaries
- one-command pipeline orchestration through `scripts/run_pipeline.py`

Planned next:

- Shuffle Dyck
- stronger geometry outputs and figures
- Markov / HMM tasks
- Needle-in-a-haystack style realistic tasks
