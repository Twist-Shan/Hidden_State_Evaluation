# Experiment Workflow

This file converts the high-level plan into the actual order of operations for the current repo. The first complete runnable path is the Dyck baseline.

## Dyck Baseline

Use `configs/dyck_no_noise.yaml` to establish the simplest closed loop:

```bash
python scripts/run_pipeline.py --config configs/dyck_no_noise.yaml --model rnn --seed 0 --steps 200 --num-examples 512
```

For a more realistic run, remove the small overrides:

```bash
python scripts/run_pipeline.py --config configs/dyck_no_noise.yaml --model rnn --seed 0
```

## Pipeline Stages

### Stage 1: Train

- script: `scripts/train_model.py`
- task: causal next-token prediction on Dyck sequences
- output:
  - `checkpoints/model_final.pt`
  - `metrics.json`
  - `config.json`

### Stage 2: Extract

- script: `scripts/extract_hidden_states.py`
- reads the saved run directory
- rebuilds the task and model from `config.json`
- saves:
  - `hidden_states.pt`
  - `labels.parquet` or `labels.csv`

### Stage 3: Probe

- script: `scripts/run_probes.py`
- Dyck sufficient-statistic targets:
  - `left`
  - `right`
  - `height`
  - `height_class`
  - `left_right_class`
  - `legal_next_class`
- compression-related labels are evaluated in the same stage
- saves:
  - `probes/summary.json`
  - `probes/compression_probe_rows.csv`

### Stage 4: Geometry Check

- script: `scripts/analyze_geometry.py`
- current lightweight behavior:
  - print `summary.json`
- main sanity check:
  - whether `w_height` is aligned with `w_left - w_right`

## Stage-Specific Commands

Train only:

```bash
python scripts/run_pipeline.py --config configs/dyck_no_noise.yaml --model rnn --stage train
```

Extract only:

```bash
python scripts/run_pipeline.py --config configs/dyck_no_noise.yaml --model rnn --seed 0 --stage extract
```

Probe only:

```bash
python scripts/run_pipeline.py --config configs/dyck_no_noise.yaml --model rnn --seed 0 --stage probe
```

Geometry only:

```bash
python scripts/run_pipeline.py --config configs/dyck_no_noise.yaml --model rnn --seed 0 --stage geometry
```

## Model Policy

- RNN, LSTM, and Transformer are the stable baseline comparison set.
- Official `mamba` remains optional because it depends on `mamba-ssm`.
- `run_pipeline.py` skips official `mamba` automatically when that dependency is unavailable, so the Dyck baseline still launches.

## Next Expansion Order

1. Dyck no noise
2. Dyck 50% noise
3. Shuffle Dyck
4. Stronger compression analysis
5. Markov / HMM tasks
6. Needle-in-a-haystack
