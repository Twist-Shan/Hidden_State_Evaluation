# Notebooks

This directory contains interactive notebooks for inspecting runs, producing figures, and developing analysis narratives. For reproducible batch runs, prefer the CLI scripts documented in [../README.md](../README.md).

High-level plans:

- [../experiment_pipeline_plan.md](../experiment_pipeline_plan.md)
- [../synthetic_transformer_experiment_plan.md](../synthetic_transformer_experiment_plan.md)

## Directory Layout

```text
notebooks/
  Dyck_Syn_to_Rea/       Synthetic-to-real counting experiments
  Dyck_Synthetic/        Dyck, Shuffle Dyck, Dyck-k, and CFG next-token notebooks
```

## Recommended Kernel

Use the `hse` environment/kernel:

```bash
conda env update -f environment.yml
conda activate hse
python -m ipykernel install --user --name hse
```

## Synthetic-to-Real Notebooks

| Notebook | Status | Purpose | Main outputs |
|---|---|---|---|
| `Dyck_Syn_to_Rea/Task_A_Length_Noise.ipynb` | Implemented | Synthetic Transformer plan Task A: measure how context length and noise affect behavior, linear count readability, output-head use, ablations, and activation patching. | `results/dyck_counter_task_a_summary.csv`, `figures/dyck_counter_task_a/`, `figures/dyck_counter_task_a_extra_probes/`, `figures/dyck_counter_task_a_ablation/`, `figures/dyck_counter_task_a_activation_patch/` |

Related scripts:

```bash
python scripts/summarize_task_a_results.py
python scripts/task_a_extra_probes.py
python scripts/task_a_height_ablation.py
python scripts/task_a_sparse_supervision_ablation.py
python scripts/task_a_layerwise_activation_patch.py --from-existing
```

## Synthetic Dyck Notebook Inventory

| Task family | Notebook | Status | Description |
|---|---|---|---|
| Dyck | `Dyck_Synthetic/Dyck_4Models_Probe_No_Noise.ipynb` | Runnable | Standard Dyck no-noise four-model probe workflow. |
| Dyck | `Dyck_Synthetic/Dyck_4Models_Probe_50_Noise.ipynb` | Runnable | Standard Dyck workflow with 50% generation probability and noise tokens. |
| Dyck | `Dyck_Synthetic/Dyck_4Models_Probe_No_Noise_Long.ipynb` | Runnable | Longer Dyck setting with geometry-oriented analysis. |
| Shuffle Dyck | `Dyck_Synthetic/Shuffle_Dyck_4Models_Probe_No_Noise.ipynb` | Runnable | Multi-stream bracket counting without noise. |
| Shuffle Dyck | `Dyck_Synthetic/Shuffle_Dyck_4Models_Probe_50_Noise.ipynb` | Runnable | Shuffle Dyck with noise. |
| Dyck-k | `Dyck_Synthetic/Dyck_k_4Models_Probe_No_Noise.ipynb` | Runnable | Stack-sensitive multi-bracket Dyck-k probes. |
| Dyck-k | `Dyck_Synthetic/Dyck_k_4Models_Probe_50_Noise.ipynb` | Runnable | Dyck-k with noise. |
| Dyck-k | `Dyck_Synthetic/Dyck_k_4Models_Probe_No_Noise_Long.ipynb` | Runnable | Longer Dyck-k setting. |
| Dyck-(2,3) CFG | `Dyck_Synthetic/Dyck23_CFG_4Models_Probe_Next_Token.ipynb` | Runnable | CFG-only valid strings, next-token prediction, length bins 0-40 / 42-80 / 82-120. |
| Dyck-(2,3) CFG | `Dyck_Synthetic/Dyck23_CFG_4Models_Probe_Next_Token_50_Noise.ipynb` | Runnable | Dyck-(2,3) CFG next-token experiment with noise. |
| Dyck-(2,5) CFG | `Dyck_Synthetic/Dyck25_CFG_4Models_Probe_Next_Token.ipynb` | Runnable | Dyck-(2,5) CFG next-token probe. |
| Dyck-2 CFG | `Dyck_Synthetic/Dyck2_CFG_Transformer_Probe_Next_Token_Len_0_400.ipynb` | Runnable | Transformer-only Dyck-2 CFG next-token experiment for lengths 0-400. |
| Dyck-2 CFG | `Dyck_Synthetic/Dyck2_CFG_Transformer_Probe_Next_Token_Fixed_Len_400.ipynb` | Runnable | Transformer-only Dyck-2 CFG next-token experiment at fixed length 400. |
| Dyck-2 CFG | `Dyck_Synthetic/Dyck2_CFG_Transformer_Probe_Next_Token_Len_0_2000.ipynb` | Runnable | Transformer-only Dyck-2 CFG next-token experiment for lengths 0-2000. |
| Dyck-2 Markov | `Dyck_Synthetic/Dyck2_Markov_Transformer_Probe_Next_Token_Len_0_400.ipynb` | Runnable | Markov-style Dyck-2 next-token comparison. |

## Planned Notebook Slots

The following items are part of the research plan but do not yet have notebook entry points in the current repository state.

| Plan item | Notebook | Outputs |
|---|---|---|
| Task B: checkpoint emergence analysis |  |  |
| Task C: direct final-count / JSON-like count answer |  |  |
| Task D: synthetic Needle-in-a-Haystack bridge |  |  |
| Task E: CoT / reasoning-token counting |  |  |
| Markov / HMM / VoMC hidden-state task |  |  |
| Realistic Needle-in-a-Haystack task |  |  |

Blank cells mean the notebook or output has not been created yet.

## Notebook Usage Policy

- Use notebooks for inspection, visualization, and narrative analysis.
- Use scripts for long-running or reproducible experiments.
- Write persistent outputs under `results/`, `figures/`, or `paper_figs/`.
- Keep one task setting per notebook where possible.
- Treat `rnn`, `lstm`, and `transformer` as stable baseline models.
- Use official `mamba-ssm` for formal Mamba comparisons.
- Treat `mamba_like` as an engineering fallback only.

## Common Workflow

For a new experiment, the intended flow is:

1. Add or edit a YAML config under `configs/`.
2. Run the experiment through `scripts/run_pipeline.py`.
3. Inspect saved `metrics.json`, hidden-state metadata, and `probes/layerwise_probe.csv`.
4. Add notebook analysis only after the run artifacts are reproducible from scripts.

Example:

```bash
python scripts/run_pipeline.py --config configs/dyck_no_noise.yaml --model transformer --seed 0
```

Then inspect:

```text
results/dyck_no_noise/transformer_seed0/
  metrics.json
  hidden_states/
  probes/layerwise_probe.csv
  probes/summary.json
```
