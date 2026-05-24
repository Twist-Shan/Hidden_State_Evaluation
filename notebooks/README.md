# Notebooks

This directory is organized by task, following `experiment_pipeline_plan.md`.

## Current Notebook Entry Points

| Task | Notebook | Status | Notes |
|---|---|---|---|
| Dyck | `Dyck_4Models_Probe_No_Noise.ipynb` and `Dyck_4Models_Probe_50_Noise.ipynb` | Runnable | Dyke for the 4-model comparison and probes |
| Dyck-k | `Dyck_k_4Models_Probe_No_Noise.ipynb`, `Dyck_k_4Models_Probe_50_Noise.ipynb`, and `Dyck_k_4Models_Probe_No_Noise_Long.ipynb` | Runnable | Stack-sensitive multi-bracket Dyck-k probes for the 4-model comparison, including the long 210-pair exact-quota setting |
| Shuffle Dyck | `Shuffle_Dyck_4Models_Probe_No_Noise.ipynb` and `Shuffle_Dyck_4Models_Probe_50_Noise.ipynb` | Runnable | Shuffle Dyck with three bracket types for the 4-model comparison and probes |
| Markov / HMM | planned | Not implemented yet | Task module is still a placeholder |
| Needle in a Haystack | planned | Not implemented yet | Task module is still a placeholder |

## Recommended Use

These notebooks are designed to:

- choose one task setting at a time
- train `rnn`, `lstm`, `transformer`, and official `mamba`
- extract hidden states
- run sufficient-statistic probes
- write results under `results/notebooks/dyck/<task_name>/`
