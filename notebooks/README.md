# Notebooks

This directory is organized by task, following `experiment_pipeline_plan.md`.

## Current Notebook Entry Points

| Task | Notebook | Status | Notes |
|---|---|---|---|
| Dyck | `Dyck_4Models_Probe.ipynb` | Runnable | Main notebook for the 4-model comparison and probes |
| Shuffle Dyck | planned | Not implemented yet | Task module is still a placeholder |
| Markov / HMM | planned | Not implemented yet | Task module is still a placeholder |
| Needle in a Haystack | planned | Not implemented yet | Task module is still a placeholder |

## Recommended Use

Start here:

- `Dyck_4Models_Probe.ipynb`

This notebook is designed to:

- choose one Dyck task setting at a time
- train `rnn`, `lstm`, `transformer`, and official `mamba`
- extract hidden states
- run sufficient-statistic probes
- write results under `results/notebooks/dyck/<task_name>/`

## Legacy Notebook

`Dyck_All_Models_Train_Probe_Compression.ipynb` is kept as an older all-in-one notebook. Prefer `Dyck_4Models_Probe.ipynb` for the current task-based workflow.
