# Experiment Workflow

This document turns `experiment_pipeline_plan.md` into an implementation checklist.

## Phase 1: Dyck Baseline

- Implement Dyck sampler and prefix labels.
- Train RNN, LSTM, Transformer, and Mamba on next-token prediction.
- Extract last-layer hidden states with the architecture-specific `state_kind`.
- Probe `left`, `right`, `height`, `height_class`, and `(left, right)`.
- Check whether `w_height` aligns with `w_left - w_right`.

## Phase 2: Shuffle Dyck

- Extend labels to per-type counters.
- Probe per-type heights and joint count-vector classes.
- Test whether separate bracket counters occupy separate directions or subspaces.

## Phase 3: Compression

- Add task-irrelevant labels for noise tokens, distractor identities, and exact prefix details.
- Report relevant retention and irrelevant forgetting separately.

## Phase 4: Realistic Tasks

- Add needle-in-a-haystack retrieval.
- Compare controlled sufficient-statistic probes with retrieval-specific probes.
