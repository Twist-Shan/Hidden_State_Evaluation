from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from hse.tasks.dyck.config import DyckConfig
from hse.tasks.dyck.labels import build_prefix_labels
from hse.tasks.dyck.sampler import DyckBatch, DyckSampler
from .labels_io import save_labels


@torch.no_grad()
def extract_hidden_states(
    *,
    model: nn.Module,
    sampler: DyckSampler,
    state_kind: str = "h",
    layer: int = -1,
    num_examples: int = 4096,
    batch_size: int = 512,
    max_prefix_len: int | None = None,
    device: str | torch.device = "cpu",
    run_dir: str | Path | None = None,
) -> tuple[torch.Tensor, pd.DataFrame]:
    """Extract row-aligned hidden states and labels for linear probes."""
    device = torch.device(device)
    model.to(device)
    model.eval()
    all_hidden = []
    all_labels = []
    seen = 0

    while seen < num_examples:
        current = min(batch_size, num_examples - seen)
        batch = sampler.sample(current)
        tokens = batch.tokens.to(device)
        states = model.extract_states(tokens, layer_index=layer, state_kind=state_kind).detach().cpu()
        labels = build_prefix_labels(
            _batch_to_cpu(batch),
            config=sampler.config,
            max_prefix_len=None,
        )
        labels["example_id"] += seen
        hidden_rows = flatten_states_for_labels(states, labels)
        if max_prefix_len is not None:
            keep = labels["dyck_seen"].to_numpy() <= max_prefix_len
            labels = labels.loc[keep].reset_index(drop=True)
            hidden_rows = hidden_rows[torch.tensor(keep, dtype=torch.bool)]
        all_hidden.append(hidden_rows)
        all_labels.append(labels)
        seen += current

    hidden = torch.cat(all_hidden, dim=0)
    labels_df = pd.concat(all_labels, ignore_index=True)
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(hidden, run_dir / "hidden_states.pt")
        save_labels(labels_df, run_dir / "labels")
    return hidden, labels_df


def flatten_states_for_labels(states: torch.Tensor, labels: pd.DataFrame) -> torch.Tensor:
    """Select `states[example_id, position]` for each row in labels."""
    example_ids = torch.tensor(labels["example_id"].to_numpy(), dtype=torch.long)
    # Labels passed to this helper are batch-local unless the caller has already
    # offset example_id. Bring them back to local ids for indexing.
    example_ids = example_ids - int(example_ids.min().item())
    positions = torch.tensor(labels["position"].to_numpy(), dtype=torch.long)
    return states[example_ids, positions].contiguous()


def _batch_to_cpu(batch: DyckBatch) -> DyckBatch:
    return DyckBatch(
        tokens=batch.tokens.detach().cpu(),
        dyck_mask=batch.dyck_mask.detach().cpu(),
        dyck_steps=batch.dyck_steps.detach().cpu(),
        noise_tokens=batch.noise_tokens.detach().cpu(),
    )
