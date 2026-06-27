from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from hse.tasks.registry import batch_to_cpu, build_labels
from .io import save_json
from .labels_io import save_labels


@torch.no_grad()
def extract_hidden_states(
    *,
    model: nn.Module,
    sampler,
    task_name: str = "dyck",
    state_kind: str = "h",
    layer: int = -1,
    layers: str | int | list[int] | tuple[int, ...] | None = None,
    num_examples: int = 4096,
    batch_size: int = 512,
    max_prefix_len: int | None = None,
    position_mode: str = "prefix",
    device: str | torch.device = "cpu",
    run_dir: str | Path | None = None,
    checkpoint_name: str = "final",
    write_legacy_files: bool = True,
) -> tuple[torch.Tensor | dict[int, torch.Tensor], pd.DataFrame]:
    """Extract row-aligned hidden states and labels for linear probes."""
    device = torch.device(device)
    model.to(device)
    model.eval()
    selected_layers = normalize_layers(layers if layers is not None else layer, num_layers=int(model.num_layers))
    all_hidden: dict[int, list[torch.Tensor]] = {layer_index: [] for layer_index in selected_layers}
    all_labels = []
    seen = 0

    while seen < num_examples:
        current = min(batch_size, num_examples - seen)
        batch = sampler.sample(current)
        tokens = batch.tokens.to(device)
        states_by_layer = extract_layer_states(model, tokens, selected_layers, state_kind=state_kind)
        labels = build_labels(task_name, batch_to_cpu(batch), sampler.config, max_prefix_len=None)
        labels["example_id"] += seen
        keep = label_position_mask(labels, mode=position_mode, max_prefix_len=max_prefix_len)
        labels = labels.loc[keep].reset_index(drop=True)
        for layer_index, states in states_by_layer.items():
            hidden_rows = flatten_states_for_labels(states, labels)
            all_hidden[layer_index].append(hidden_rows)
        all_labels.append(labels)
        seen += current

    hidden_by_layer = {layer_index: torch.cat(parts, dim=0) for layer_index, parts in all_hidden.items()}
    labels_df = pd.concat(all_labels, ignore_index=True)
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        canonical_dir = run_dir / "hidden_states" / checkpoint_name
        canonical_dir.mkdir(parents=True, exist_ok=True)
        for layer_index, hidden in hidden_by_layer.items():
            torch.save(hidden, canonical_dir / f"layer_{layer_index}.pt")
        save_labels(labels_df, canonical_dir / "labels")
        save_json(
            {
                "checkpoint": checkpoint_name,
                "task_name": task_name,
                "state_kind": state_kind,
                "layers": selected_layers,
                "position_mode": position_mode,
                "max_prefix_len": max_prefix_len,
                "num_examples": num_examples,
                "hidden_rows": int(labels_df.shape[0]),
            },
            canonical_dir / "metadata.json",
        )
        if write_legacy_files and checkpoint_name == "final" and len(selected_layers) == 1:
            torch.save(hidden_by_layer[selected_layers[0]], run_dir / "hidden_states.pt")
            save_labels(labels_df, run_dir / "labels")
    if len(selected_layers) == 1:
        return hidden_by_layer[selected_layers[0]], labels_df
    return hidden_by_layer, labels_df


def flatten_states_for_labels(states: torch.Tensor, labels: pd.DataFrame) -> torch.Tensor:
    """Select `states[example_id, position]` for each row in labels."""
    example_ids = torch.tensor(labels["example_id"].to_numpy(), dtype=torch.long)
    # Labels passed to this helper are batch-local unless the caller has already
    # offset example_id. Bring them back to local ids for indexing.
    example_ids = example_ids - int(example_ids.min().item())
    positions = torch.tensor(labels["position"].to_numpy(), dtype=torch.long)
    return states[example_ids, positions].contiguous()


def normalize_layers(layers: str | int | list[int] | tuple[int, ...], *, num_layers: int) -> list[int]:
    if isinstance(layers, str):
        if layers == "all":
            return list(range(num_layers))
        raw_layers = [int(item.strip()) for item in layers.split(",") if item.strip()]
    elif isinstance(layers, int):
        raw_layers = [layers]
    else:
        raw_layers = [int(layer) for layer in layers]
    normalized = []
    for layer in raw_layers:
        layer_index = num_layers + layer if layer < 0 else layer
        if layer_index < 0 or layer_index >= num_layers:
            raise ValueError(f"Layer index {layer} resolves to {layer_index}, outside [0, {num_layers})")
        if layer_index not in normalized:
            normalized.append(layer_index)
    return normalized


@torch.no_grad()
def extract_layer_states(
    model: nn.Module,
    tokens: torch.Tensor,
    layers: list[int],
    *,
    state_kind: str,
) -> dict[int, torch.Tensor]:
    if len(layers) == 1:
        layer_index = layers[0]
        return {
            layer_index: model.extract_states(tokens, layer_index=layer_index, state_kind=state_kind).detach().cpu()
        }
    _, traces = model(tokens, return_traces=True)
    if state_kind not in traces:
        raise ValueError(f"State kind {state_kind!r} unavailable; options={sorted(traces)}")
    states = traces[state_kind].detach().cpu()
    return {layer_index: states[layer_index] for layer_index in layers}


def label_position_mask(labels: pd.DataFrame, *, mode: str, max_prefix_len: int | None) -> pd.Series:
    mode = mode or "prefix"
    if mode == "all":
        keep = pd.Series(True, index=labels.index)
    elif mode == "prefix":
        if max_prefix_len is None:
            keep = pd.Series(True, index=labels.index)
        else:
            keep = labels["dyck_seen"] <= max_prefix_len
    elif mode == "dyck":
        keep = labels["is_dyck_position"].astype(bool)
    elif mode == "final":
        final_positions = labels.groupby("example_id")["position"].transform("max")
        keep = labels["position"] == final_positions
    else:
        raise ValueError("position_mode must be one of: all, prefix, dyck, final")
    return keep
