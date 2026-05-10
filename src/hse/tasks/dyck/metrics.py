from __future__ import annotations

import torch


@torch.no_grad()
def dyck_token_accuracy(logits: torch.Tensor, tokens: torch.Tensor, dyck_mask: torch.Tensor) -> float:
    """Next-token accuracy restricted to positions whose target is a Dyck token."""
    pred = logits[:, :-1].argmax(dim=-1)
    target = tokens[:, 1:]
    target_mask = dyck_mask[:, 1:]
    if not bool(target_mask.any()):
        return float("nan")
    return float((pred[target_mask] == target[target_mask]).float().mean().item())


@torch.no_grad()
def next_token_accuracy(logits: torch.Tensor, tokens: torch.Tensor) -> float:
    pred = logits[:, :-1].argmax(dim=-1)
    target = tokens[:, 1:]
    return float((pred == target).float().mean().item())
