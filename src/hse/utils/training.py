from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

from hse.tasks.dyck.metrics import dyck_token_accuracy, next_token_accuracy


def train_causal_lm(
    *,
    model: nn.Module,
    sampler,
    steps: int,
    batch_size: int,
    lr: float,
    run_dir: str | Path | None = None,
    eval_every: int = 200,
    grad_clip: float = 1.0,
    checkpoint_steps: Iterable[int] | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, list[float]]:
    device = torch.device(device)
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    log: dict[str, list[float]] = {"step": [], "loss": [], "eval_loss": [], "eval_acc": [], "eval_dyck_acc": []}
    checkpoint_step_set = {int(step) for step in (checkpoint_steps or []) if int(step) >= 0}
    ckpt_dir = None
    if run_dir is not None:
        run_dir = Path(run_dir)
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if 0 in checkpoint_step_set:
            _save_checkpoint(model, ckpt_dir, step=0)

    for step in range(1, int(steps) + 1):
        batch = sampler.sample(batch_size)
        tokens = batch.tokens.to(device)
        logits = model(tokens)
        loss = _next_token_loss(
            criterion=criterion,
            logits=logits,
            tokens=tokens,
            vocab_size=sampler.vocab_size,
            target_mask=getattr(batch, "target_mask", None),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        if step == 1 or step % eval_every == 0 or step == steps:
            eval_metrics = evaluate_causal_lm(model=model, sampler=sampler, batch_size=batch_size, device=device)
            log["step"].append(float(step))
            log["loss"].append(float(loss.item()))
            log["eval_loss"].append(float(eval_metrics["loss"]))
            log["eval_acc"].append(float(eval_metrics["accuracy"]))
            log["eval_dyck_acc"].append(float(eval_metrics["dyck_accuracy"]))
            print(
                f"step={step}/{steps} "
                f"loss={loss.item():.4f} "
                f"eval_loss={eval_metrics['loss']:.4f} "
                f"acc={eval_metrics['accuracy']:.4f} "
                f"dyck_acc={eval_metrics['dyck_accuracy']:.4f}",
                flush=True,
            )
        if ckpt_dir is not None and step in checkpoint_step_set:
            _save_checkpoint(model, ckpt_dir, step=step)

    if ckpt_dir is not None:
        torch.save({"model": model.state_dict(), "step": steps}, ckpt_dir / "model_final.pt")
    return log


@torch.no_grad()
def evaluate_causal_lm(
    *,
    model: nn.Module,
    sampler,
    batch_size: int = 512,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    device = torch.device(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    batch = sampler.sample(batch_size)
    tokens = batch.tokens.to(device)
    dyck_mask = batch.dyck_mask.to(device)
    target_mask = getattr(batch, "target_mask", None)
    if target_mask is not None:
        target_mask = target_mask.to(device)
    logits = model(tokens)
    loss = _next_token_loss(
        criterion=criterion,
        logits=logits,
        tokens=tokens,
        vocab_size=sampler.vocab_size,
        target_mask=target_mask,
    )
    return {
        "loss": float(loss.item()),
        "accuracy": _next_token_accuracy(logits, tokens, target_mask),
        "dyck_accuracy": dyck_token_accuracy(logits, tokens, dyck_mask),
    }


def _next_token_loss(
    *,
    criterion: nn.CrossEntropyLoss,
    logits: torch.Tensor,
    tokens: torch.Tensor,
    vocab_size: int,
    target_mask: Any = None,
) -> torch.Tensor:
    flat_logits = logits[:, :-1].reshape(-1, vocab_size)
    flat_target = tokens[:, 1:].reshape(-1)
    if target_mask is None:
        return criterion(flat_logits, flat_target)
    flat_mask = target_mask[:, 1:].to(logits.device).reshape(-1).bool()
    return criterion(flat_logits[flat_mask], flat_target[flat_mask])


@torch.no_grad()
def _next_token_accuracy(logits: torch.Tensor, tokens: torch.Tensor, target_mask: torch.Tensor | None = None) -> float:
    if target_mask is None:
        return next_token_accuracy(logits, tokens)
    pred = logits[:, :-1].argmax(dim=-1)
    target = tokens[:, 1:]
    mask = target_mask[:, 1:].bool()
    if not bool(mask.any()):
        return float("nan")
    return float((pred[mask] == target[mask]).float().mean().item())


def _save_checkpoint(model: nn.Module, ckpt_dir: Path, *, step: int) -> None:
    torch.save({"model": model.state_dict(), "step": int(step)}, ckpt_dir / f"model_step_{int(step)}.pt")
