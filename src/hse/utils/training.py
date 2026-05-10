from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from hse.tasks.dyck.metrics import dyck_token_accuracy, next_token_accuracy
from hse.tasks.dyck.sampler import DyckSampler


def train_causal_lm(
    *,
    model: nn.Module,
    sampler: DyckSampler,
    steps: int,
    batch_size: int,
    lr: float,
    run_dir: str | Path | None = None,
    eval_every: int = 200,
    grad_clip: float = 1.0,
    device: str | torch.device = "cpu",
) -> dict[str, list[float]]:
    device = torch.device(device)
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    log: dict[str, list[float]] = {"step": [], "loss": [], "eval_loss": [], "eval_acc": [], "eval_dyck_acc": []}

    for step in range(1, int(steps) + 1):
        batch = sampler.sample(batch_size)
        tokens = batch.tokens.to(device)
        logits = model(tokens)
        loss = criterion(logits[:, :-1].reshape(-1, sampler.vocab_size), tokens[:, 1:].reshape(-1))
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

    if run_dir is not None:
        run_dir = Path(run_dir)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "step": steps}, run_dir / "checkpoints" / "model_final.pt")
    return log


@torch.no_grad()
def evaluate_causal_lm(
    *,
    model: nn.Module,
    sampler: DyckSampler,
    batch_size: int = 512,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    device = torch.device(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    batch = sampler.sample(batch_size)
    tokens = batch.tokens.to(device)
    dyck_mask = batch.dyck_mask.to(device)
    logits = model(tokens)
    loss = criterion(logits[:, :-1].reshape(-1, sampler.vocab_size), tokens[:, 1:].reshape(-1))
    return {
        "loss": float(loss.item()),
        "accuracy": next_token_accuracy(logits, tokens),
        "dyck_accuracy": dyck_token_accuracy(logits, tokens, dyck_mask),
    }
