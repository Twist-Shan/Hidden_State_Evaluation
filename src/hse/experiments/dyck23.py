from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from hse.analysis.compression import run_compression_probes
from hse.analysis.geometry.directions import dyck_direction_geometry
from hse.analysis.probes.linear import fit_logistic_probe, fit_ridge_probe
from hse.models import build_model
from hse.tasks.dyck23 import Dyck23Batch, Dyck23Config, Dyck23Sampler, build_prefix_labels
from hse.utils.io import save_json
from hse.utils.labels_io import save_labels


DYCK23_LENGTH_BINS = {
    "len_0_40": {"min_length": 0, "max_length": 40},
    "len_40_80": {"min_length": 42, "max_length": 80},
    "len_80_120": {"min_length": 82, "max_length": 120},
}


DYCK23_MODEL_SPECS = {
    "rnn": {
        "layers": 5,
        "emb_dim": 80,
        "hidden_dim": 80,
        "dropout": 0.1,
        "learned_initial_state": True,
        "fused": True,
        "state_kind": "h",
    },
    "lstm": {
        "layers": 5,
        "emb_dim": 40,
        "hidden_dim": 40,
        "dropout": 0.1,
        "learned_initial_state": True,
        "fused": True,
        "state_kind": "h",
    },
    "transformer": {
        "layers": 5,
        "emb_dim": 32,
        "hidden_dim": 32,
        "n_heads": 8,
        "ffn_dim": 128,
        "dropout": 0.1,
        "pos_encoding": "sinusoidal",
        "embed_scale": True,
        "final_layer_norm": True,
        "state_kind": "h",
    },
    "mamba": {
        "layers": 5,
        "emb_dim": 36,
        "hidden_dim": 36,
        "state_dim": 16,
        "expansion_factor": 2,
        "require_official_mamba": True,
        "state_kind": "h",
    },
}


def run_dyck23_suite(
    *,
    seed: int = 0,
    models: list[str] | tuple[str, ...] | None = None,
    length_bins: dict[str, dict] | None = None,
    train_examples: int = 10_000,
    test_examples: int = 2_000,
    probe_examples: int = 1_024,
    steps: int | None = 15_000,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 3e-4,
    grad_clip: float = 5.0,
    eval_every_epochs: int = 1,
    extract_batch_size: int = 256,
    max_probe_rows: int = 20_000,
    device: str | None = None,
    results_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train 4 x 3 Dyck-(2,3) next-token models and run linear probes."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_names = list(models or DYCK23_MODEL_SPECS)
    unknown = sorted(set(model_names) - set(DYCK23_MODEL_SPECS))
    if unknown:
        raise ValueError(f"Unknown Dyck23 model names: {unknown}")

    bins = length_bins or DYCK23_LENGTH_BINS
    results_root = Path(results_root or Path("results") / "dyck23_cfg_next_token")
    results_root.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict] = []
    probe_rows: list[dict] = []
    in_progress_runs_path = results_root / "runs.in_progress.csv"
    in_progress_probe_path = results_root / "probe_summary.in_progress.csv"
    for bin_idx, (bin_name, bin_kwargs) in enumerate(bins.items()):
        cfg = Dyck23Config(**bin_kwargs, device="cpu")
        train_batch = Dyck23Sampler(cfg, seed=seed + 10_000 * bin_idx).sample(train_examples)
        test_batch = Dyck23Sampler(cfg, seed=seed + 10_000 * bin_idx + 1).sample(test_examples)
        probe_batch = Dyck23Sampler(cfg, seed=seed + 10_000 * bin_idx + 2).sample(probe_examples)

        bin_dir = results_root / bin_name
        bin_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            {
                "bin": bin_name,
                "task": asdict(cfg),
                "train_examples": train_examples,
                "test_examples": test_examples,
                "probe_examples": probe_examples,
                "valid_lengths": list(cfg.valid_lengths),
            },
            bin_dir / "data_summary.json",
        )

        for model_idx, model_name in enumerate(model_names):
            spec = dict(DYCK23_MODEL_SPECS[model_name])
            model_kwargs = {k: v for k, v in spec.items() if k != "state_kind"}
            torch.manual_seed(seed + 1_000 * bin_idx + model_idx)
            model = build_model(model_name=model_name, vocab_size=cfg.vocab_size, **model_kwargs)
            run_dir = bin_dir / f"{model_name}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            total_params = count_trainable_parameters(model)
            backbone_params = count_trainable_parameters(model, exclude_output=True)
            save_json(
                {
                    "setting_name": "dyck23_cfg_next_token",
                    "length_bin": bin_name,
                    "task": asdict(cfg),
                    "model_name": model_name,
                    "model": spec,
                    "seed": seed,
                    "training_steps": steps,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "grad_clip": grad_clip,
                    "device": device,
                    "train_examples": train_examples,
                    "test_examples": test_examples,
                    "probe_examples": probe_examples,
                    "trainable_parameters": total_params,
                    "backbone_parameters_excluding_lm_head": backbone_params,
                },
                run_dir / "config.json",
            )

            print(f"\n[{bin_name}] training {model_name} ({backbone_params:,} backbone params)")
            started = time.time()
            train_log, eval_metrics = train_dyck23_model(
                model=model,
                train_batch=train_batch,
                test_batch=test_batch,
                vocab_size=cfg.vocab_size,
                steps=steps,
                epochs=epochs,
                batch_size=batch_size,
                lr=learning_rate,
                grad_clip=grad_clip,
                eval_every_epochs=eval_every_epochs,
                device=device,
                run_dir=run_dir,
            )

            hidden, labels = extract_dyck23_hidden_states(
                model=model,
                batch=probe_batch,
                config=cfg,
                state_kind=spec["state_kind"],
                batch_size=extract_batch_size,
                device=device,
                run_dir=run_dir,
            )
            probe_summary, compression_summary = run_dyck23_probes(
                hidden,
                labels,
                seed=seed,
                max_rows=max_probe_rows,
                run_dir=run_dir,
            )

            elapsed = time.time() - started
            metrics_payload = {
                "train": train_log,
                "eval": eval_metrics,
                "probe": probe_summary,
                "compression": compression_summary,
                "elapsed_seconds": elapsed,
            }
            save_json(metrics_payload, run_dir / "metrics.json")

            run_row = {
                "length_bin": bin_name,
                "model": model_name,
                "seed": seed,
                "run_dir": str(run_dir.resolve()),
                "loss": eval_metrics["loss"],
                "accuracy": eval_metrics["accuracy"],
                "dyck_accuracy": eval_metrics["dyck_accuracy"],
                "hidden_rows": int(hidden.shape[0]),
                "hidden_dim": int(hidden.shape[1]),
                "params": total_params,
                "backbone_params": backbone_params,
                "elapsed_seconds": elapsed,
            }
            probe_row = {"length_bin": bin_name, "model": model_name, "seed": seed, **probe_summary, **compression_summary}
            run_rows.append(run_row)
            probe_rows.append(probe_row)
            pd.DataFrame(run_rows).to_csv(in_progress_runs_path, index=False)
            pd.DataFrame(probe_rows).to_csv(in_progress_probe_path, index=False)
            print(f"[{bin_name}] finished {model_name}: acc={eval_metrics['accuracy']:.3f}, loss={eval_metrics['loss']:.3f}")

    runs_df = pd.DataFrame(run_rows)
    probe_df = pd.DataFrame(probe_rows)
    runs_df.to_csv(results_root / "runs.csv", index=False)
    probe_df.to_csv(results_root / "probe_summary.csv", index=False)
    return runs_df, probe_df


def train_dyck23_model(
    *,
    model: nn.Module,
    train_batch: Dyck23Batch,
    test_batch: Dyck23Batch,
    vocab_size: int,
    steps: int | None,
    epochs: int,
    batch_size: int,
    lr: float,
    grad_clip: float,
    eval_every_epochs: int,
    device: str,
    run_dir: Path,
) -> tuple[list[dict], dict]:
    device_t = torch.device(device)
    model.to(device_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(0)
    train_size = int(train_batch.tokens.shape[0])
    log: list[dict] = []
    best_loss = float("inf")
    best_state = None
    best_step = 0
    best_epoch = 0.0
    global_step = 0

    if steps is not None:
        eval_every_steps = max(1, min(1000, int(steps)))
        order = torch.randperm(train_size, generator=generator)
        cursor = 0
        train_loss_sum = 0.0
        train_token_count = 0
        for step in range(1, int(steps) + 1):
            model.train()
            if cursor + batch_size > train_size:
                order = torch.randperm(train_size, generator=generator)
                cursor = 0
            idx = order[cursor : cursor + batch_size]
            cursor += batch_size
            loss, token_count = _train_one_batch(
                model=model,
                train_batch=train_batch,
                idx=idx,
                vocab_size=vocab_size,
                optimizer=optimizer,
                criterion=criterion,
                grad_clip=grad_clip,
                device=device_t,
            )
            train_loss_sum += loss * token_count
            train_token_count += token_count
            global_step = step

            should_eval = step == 1 or step % eval_every_steps == 0 or step == int(steps)
            if should_eval:
                eval_metrics = evaluate_dyck23_model(
                    model=model,
                    batch=test_batch,
                    vocab_size=vocab_size,
                    batch_size=batch_size,
                    device=device,
                )
                train_loss = train_loss_sum / max(train_token_count, 1)
                row = {"epoch": float(step * batch_size / train_size), "step": step, "train_loss": train_loss, **eval_metrics}
                log.append(row)
                print(
                    f"  step {step:05d}/{int(steps)}: train_loss={train_loss:.4f} "
                    f"test_loss={eval_metrics['loss']:.4f} acc={eval_metrics['accuracy']:.3f}"
                )
                train_loss_sum = 0.0
                train_token_count = 0
                if eval_metrics["loss"] < best_loss:
                    best_loss = eval_metrics["loss"]
                    best_step = step
                    best_epoch = row["epoch"]
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    else:
        for epoch in range(1, epochs + 1):
            model.train()
            permutation = torch.randperm(train_size, generator=generator)
            train_loss_sum = 0.0
            train_token_count = 0
            for start in range(0, train_size, batch_size):
                idx = permutation[start : start + batch_size]
                loss, token_count = _train_one_batch(
                    model=model,
                    train_batch=train_batch,
                    idx=idx,
                    vocab_size=vocab_size,
                    optimizer=optimizer,
                    criterion=criterion,
                    grad_clip=grad_clip,
                    device=device_t,
                )
                train_loss_sum += loss * token_count
                train_token_count += token_count
                global_step += 1

            should_eval = epoch == 1 or epoch % eval_every_epochs == 0 or epoch == epochs
            if should_eval:
                eval_metrics = evaluate_dyck23_model(
                    model=model,
                    batch=test_batch,
                    vocab_size=vocab_size,
                    batch_size=batch_size,
                    device=device,
                )
                train_loss = train_loss_sum / max(train_token_count, 1)
                row = {"epoch": epoch, "step": global_step, "train_loss": train_loss, **eval_metrics}
                log.append(row)
                print(
                    f"  epoch {epoch:03d}/{epochs}: train_loss={train_loss:.4f} "
                    f"test_loss={eval_metrics['loss']:.4f} acc={eval_metrics['accuracy']:.3f}"
                )
                if eval_metrics["loss"] < best_loss:
                    best_loss = eval_metrics["loss"]
                    best_step = global_step
                    best_epoch = float(epoch)
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": epochs, "step": global_step}, ckpt_dir / "model_final.pt")
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save({"model": best_state, "step": best_step, "epoch": best_epoch, "loss": best_loss}, ckpt_dir / "model_best.pt")
    eval_metrics = evaluate_dyck23_model(
        model=model,
        batch=test_batch,
        vocab_size=vocab_size,
        batch_size=batch_size,
        device=device,
    )
    return log, eval_metrics


def _train_one_batch(
    *,
    model: nn.Module,
    train_batch: Dyck23Batch,
    idx: torch.Tensor,
    vocab_size: int,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    grad_clip: float,
    device: torch.device,
) -> tuple[float, int]:
    tokens = train_batch.tokens[idx].to(device)
    target_mask = train_batch.target_mask[idx].to(device)
    logits = model(tokens)
    loss = _masked_ce_mean(criterion, logits, tokens, target_mask, vocab_size)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    token_count = int(target_mask[:, 1:].sum().item())
    return float(loss.item()), token_count


@torch.no_grad()
def evaluate_dyck23_model(
    *,
    model: nn.Module,
    batch: Dyck23Batch,
    vocab_size: int,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    device_t = torch.device(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    target_count = 0
    correct = 0
    dyck_correct = 0
    dyck_count = 0

    for start in range(0, int(batch.tokens.shape[0]), batch_size):
        stop = min(start + batch_size, int(batch.tokens.shape[0]))
        tokens = batch.tokens[start:stop].to(device_t)
        target_mask = batch.target_mask[start:stop].to(device_t)
        dyck_mask = batch.dyck_mask[start:stop].to(device_t)
        logits = model(tokens)
        pred = logits[:, :-1].argmax(dim=-1)
        target = tokens[:, 1:]
        valid = target_mask[:, 1:].bool()
        flat_logits = logits[:, :-1].reshape(-1, vocab_size)
        flat_target = target.reshape(-1)
        flat_valid = valid.reshape(-1)
        loss_sum += float(criterion(flat_logits[flat_valid], flat_target[flat_valid]).item())
        target_count += int(flat_valid.sum().item())
        correct += int((pred[valid] == target[valid]).sum().item())

        dyck_valid = dyck_mask[:, 1:].bool()
        dyck_count += int(dyck_valid.sum().item())
        if bool(dyck_valid.any()):
            dyck_correct += int((pred[dyck_valid] == target[dyck_valid]).sum().item())

    return {
        "loss": loss_sum / max(target_count, 1),
        "accuracy": correct / max(target_count, 1),
        "dyck_accuracy": dyck_correct / dyck_count if dyck_count else float("nan"),
        "tokens": float(target_count),
    }


@torch.no_grad()
def extract_dyck23_hidden_states(
    *,
    model: nn.Module,
    batch: Dyck23Batch,
    config: Dyck23Config,
    state_kind: str,
    batch_size: int,
    device: str,
    run_dir: Path | None = None,
) -> tuple[torch.Tensor, pd.DataFrame]:
    device_t = torch.device(device)
    model.to(device_t)
    model.eval()
    all_hidden = []
    all_labels = []

    for start in range(0, int(batch.tokens.shape[0]), batch_size):
        stop = min(start + batch_size, int(batch.tokens.shape[0]))
        local = _slice_batch(batch, start, stop)
        tokens = local.tokens.to(device_t)
        states = model.extract_states(tokens, layer_index=-1, state_kind=state_kind).detach().cpu()
        labels = build_prefix_labels(local, config=config)
        labels["example_id"] += start
        local_ids = torch.tensor(labels["example_id"].to_numpy() - start, dtype=torch.long)
        positions = torch.tensor(labels["position"].to_numpy(), dtype=torch.long)
        all_hidden.append(states[local_ids, positions].contiguous())
        all_labels.append(labels)

    hidden = torch.cat(all_hidden, dim=0)
    labels_df = pd.concat(all_labels, ignore_index=True)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(hidden, run_dir / "hidden_states.pt")
        save_labels(labels_df, run_dir / "labels")
    return hidden, labels_df


def run_dyck23_probes(
    hidden: torch.Tensor,
    labels: pd.DataFrame,
    *,
    seed: int,
    max_rows: int,
    run_dir: Path,
) -> tuple[dict, dict]:
    X = hidden.numpy()
    probe_labels = labels
    if max_rows and len(probe_labels) > max_rows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(probe_labels), size=max_rows, replace=False))
        X = X[idx]
        probe_labels = probe_labels.iloc[idx].reset_index(drop=True)

    results = {}
    regression_targets = [
        "left",
        "right",
        "height",
        "left_round",
        "right_round",
        "height_round",
        "left_square",
        "right_square",
        "height_square",
    ]
    for target in regression_targets:
        if target in probe_labels:
            results[target] = fit_ridge_probe(X, probe_labels[target].to_numpy(), seed=seed)

    classification_targets = ["height_class", "top_type_class", "depth_top_class", "legal_next_class"]
    for target in classification_targets:
        if target in probe_labels and probe_labels[target].nunique() > 1:
            results[target] = fit_logistic_probe(X, probe_labels[target].to_numpy(), seed=seed)

    summary = {}
    for target, result in results.items():
        if "r2" in result:
            summary[f"{target}_r2"] = result["r2"]
            summary[f"{target}_mae"] = result["mae"]
        elif "accuracy" in result:
            summary[f"{target}_accuracy"] = result["accuracy"]

    if {"left", "right", "height"}.issubset(results):
        summary.update(dyck_direction_geometry(results))

    compression_rows, compression_summary = run_compression_probes(X, probe_labels, seed=seed)
    probes_dir = run_dir / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, probes_dir / "summary.json")
    compression_rows.to_csv(probes_dir / "compression_probe_rows.csv", index=False)
    return summary, compression_summary


def count_trainable_parameters(model: nn.Module, *, exclude_output: bool = False) -> int:
    total = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if exclude_output and name.startswith("output."):
            continue
        total += param.numel()
    return total


def _masked_ce_mean(
    criterion: nn.CrossEntropyLoss,
    logits: torch.Tensor,
    tokens: torch.Tensor,
    target_mask: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    flat_logits = logits[:, :-1].reshape(-1, vocab_size)
    flat_target = tokens[:, 1:].reshape(-1)
    flat_mask = target_mask[:, 1:].reshape(-1).bool()
    return criterion(flat_logits[flat_mask], flat_target[flat_mask])


def _slice_batch(batch: Dyck23Batch, start: int, stop: int) -> Dyck23Batch:
    return Dyck23Batch(
        tokens=batch.tokens[start:stop],
        dyck_mask=batch.dyck_mask[start:stop],
        target_mask=batch.target_mask[start:stop],
        lengths=batch.lengths[start:stop],
        bracket_lengths=batch.bracket_lengths[start:stop],
        dyck_steps=batch.dyck_steps[start:stop],
        bracket_type_ids=batch.bracket_type_ids[start:stop],
    )
