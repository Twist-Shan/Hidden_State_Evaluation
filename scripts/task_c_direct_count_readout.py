from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hse_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class TaskCConfig:
    seed: int = 0
    seq_len: int = 128
    max_count: int = 10
    noise_vocab: int = 32
    steps: int = 1000
    batch_size: int = 128
    eval_examples: int = 2048
    probe_examples: int = 4096
    lr: float = 3e-4
    weight_decay: float = 0.01
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.0
    ridge_alpha: float = 1.0
    device: str = "auto"

    @property
    def needle_token(self) -> int:
        return self.noise_vocab

    @property
    def query_token(self) -> int:
        return self.noise_vocab + 1

    @property
    def num_start(self) -> int:
        return self.noise_vocab + 2

    @property
    def vocab_size(self) -> int:
        return self.num_start + self.max_count + 1

    @property
    def prompt_len(self) -> int:
        return self.seq_len + 1


class TinyCausalTransformer(nn.Module):
    def __init__(self, config: TaskCConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.prompt_len, config.d_model)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.n_heads,
                    dim_feedforward=config.d_ff,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids: torch.Tensor, return_hiddens: bool = False):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
            diagonal=1,
        )
        hiddens = []
        for layer in self.layers:
            hidden = layer(hidden, src_mask=mask)
            if return_hiddens:
                hiddens.append(hidden)
        logits = self.output(self.ln_f(hidden))
        if return_hiddens:
            return logits, hiddens
        return logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "needle_count_task_c_direct_count"))
    parser.add_argument("--figure-dir", default=str(ROOT / "figures" / "needle_count_task_c_direct_count"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-count", type=int, default=10)
    parser.add_argument("--noise-vocab", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-examples", type=int, default=2048)
    parser.add_argument("--probe-examples", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint-steps", default="0,50,100,200,500,1000")
    args = parser.parse_args()

    config = TaskCConfig(
        seed=args.seed,
        seq_len=args.seq_len,
        max_count=args.max_count,
        noise_vocab=args.noise_vocab,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_examples=args.eval_examples,
        probe_examples=args.probe_examples,
        lr=args.lr,
        device=args.device,
    )
    checkpoint_steps = sorted({int(step) for step in args.checkpoint_steps.split(",") if step.strip()})
    checkpoint_steps = [step for step in checkpoint_steps if 0 <= step <= config.steps]
    if config.steps not in checkpoint_steps:
        checkpoint_steps.append(config.steps)

    out_dir = Path(args.out_dir)
    figure_dir = Path(args.figure_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    device = resolve_device(config.device)
    model = TinyCausalTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    train_rng = np.random.default_rng(config.seed + 100)
    eval_rng = np.random.default_rng(config.seed + 200)
    probe_rng = np.random.default_rng(config.seed + 300)

    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "task": "NeedleCount-synthetic direct final count",
                "format": "<seq> <QUERY_COUNT> <NUM_k>",
                "description": (
                    "A balanced direct-count smoke test for Experiment C. Each prefix contains k NEEDLE tokens "
                    "at random positions among distractors, with k sampled uniformly from 0..max_count. The model "
                    "predicts one single-token answer NUM_k at the query position."
                ),
                **asdict(config),
                "needle_token": config.needle_token,
                "query_token": config.query_token,
                "num_start": config.num_start,
                "vocab_size": config.vocab_size,
                "checkpoint_steps": checkpoint_steps,
                "device_resolved": str(device),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    checkpoint_metrics = []
    probe_rows = []
    train_losses = []
    last_loss = math.nan

    def evaluate_checkpoint(step: int) -> None:
        nonlocal last_loss
        metrics = evaluate_model(model, config, eval_rng, device, config.eval_examples)
        metrics.update({"step": step, "train_loss": last_loss})
        checkpoint_metrics.append(metrics)
        checkpoint_probe_rows, checkpoint_probe_state = probe_checkpoint(
            model=model,
            config=config,
            rng=probe_rng,
            device=device,
            n_examples=config.probe_examples,
            checkpoint_step=step,
        )
        probe_rows.extend(checkpoint_probe_rows)
        if step == config.steps:
            steering = run_final_query_steering(
                model=model,
                config=config,
                rng=np.random.default_rng(config.seed + 400),
                device=device,
                probe_state=checkpoint_probe_state,
                n_examples=config.eval_examples,
            )
            steering.to_csv(out_dir / "steering.csv", index=False)

    if 0 in checkpoint_steps:
        evaluate_checkpoint(0)

    for step in range(1, config.steps + 1):
        input_ids, answer_ids, _counts = sample_batch(config, config.batch_size, train_rng, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = F.cross_entropy(logits[:, -1, :], answer_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        train_losses.append({"step": step, "train_loss": last_loss})
        if step in checkpoint_steps:
            evaluate_checkpoint(step)
            print(f"step {step}: loss={last_loss:.4f}")

    checkpoint_df = pd.DataFrame(checkpoint_metrics)
    probe_df = pd.DataFrame(probe_rows)
    train_df = pd.DataFrame(train_losses)
    checkpoint_df.to_csv(out_dir / "checkpoint_metrics.csv", index=False)
    probe_df.to_csv(out_dir / "probe_readout.csv", index=False)
    train_df.to_csv(out_dir / "train_loss.csv", index=False)

    save_summary(checkpoint_df, probe_df, out_dir)
    plot_all(out_dir, figure_dir)
    torch.save({"model": model.state_dict(), "config": asdict(config)}, out_dir / "model_final.pt")
    print(f"saved Task C results to {out_dir}")
    print(f"saved Task C figures to {figure_dir}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def sample_batch(config: TaskCConfig, batch_size: int, rng: np.random.Generator, device: torch.device):
    tokens = rng.integers(0, config.noise_vocab, size=(batch_size, config.seq_len), dtype=np.int64)
    counts = rng.integers(0, config.max_count + 1, size=batch_size, dtype=np.int64)
    for row, count in enumerate(counts):
        if count > 0:
            positions = rng.choice(config.seq_len, size=int(count), replace=False)
            tokens[row, positions] = config.needle_token
    query = np.full((batch_size, 1), config.query_token, dtype=np.int64)
    input_ids = np.concatenate([tokens, query], axis=1)
    answers = config.num_start + counts
    return (
        torch.as_tensor(input_ids, dtype=torch.long, device=device),
        torch.as_tensor(answers, dtype=torch.long, device=device),
        torch.as_tensor(counts, dtype=torch.long, device=device),
    )


@torch.no_grad()
def evaluate_model(
    model: TinyCausalTransformer,
    config: TaskCConfig,
    rng: np.random.Generator,
    device: torch.device,
    n_examples: int,
) -> dict[str, float]:
    model.eval()
    rows = []
    batch_size = min(512, n_examples)
    for start in range(0, n_examples, batch_size):
        current = min(batch_size, n_examples - start)
        input_ids, answer_ids, counts = sample_batch(config, current, rng, device)
        logits = model(input_ids)[:, -1, :]
        loss = F.cross_entropy(logits, answer_ids)
        full_pred = logits.argmax(dim=-1)
        num_logits = logits[:, config.num_start : config.num_start + config.max_count + 1]
        pred_count = num_logits.argmax(dim=-1)
        true_count = counts
        true_num_logit = num_logits.gather(1, true_count[:, None]).squeeze(1)
        wrong_num_logits = num_logits.clone()
        wrong_num_logits.scatter_(1, true_count[:, None], float("-inf"))
        margin = true_num_logit - wrong_num_logits.max(dim=1).values
        non_num_mask = torch.ones(config.vocab_size, dtype=torch.bool, device=device)
        non_num_mask[config.num_start : config.num_start + config.max_count + 1] = False
        probs = logits.softmax(dim=-1)
        non_num_mass = probs[:, non_num_mask].sum(dim=-1)
        rows.append(
            {
                "loss": float(loss.detach().cpu()),
                "answer_acc": float((full_pred == answer_ids).float().mean().cpu()),
                "num_restricted_acc": float((pred_count == true_count).float().mean().cpu()),
                "off_by_one_rate": float((torch.abs(pred_count - true_count) == 1).float().mean().cpu()),
                "mae": float(torch.abs(pred_count - true_count).float().mean().cpu()),
                "bias_pred_minus_true": float((pred_count - true_count).float().mean().cpu()),
                "mean_answer_margin": float(margin.mean().cpu()),
                "non_num_top1_rate": float(((full_pred < config.num_start) | (full_pred > config.num_start + config.max_count)).float().mean().cpu()),
                "non_num_mass": float(non_num_mass.mean().cpu()),
            }
        )
    return weighted_average(rows)


@torch.no_grad()
def collect_query_hiddens(
    model: TinyCausalTransformer,
    config: TaskCConfig,
    rng: np.random.Generator,
    device: torch.device,
    n_examples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    layer_chunks: list[list[np.ndarray]] = [[] for _ in range(config.n_layers)]
    count_chunks = []
    logit_chunks = []
    batch_size = min(512, n_examples)
    for start in range(0, n_examples, batch_size):
        current = min(batch_size, n_examples - start)
        input_ids, _answer_ids, counts = sample_batch(config, current, rng, device)
        logits, hiddens = model(input_ids, return_hiddens=True)
        for layer_idx, hidden in enumerate(hiddens):
            lens_hidden = model.ln_f(hidden[:, -1, :])
            layer_chunks[layer_idx].append(lens_hidden.detach().cpu().numpy())
        logit_chunks.append(logits[:, -1, :].detach().cpu().numpy())
        count_chunks.append(counts.detach().cpu().numpy())
    hidden_by_layer = np.stack([np.concatenate(chunks, axis=0) for chunks in layer_chunks], axis=1)
    counts = np.concatenate(count_chunks, axis=0).astype(np.int64)
    logits = np.concatenate(logit_chunks, axis=0)
    return hidden_by_layer, counts, logits


def probe_checkpoint(
    *,
    model: TinyCausalTransformer,
    config: TaskCConfig,
    rng: np.random.Generator,
    device: torch.device,
    n_examples: int,
    checkpoint_step: int,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    hidden_by_layer, counts, _logits = collect_query_hiddens(model, config, rng, device, n_examples)
    split = n_examples // 2
    answer_weight = model.output.weight.detach().cpu().numpy()
    answer_bias = model.output.bias.detach().cpu().numpy()
    num_weight = answer_weight[config.num_start : config.num_start + config.max_count + 1]
    adjacent_answer_delta = np.diff(num_weight, axis=0).mean(axis=0)

    rows = []
    states = []
    for layer_idx in range(config.n_layers):
        X_train = hidden_by_layer[:split, layer_idx, :]
        X_test = hidden_by_layer[split:, layer_idx, :]
        y_train = counts[:split].astype(np.float64)
        y_test = counts[split:].astype(np.float64)
        scalar = fit_ridge_scalar(X_train, y_train, X_test, y_test, alpha=config.ridge_alpha)
        multiclass = fit_ridge_multiclass(
            X_train,
            counts[:split],
            X_test,
            counts[split:],
            n_classes=config.max_count + 1,
            alpha=config.ridge_alpha,
        )
        X_test_logits = X_test @ answer_weight.T + answer_bias
        num_logits = X_test_logits[:, config.num_start : config.num_start + config.max_count + 1]
        logit_lens_pred = num_logits.argmax(axis=1)
        logit_lens_acc = float((logit_lens_pred == counts[split:]).mean())
        margins = true_margin(num_logits, counts[split:])
        cosine = cosine_similarity(scalar["direction"], adjacent_answer_delta)
        row = {
            "step": checkpoint_step,
            "layer": layer_idx,
            "count_r2": scalar["r2"],
            "count_mae": scalar["mae"],
            "ridge_round_acc": scalar["round_acc"],
            "linear_answer_acc": multiclass["accuracy"],
            "logit_lens_acc": logit_lens_acc,
            "logit_lens_margin": float(np.mean(margins)),
            "unembedding_adjacent_cosine": cosine,
            "axis_std": scalar["axis_std"],
        }
        rows.append(row)
        states.append({"layer": layer_idx, **scalar, "logit_lens_acc": logit_lens_acc})

    steering_state = max(states, key=lambda row: (row["logit_lens_acc"], row["r2"], row["layer"]))
    return rows, {
        "steering_layer": steering_state["layer"],
        "steering_direction": steering_state["direction"],
        "steering_layer_selection": "best_logit_lens_acc",
    }


def fit_ridge_scalar(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    alpha: float,
) -> dict[str, object]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    Xtr = (X_train - mean) / std
    Xte = (X_test - mean) / std
    Xtr_aug = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1))], axis=1)
    Xte_aug = np.concatenate([Xte, np.ones((Xte.shape[0], 1))], axis=1)
    reg = np.eye(Xtr_aug.shape[1]) * alpha
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(Xtr_aug.T @ Xtr_aug + reg, Xtr_aug.T @ y_train)
    pred = Xte_aug @ coef
    residual = y_test - pred
    denom = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1.0 - float(np.sum(residual**2) / max(denom, 1e-12))
    clipped = np.clip(np.rint(pred), y_test.min(), y_test.max()).astype(np.int64)
    direction = coef[:-1] / std.reshape(-1)
    direction_norm = np.linalg.norm(direction)
    if direction_norm > 0:
        unit = direction / direction_norm
        axis_std = float(np.std(X_test @ unit))
    else:
        axis_std = 0.0
    return {
        "r2": r2,
        "mae": float(np.mean(np.abs(pred - y_test))),
        "round_acc": float((clipped == y_test.astype(np.int64)).mean()),
        "direction": direction.astype(np.float64),
        "axis_std": axis_std,
    }


def fit_ridge_multiclass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_classes: int,
    alpha: float,
) -> dict[str, float]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    Xtr = (X_train - mean) / std
    Xte = (X_test - mean) / std
    Xtr_aug = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1))], axis=1)
    Xte_aug = np.concatenate([Xte, np.ones((Xte.shape[0], 1))], axis=1)
    Y = np.zeros((Xtr.shape[0], n_classes), dtype=np.float64)
    Y[np.arange(y_train.shape[0]), y_train.astype(np.int64)] = 1.0
    reg = np.eye(Xtr_aug.shape[1]) * alpha
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(Xtr_aug.T @ Xtr_aug + reg, Xtr_aug.T @ Y)
    pred = (Xte_aug @ coef).argmax(axis=1)
    return {"accuracy": float((pred == y_test.astype(np.int64)).mean())}


@torch.no_grad()
def run_final_query_steering(
    *,
    model: TinyCausalTransformer,
    config: TaskCConfig,
    rng: np.random.Generator,
    device: torch.device,
    probe_state: dict[str, object],
    n_examples: int,
) -> pd.DataFrame:
    hidden_by_layer, counts, _logits = collect_query_hiddens(model, config, rng, device, n_examples)
    layer = int(probe_state["steering_layer"])
    direction = np.asarray(probe_state["steering_direction"], dtype=np.float64)
    direction_norm = np.linalg.norm(direction)
    if direction_norm <= 0:
        direction = np.zeros_like(direction)
    else:
        direction = direction / direction_norm
    hidden = hidden_by_layer[:, layer, :]
    axis_std = float(np.std(hidden @ direction))
    if axis_std <= 1e-8:
        axis_std = 1.0
    weight = model.output.weight.detach().cpu().numpy()
    bias = model.output.bias.detach().cpu().numpy()
    non_num_mask = np.ones(config.vocab_size, dtype=bool)
    non_num_mask[config.num_start : config.num_start + config.max_count + 1] = False

    rows = []
    baseline_logits = hidden @ weight.T + bias
    baseline_num_logits = baseline_logits[:, config.num_start : config.num_start + config.max_count + 1]
    baseline_pred = baseline_num_logits.argmax(axis=1)
    for beta in [-5, -3, -1, -0.5, 0.0, 0.5, 1, 3, 5]:
        steered = hidden + beta * axis_std * direction[None, :]
        logits = steered @ weight.T + bias
        num_logits = logits[:, config.num_start : config.num_start + config.max_count + 1]
        pred = num_logits.argmax(axis=1)
        margins = true_margin(num_logits, counts)
        probs = softmax_np(logits)
        rows.append(
            {
                "layer": layer,
                "beta_axis_std": beta,
                "axis_std": axis_std,
                "answer_acc": float((pred == counts).mean()),
                "mean_pred_count": float(pred.mean()),
                "mean_pred_count_delta_vs_baseline": float((pred - baseline_pred).mean()),
                "frac_pred_plus_one_vs_baseline": float((pred == baseline_pred + 1).mean()),
                "frac_pred_minus_one_vs_baseline": float((pred == baseline_pred - 1).mean()),
                "mean_answer_margin": float(margins.mean()),
                "non_num_mass": float(probs[:, non_num_mask].sum(axis=1).mean()),
            }
        )
    return pd.DataFrame(rows)


def true_margin(num_logits: np.ndarray, counts: np.ndarray) -> np.ndarray:
    true = num_logits[np.arange(num_logits.shape[0]), counts.astype(np.int64)]
    wrong = num_logits.copy()
    wrong[np.arange(wrong.shape[0]), counts.astype(np.int64)] = -np.inf
    return true - wrong.max(axis=1)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(left, right) / denom)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def weighted_average(rows: list[dict[str, float]]) -> dict[str, float]:
    frame = pd.DataFrame(rows)
    return {column: float(frame[column].mean()) for column in frame.columns}


def save_summary(checkpoint_df: pd.DataFrame, probe_df: pd.DataFrame, out_dir: Path) -> None:
    final_step = int(checkpoint_df["step"].max())
    final_metrics = checkpoint_df.loc[checkpoint_df["step"] == final_step].iloc[0].to_dict()
    final_probe = probe_df.loc[probe_df["step"] == final_step].copy()
    best_count_probe = final_probe.sort_values("count_r2", ascending=False).iloc[0].to_dict()
    best_logit_probe = final_probe.sort_values("logit_lens_acc", ascending=False).iloc[0].to_dict()
    final_layer_probe = final_probe.sort_values("layer").iloc[-1].to_dict()
    summary = {
        "final_step": final_step,
        "answer_acc": final_metrics["answer_acc"],
        "num_restricted_acc": final_metrics["num_restricted_acc"],
        "off_by_one_rate": final_metrics["off_by_one_rate"],
        "mae": final_metrics["mae"],
        "mean_answer_margin": final_metrics["mean_answer_margin"],
        "best_layer": best_count_probe["layer"],
        "best_count_layer": best_count_probe["layer"],
        "best_count_r2": best_count_probe["count_r2"],
        "best_count_ridge_round_acc": best_count_probe["ridge_round_acc"],
        "best_count_linear_answer_acc": best_count_probe["linear_answer_acc"],
        "best_count_logit_lens_acc": best_count_probe["logit_lens_acc"],
        "best_count_unembedding_adjacent_cosine": best_count_probe["unembedding_adjacent_cosine"],
        "best_logit_lens_layer": best_logit_probe["layer"],
        "best_logit_lens_acc": best_logit_probe["logit_lens_acc"],
        "best_logit_lens_count_r2": best_logit_probe["count_r2"],
        "best_logit_lens_unembedding_adjacent_cosine": best_logit_probe["unembedding_adjacent_cosine"],
        "final_layer": final_layer_probe["layer"],
        "final_layer_count_r2": final_layer_probe["count_r2"],
        "final_layer_ridge_round_acc": final_layer_probe["ridge_round_acc"],
        "final_layer_linear_answer_acc": final_layer_probe["linear_answer_acc"],
        "final_layer_logit_lens_acc": final_layer_probe["logit_lens_acc"],
        "final_layer_unembedding_adjacent_cosine": final_layer_probe["unembedding_adjacent_cosine"],
    }
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def plot_all(out_dir: Path, figure_dir: Path) -> None:
    checkpoint = pd.read_csv(out_dir / "checkpoint_metrics.csv")
    probe = pd.read_csv(out_dir / "probe_readout.csv")
    steering_path = out_dir / "steering.csv"
    steering = pd.read_csv(steering_path) if steering_path.exists() else pd.DataFrame()

    plot_training_dynamics(checkpoint, probe, figure_dir / "task_c_training_dynamics.png")
    plot_layerwise_readout(probe, figure_dir / "task_c_layerwise_readout.png")
    if not steering.empty:
        plot_steering(steering, figure_dir / "task_c_final_query_steering.png")


def plot_training_dynamics(checkpoint: pd.DataFrame, probe: pd.DataFrame, path: Path) -> None:
    best_count = probe.sort_values(["step", "count_r2"], ascending=[True, False]).groupby("step", as_index=False).first()
    best_logit = probe.sort_values(["step", "logit_lens_acc"], ascending=[True, False]).groupby("step", as_index=False).first()
    best_linear = probe.sort_values(["step", "linear_answer_acc"], ascending=[True, False]).groupby("step", as_index=False).first()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(checkpoint["step"], checkpoint["answer_acc"], marker="o", label="model answer acc")
    axes[0].plot(checkpoint["step"], checkpoint["num_restricted_acc"], marker="o", label="NUM-restricted acc")
    axes[0].plot(best_count["step"], best_count["ridge_round_acc"], marker="o", label="best count-layer ridge round acc")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(best_count["step"], best_count["count_r2"], marker="o", label="best count R2")
    axes[1].plot(best_logit["step"], best_logit["logit_lens_acc"], marker="o", label="best logit-lens acc")
    axes[1].plot(best_linear["step"], best_linear["linear_answer_acc"], marker="o", label="best learned answer acc")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("readout score")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle("Experiment C: hidden count to answer-token readout over training")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_layerwise_readout(probe: pd.DataFrame, path: Path) -> None:
    final_step = probe["step"].max()
    final = probe.loc[probe["step"] == final_step].sort_values("layer")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(final["layer"], final["count_r2"], marker="o", label="count R2")
    axes[0].plot(final["layer"], final["ridge_round_acc"], marker="o", label="ridge round acc")
    axes[0].plot(final["layer"], final["linear_answer_acc"], marker="o", label="learned answer acc")
    axes[0].plot(final["layer"], final["logit_lens_acc"], marker="o", label="logit-lens acc")
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("score")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].bar(final["layer"].astype(str), final["unembedding_adjacent_cosine"])
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("cosine")
    axes[1].set_title("count direction vs adjacent NUM-token unembedding")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(f"Final checkpoint layer-wise readout (step {int(final_step)})")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_steering(steering: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(steering["beta_axis_std"], steering["mean_pred_count_delta_vs_baseline"], marker="o")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xlabel("beta along count direction, in axis std")
    axes[0].set_ylabel("mean predicted count shift")
    axes[0].grid(alpha=0.25)

    axes[1].plot(steering["beta_axis_std"], steering["answer_acc"], marker="o", label="answer acc")
    axes[1].plot(steering["beta_axis_std"], steering["non_num_mass"], marker="o", label="non-NUM prob mass")
    axes[1].set_xlabel("beta along count direction")
    axes[1].set_ylabel("score")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(steering["beta_axis_std"], steering["mean_answer_margin"], marker="o")
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_xlabel("beta along count direction")
    axes[2].set_ylabel("true answer margin")
    axes[2].grid(alpha=0.25)
    fig.suptitle("Final query-position steering via ridge count direction")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
