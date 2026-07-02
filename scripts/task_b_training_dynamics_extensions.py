from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hse_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from hse.models import build_model
from hse.tasks.registry import build_sampler, task_name_from_run_config


OUT_DIR = ROOT / "results" / "dyck_counter_task_b_extensions"
FIG_DIR = ROOT / "figures" / "dyck_counter_task_b_extensions"
BASE_EXPERIMENTS = [
    "dyck_counter_task_b_clean_short_smoke",
    "dyck_counter_task_b_noisy_short_smoke",
]
EXTENDED_EXPERIMENTS = [
    "dyck_counter_task_b_clean_short_5k",
    "dyck_counter_task_b_noisy_short_5k",
]
CHECKPOINT_RE = re.compile(r"^step_(\d+)$")


@dataclass
class RunSpec:
    experiment: str
    run_key: str
    run_dir: Path
    config: dict
    family: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-rows", type=int, default=20_000)
    parser.add_argument("--eval-examples", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    base_runs = existing_runs(BASE_EXPERIMENTS, family="base_2k")
    extended_runs = existing_runs(EXTENDED_EXPERIMENTS, family="extended_5k")
    all_runs = base_runs + extended_runs
    if not base_runs:
        raise FileNotFoundError("Task B base runs are missing. Run clean/noisy smoke first.")

    extended = experiment_1_extended_training(all_runs)
    baselines = experiment_2_baseline_controls(base_runs, max_rows=args.max_rows)
    transfer = experiment_3_probe_transfer(base_runs, max_rows=args.max_rows)
    alignment = experiment_4_output_head_alignment(base_runs, max_rows=args.max_rows)
    branch_manifold = experiment_4b_branch_manifold_split(base_runs, max_rows=args.max_rows)
    intervention = experiment_5_checkpoint_intervention(base_runs, max_rows=args.max_rows)
    length = experiment_6_length_generalization(
        base_runs,
        device=args.device,
        num_examples=args.eval_examples,
        batch_size=args.eval_batch_size,
    )
    density = experiment_7_density_curve()

    write_table(extended, "experiment_1_extended_training.csv")
    write_table(baselines, "experiment_2_position_random_controls.csv")
    write_table(transfer, "experiment_3_probe_transfer.csv")
    write_table(alignment, "experiment_4_output_head_alignment.csv")
    write_table(branch_manifold, "experiment_4b_branch_manifold_split.csv")
    write_table(intervention, "experiment_5_checkpoint_intervention.csv")
    write_table(length, "experiment_6_length_generalization.csv")
    write_table(density, "experiment_7_density_curve.csv")

    make_figures(extended, baselines, transfer, alignment, branch_manifold, intervention, length, density)
    print(f"saved Task B extension tables to {OUT_DIR}")
    print(f"saved Task B extension figures to {FIG_DIR}")


def existing_runs(experiments: list[str], *, family: str) -> list[RunSpec]:
    runs = []
    for experiment in experiments:
        exp_dir = ROOT / "results" / experiment
        if not exp_dir.exists():
            continue
        for run_dir in sorted(path for path in exp_dir.iterdir() if path.is_dir()):
            config_path = run_dir / "config.json"
            if not config_path.exists():
                continue
            config = load_json(config_path)
            runs.append(
                RunSpec(
                    experiment=experiment,
                    run_key=f"{experiment}/{run_dir.name}",
                    run_dir=run_dir,
                    config=config,
                    family=family,
                )
            )
    return runs


def experiment_1_extended_training(runs: list[RunSpec]) -> pd.DataFrame:
    rows = []
    for run in runs:
        metrics_path = run.run_dir / "metrics.json"
        probe_path = run.run_dir / "probes" / "layerwise_probe.csv"
        if not metrics_path.exists():
            continue
        metrics = load_json(metrics_path)
        behavior = behavior_log(metrics)
        max_eval = float(behavior["eval_dyck_acc"].max()) if not behavior.empty else np.nan
        final_eval = float(metrics.get("eval", {}).get("dyck_accuracy", np.nan))
        final_loss = float(metrics.get("eval", {}).get("loss", np.nan))
        best_probe = best_probe_summary(probe_path, run.config) if probe_path.exists() else {}
        rows.append(
            {
                "experiment": run.experiment,
                "family": run.family,
                "run_key": run.run_key,
                "training_steps": int(run.config.get("training_steps", 0)),
                "max_eval_dyck_acc": max_eval,
                "final_eval_dyck_acc": final_eval,
                "final_eval_loss": final_loss,
                **best_probe,
            }
        )
    return pd.DataFrame(rows)


def experiment_2_baseline_controls(runs: list[RunSpec], *, max_rows: int) -> pd.DataFrame:
    rows = []
    for run in runs:
        final_layer = best_final_layer(run)
        labels = load_labels(run, "final")
        y = labels["height"].to_numpy(dtype=float)
        idx = sample_indices(len(labels), max_rows=max_rows, seed=stable_seed(run.run_key, 2))
        labels_sub = labels.iloc[idx].reset_index(drop=True)
        y_sub = y[idx]

        position_features = one_hot(labels_sub["position"].to_numpy(dtype=int))
        rows.append(score_control(run, "position_one_hot", position_features, y_sub))

        progress_features = np.stack(
            [
                labels_sub["position"].to_numpy(dtype=float) / max(float(run.config["task"]["seq_len"] - 1), 1.0),
                labels_sub["dyck_seen"].to_numpy(dtype=float) / max(float(run.config["task"]["total_length"]), 1.0),
            ],
            axis=1,
        )
        rows.append(score_control(run, "position_plus_progress", progress_features, y_sub))

        X_step0 = load_hidden(run, "step_0", final_layer, idx)
        rows.append(score_control(run, "random_model_hidden_step0", X_step0, y_sub))

        X_final = load_hidden(run, "final", final_layer, idx)
        rows.append(score_control(run, "trained_hidden_final", X_final, y_sub))

        rng = np.random.default_rng(stable_seed(run.run_key, 22))
        rows.append(score_control(run, "trained_hidden_shuffled_height", X_final, rng.permutation(y_sub)))
    return pd.DataFrame(rows)


def experiment_3_probe_transfer(runs: list[RunSpec], *, max_rows: int) -> pd.DataFrame:
    rows = []
    for run in runs:
        layer = best_final_layer(run)
        checkpoints = available_checkpoints(run)
        labels = load_labels(run, checkpoints[0])
        y = labels["height"].to_numpy(dtype=float)
        idx = sample_indices(len(labels), max_rows=max_rows, seed=stable_seed(run.run_key, 3))
        train_idx, test_idx = split_indices(len(idx), seed=stable_seed(run.run_key, 31))
        y_train = y[idx][train_idx]
        y_test = y[idx][test_idx]
        hidden = {checkpoint: load_hidden(run, checkpoint, layer, idx) for checkpoint in checkpoints}
        models = {}
        for source in checkpoints:
            models[source] = fit_ridge(hidden[source][train_idx], y_train)
        for source in checkpoints:
            for target in checkpoints:
                pred = predict_ridge(models[source], hidden[target][test_idx])
                rows.append(
                    {
                        "experiment": run.experiment,
                        "run_key": run.run_key,
                        "layer": int(layer),
                        "source_checkpoint": source,
                        "target_checkpoint": target,
                        "source_step": checkpoint_to_step(source, run.training_steps),
                        "target_step": checkpoint_to_step(target, run.training_steps),
                        "transfer_r2": r2_score(y_test, pred),
                        "transfer_mae": float(np.abs(y_test - pred).mean()),
                    }
                )
    return pd.DataFrame(rows)


def experiment_4_output_head_alignment(runs: list[RunSpec], *, max_rows: int) -> pd.DataFrame:
    rows = []
    for run in runs:
        labels_by_checkpoint = {}
        for checkpoint in available_checkpoints(run):
            labels = prepare_target_labels(load_labels(run, checkpoint), run.config)
            target_mask = labels["has_target_label"].to_numpy(dtype=bool) & labels["target_is_dyck_position"].to_numpy(dtype=bool)
            idx = choose_rows(target_mask, max_rows=max_rows, seed=stable_seed(run.run_key + checkpoint, 4))
            labels_by_checkpoint[checkpoint] = (labels, idx)
            head = load_output_head(run, checkpoint)
            close_token = int(run.config["task"]["num_noise_tokens"])
            open_token = close_token + 1
            w_margin = head["weight"][close_token] - head["weight"][open_token]
            b_margin = head["bias"][close_token] - head["bias"][open_token]
            for layer in available_layers(run, checkpoint):
                direction = load_direction(run, checkpoint, layer, "height")
                X = load_hidden(run, checkpoint, layer, idx)
                axis = centered_projection(X, direction)
                margin = X @ w_margin + b_margin
                height = labels.iloc[idx]["height"].to_numpy(dtype=float)
                rows.append(
                    {
                        "experiment": run.experiment,
                        "run_key": run.run_key,
                        "checkpoint": checkpoint,
                        "step": checkpoint_to_step(checkpoint, run.training_steps),
                        "layer": int(layer),
                        "height_r2": probe_metric(run, checkpoint, layer, "height_r2"),
                        "cosine_height_dir_close_minus_open": cosine(direction, w_margin),
                        "corr_height_axis_with_close_minus_open_margin": pearson(axis, margin),
                        "corr_true_height_with_close_minus_open_margin": pearson(height, margin),
                        "margin_std": float(np.std(margin)),
                        "height_axis_std": float(np.std(axis)),
                    }
                )
    return pd.DataFrame(rows)


def experiment_4b_branch_manifold_split(runs: list[RunSpec], *, max_rows: int) -> pd.DataFrame:
    rows = []
    for run in runs:
        close_token = int(run.config["task"]["num_noise_tokens"])
        open_token = close_token + 1
        for checkpoint in available_checkpoints(run):
            layer = best_layer_at_checkpoint(run, checkpoint)
            labels = prepare_target_labels(load_labels(run, checkpoint), run.config)
            target = labels["target_token"].fillna(-1).to_numpy(dtype=int)
            current = labels["token"].fillna(-1).to_numpy(dtype=int)
            branch_specs = [
                (
                    "current_token",
                    labels["is_dyck_position"].to_numpy(dtype=bool) & np.isin(current, [close_token, open_token]),
                    current,
                ),
                (
                    "next_target",
                    labels["has_target_label"].to_numpy(dtype=bool)
                    & labels["target_is_dyck_position"].to_numpy(dtype=bool)
                    & np.isin(target, [close_token, open_token]),
                    target,
                ),
            ]
            for branch_source, branch_mask, branch_tokens in branch_specs:
                idx = choose_rows(branch_mask, max_rows=max_rows, seed=stable_seed(run.run_key + checkpoint + branch_source, 44))
                if len(idx) < 20:
                    continue
                X = load_hidden(run, checkpoint, layer, idx).astype(np.float64)
                y_height = labels.iloc[idx]["height"].to_numpy(dtype=float)
                y_branch = np.where(branch_tokens[idx] == close_token, 1.0, -1.0)
                if len(np.unique(y_branch)) < 2:
                    continue

                train_idx, test_idx = split_indices(len(idx), seed=stable_seed(run.run_key + checkpoint + branch_source, 45))
                X_train, X_test = X[train_idx], X[test_idx]
                h_train = y_height[train_idx]
                branch_train, branch_test = y_branch[train_idx], y_branch[test_idx]

                stats = feature_stats(X_train)
                height_model = fit_ridge_fixed_stats(X_train, h_train, stats=stats)
                height_dir = normalize(raw_direction(height_model))
                train_mean = stats["mean"]
                X_train_res = remove_direction_component(X_train, height_dir, train_mean)
                X_test_res = remove_direction_component(X_test, height_dir, train_mean)

                full_branch_model = fit_ridge_fixed_stats(X_train, branch_train, stats=stats)
                full_scores = predict_ridge_fixed_stats(full_branch_model, X_test)
                residual_stats = feature_stats(X_train_res)
                residual_branch_model = fit_ridge_fixed_stats(X_train_res, branch_train, stats=residual_stats)
                residual_scores = predict_ridge_fixed_stats(residual_branch_model, X_test_res)
                axis_scores = fit_predict_scalar_classifier((X_train - train_mean) @ height_dir, branch_train, (X_test - train_mean) @ height_dir)

                close_train = branch_train > 0
                open_train = branch_train < 0
                if close_train.sum() >= 5 and open_train.sum() >= 5:
                    close_height_dir = normalize(raw_direction(fit_ridge_fixed_stats(X_train[close_train], h_train[close_train], stats=stats)))
                    open_height_dir = normalize(raw_direction(fit_ridge_fixed_stats(X_train[open_train], h_train[open_train], stats=stats)))
                    branch_height_cos = cosine(close_height_dir, open_height_dir)
                else:
                    branch_height_cos = float("nan")

                head = load_output_head(run, checkpoint)
                w_margin = head["weight"][close_token] - head["weight"][open_token]
                w_margin_res = remove_vector_component(w_margin, height_dir)
                residual_branch_dir = normalize(raw_direction(residual_branch_model))
                full_branch_dir = normalize(raw_direction(full_branch_model))
                height_axis = (X - train_mean) @ height_dir
                residual_branch_axis = remove_direction_component(X, height_dir, train_mean) @ residual_branch_dir

                rows.append(
                    {
                        "experiment": run.experiment,
                        "run_key": run.run_key,
                        "checkpoint": checkpoint,
                        "step": checkpoint_to_step(checkpoint, run.training_steps),
                        "layer": int(layer),
                        "branch_source": branch_source,
                        "n": int(len(idx)),
                        "height_r2": probe_metric(run, checkpoint, layer, "height_r2"),
                        "close_fraction": float((y_branch > 0).mean()),
                        "branch_acc_full_hidden": balanced_accuracy(branch_test, full_scores),
                        "branch_acc_height_axis_only": balanced_accuracy(branch_test, axis_scores),
                        "branch_acc_residual_after_height": balanced_accuracy(branch_test, residual_scores),
                        "height_dir_close_open_cosine": branch_height_cos,
                        "cosine_height_dir_close_minus_open_head": cosine(height_dir, w_margin),
                        "cosine_full_branch_dir_close_minus_open_head": cosine(full_branch_dir, w_margin),
                        "cosine_residual_branch_dir_close_minus_open_head": cosine(residual_branch_dir, w_margin_res),
                        "corr_height_axis_with_branch": pearson(height_axis, y_branch),
                        "corr_residual_branch_axis_with_branch": pearson(residual_branch_axis, y_branch),
                        "residual_branch_axis_std": float(np.std(residual_branch_axis)),
                    }
                )
    return pd.DataFrame(rows)


def experiment_5_checkpoint_intervention(runs: list[RunSpec], *, max_rows: int) -> pd.DataFrame:
    rows = []
    deltas = [-2.0, -1.0, 0.0, 1.0, 2.0]
    for run in runs:
        for checkpoint in available_checkpoints(run):
            layer = best_layer_at_checkpoint(run, checkpoint)
            labels = prepare_target_labels(load_labels(run, checkpoint), run.config)
            target_mask = labels["has_target_label"].to_numpy(dtype=bool) & labels["target_is_dyck_position"].to_numpy(dtype=bool)
            idx = choose_rows(target_mask, max_rows=max_rows, seed=stable_seed(run.run_key + checkpoint, 5))
            if len(idx) == 0:
                continue
            X = load_hidden(run, checkpoint, layer, idx)
            direction = normalize(load_direction(run, checkpoint, layer, "height"))
            head = load_output_head(run, checkpoint)
            close_token = int(run.config["task"]["num_noise_tokens"])
            open_token = close_token + 1
            logits = X @ head["weight"].T + head["bias"]
            axis_scale = float(np.std(X @ direction))
            if axis_scale < 1e-8:
                axis_scale = 1.0
            shift_unit = axis_scale * direction
            target = labels.iloc[idx]["target_token"].to_numpy(dtype=int)
            state = labels.iloc[idx]["forced_state"].to_numpy()
            for delta in deltas:
                shifted = logits + float(delta) * (shift_unit @ head["weight"].T)
                pred = shifted.argmax(axis=1)
                margin = shifted[:, close_token] - shifted[:, open_token]
                p_close = sigmoid(margin)
                for split_name, mask in split_masks(state).items():
                    if not bool(mask.any()):
                        continue
                    rows.append(
                        {
                            "experiment": run.experiment,
                            "run_key": run.run_key,
                            "checkpoint": checkpoint,
                            "step": checkpoint_to_step(checkpoint, run.training_steps),
                            "layer": int(layer),
                            "split": split_name,
                            "delta_height_axis_std": float(delta),
                            "n": int(mask.sum()),
                            "accuracy": float((pred[mask] == target[mask]).mean()),
                            "mean_p_close_given_bracket": float(p_close[mask].mean()),
                            "mean_close_minus_open_margin": float(margin[mask].mean()),
                            "axis_scale": axis_scale,
                        }
                    )
    return pd.DataFrame(rows)


@torch.no_grad()
def experiment_6_length_generalization(
    runs: list[RunSpec],
    *,
    device: str,
    num_examples: int,
    batch_size: int,
) -> pd.DataFrame:
    rows = []
    for run in runs:
        checkpoints = select_checkpoints_for_eval(run)
        for checkpoint in checkpoints:
            model = load_model(run, checkpoint, device)
            for eval_name, task_kwargs in length_eval_tasks(run.config["task"]).items():
                sampler = build_sampler(task_name_from_run_config(run.config), task_kwargs, device="cpu", seed=int(run.config["seed"]) + 70_000)
                metrics = evaluate_model(model, sampler, num_examples=num_examples, batch_size=batch_size, device=device)
                rows.append(
                    {
                        "experiment": run.experiment,
                        "run_key": run.run_key,
                        "checkpoint": checkpoint,
                        "step": checkpoint_to_step(checkpoint, run.training_steps),
                        "eval_setting": eval_name,
                        "eval_seq_len": int(task_kwargs["seq_len"]),
                        "eval_repeat_prob": float(task_kwargs["repeat_prob"]),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def experiment_7_density_curve() -> pd.DataFrame:
    path = ROOT / "results" / "dyck_counter_sparse_supervision_ablation" / "summary.csv"
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path)
    rename = {
        "bracket_density": "density",
        "train_eval_dyck_accuracy": "dyck_accuracy",
        "all_dyck_targets_oracle_acc": "oracle_accuracy",
        "forced_model_acc": "forced_accuracy",
        "free_model_acc": "free_accuracy",
        "legal_next_class_accuracy": "legal_next_accuracy",
    }
    data = data.rename(columns={old: new for old, new in rename.items() if old in data.columns})
    columns = [
        "source",
        "bracket_tokens",
        "density",
        "dyck_accuracy",
        "oracle_accuracy",
        "forced_accuracy",
        "free_accuracy",
        "height_r2",
        "legal_next_accuracy",
    ]
    available = [column for column in columns if column in data.columns]
    return data[available].sort_values("bracket_tokens").reset_index(drop=True)


def make_figures(
    extended: pd.DataFrame,
    baselines: pd.DataFrame,
    transfer: pd.DataFrame,
    alignment: pd.DataFrame,
    branch_manifold: pd.DataFrame,
    intervention: pd.DataFrame,
    length: pd.DataFrame,
    density: pd.DataFrame,
) -> None:
    plot_extended_training(extended)
    plot_baseline_controls(baselines)
    plot_probe_transfer(transfer)
    plot_output_head_alignment(alignment)
    plot_branch_manifold_split(branch_manifold)
    plot_intervention(intervention)
    plot_length_generalization(length)
    plot_density_curve(density)


def plot_extended_training(df: pd.DataFrame) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for experiment, group in df.sort_values("training_steps").groupby("experiment"):
        ax.scatter(group["training_steps"], group["final_eval_dyck_acc"], label=experiment)
    ax.set_title("Extended training smoke")
    ax.set_xlabel("training steps")
    ax.set_ylabel("final Dyck next-token accuracy")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "experiment_1_extended_training.png", dpi=180)
    plt.close(fig)


def plot_baseline_controls(df: pd.DataFrame) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = []
    x = np.arange(len(df))
    ax.bar(x, df["height_r2"])
    for _, row in df.iterrows():
        labels.append(f"{short_name(row['experiment'])}\n{row['control']}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_title("Position/random/shuffle controls")
    ax.set_ylabel("held-out height R2")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "experiment_2_position_random_controls.png", dpi=180)
    plt.close(fig)


def plot_probe_transfer(df: pd.DataFrame) -> None:
    if df.empty:
        return
    runs = list(df["experiment"].drop_duplicates())
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 5), squeeze=False)
    for ax, experiment in zip(axes[0], runs):
        sub = df[df["experiment"].eq(experiment)]
        pivot = sub.pivot_table(index="source_step", columns="target_step", values="transfer_r2", aggfunc="mean")
        image = ax.imshow(pivot.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_title(short_name(experiment))
        ax.set_xlabel("target checkpoint step")
        ax.set_ylabel("source probe step")
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels([str(int(v)) for v in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels([str(int(v)) for v in pivot.index])
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.suptitle("Checkpoint-to-checkpoint height-probe transfer R2")
    fig.savefig(FIG_DIR / "experiment_3_probe_transfer.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_output_head_alignment(df: pd.DataFrame) -> None:
    if df.empty:
        return
    best = best_layer_rows(df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for experiment, group in best.groupby("experiment"):
        group = group.sort_values("step")
        axes[0].plot(group["step"], group["cosine_height_dir_close_minus_open"], marker="o", label=short_name(experiment))
        axes[1].plot(group["step"], group["corr_height_axis_with_close_minus_open_margin"], marker="o", label=short_name(experiment))
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("height direction vs output head")
    axes[1].set_title("height axis vs close-open margin")
    for ax in axes:
        ax.set_xlabel("training step")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("cosine")
    axes[1].set_ylabel("Pearson r")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "experiment_4_output_head_alignment.png", dpi=180)
    plt.close(fig)


def plot_branch_manifold_split(df: pd.DataFrame) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for (experiment, branch_source), group in df.groupby(["experiment", "branch_source"]):
        group = group.sort_values("step")
        label = f"{short_name(experiment)} {branch_source}"
        axes[0].plot(group["step"], group["branch_acc_residual_after_height"], marker="o", label=f"{label} residual")
        axes[0].plot(group["step"], group["branch_acc_height_axis_only"], marker="x", linestyle="--", label=f"{label} height-only")
        axes[1].plot(group["step"], group["height_dir_close_open_cosine"], marker="o", label=label)
    next_target = df[df["branch_source"].eq("next_target")]
    for experiment, group in next_target.groupby("experiment"):
        group = group.sort_values("step")
        label = short_name(experiment)
        axes[2].plot(group["step"], group["cosine_residual_branch_dir_close_minus_open_head"], marker="o", label=label)
    axes[0].axhline(0.5, color="black", linewidth=0.8)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Open/close branch separability")
    axes[1].set_title("Height direction shared across branches")
    axes[2].set_title("Next-target residual branch vs output head")
    axes[0].set_ylabel("balanced accuracy")
    axes[1].set_ylabel("cosine")
    axes[2].set_ylabel("cosine")
    for ax in axes:
        ax.set_xlabel("training step")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "experiment_4b_branch_manifold_split.png", dpi=180)
    plt.close(fig)


def plot_intervention(df: pd.DataFrame) -> None:
    if df.empty:
        return
    focused = df[df["split"].eq("all_dyck_targets")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for experiment, group in focused.groupby("experiment"):
        for step, sub in group.groupby("step"):
            if step not in {0, 1000, 2000}:
                continue
            sub = sub.sort_values("delta_height_axis_std")
            axes[0].plot(sub["delta_height_axis_std"], sub["mean_p_close_given_bracket"], marker="o", label=f"{short_name(experiment)} s{step}")
            axes[1].plot(sub["delta_height_axis_std"], sub["accuracy"], marker="o", label=f"{short_name(experiment)} s{step}")
    axes[0].set_title("Direct height-axis intervention: P(close)")
    axes[1].set_title("Direct height-axis intervention: accuracy")
    for ax in axes:
        ax.set_xlabel("delta along height axis std")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6)
    axes[0].set_ylabel("mean P(close | bracket logits)")
    axes[1].set_ylabel("next-token accuracy")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "experiment_5_checkpoint_intervention.png", dpi=180)
    plt.close(fig)


def plot_length_generalization(df: pd.DataFrame) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (experiment, group) in zip(axes, df.groupby("experiment")):
        for checkpoint, sub in group.groupby("checkpoint"):
            sub = sub.sort_values("eval_seq_len")
            ax.plot(sub["eval_seq_len"], sub["dyck_accuracy"], marker="o", label=checkpoint)
        ax.set_xscale("log")
        ax.set_title(short_name(experiment))
        ax.set_xlabel("eval seq_len")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Dyck next-token accuracy")
    fig.suptitle("Length generalization over checkpoints")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "experiment_6_length_generalization.png", dpi=180)
    plt.close(fig)


def plot_density_curve(df: pd.DataFrame) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if "dyck_accuracy" in df:
        axes[0].plot(df["bracket_tokens"], df["dyck_accuracy"], marker="o", label="Dyck acc")
    if "forced_accuracy" in df:
        axes[0].plot(df["bracket_tokens"], df["forced_accuracy"], marker="o", label="forced acc")
    if "free_accuracy" in df:
        axes[0].plot(df["bracket_tokens"], df["free_accuracy"], marker="o", label="free acc")
    if "height_r2" in df:
        axes[1].plot(df["bracket_tokens"], df["height_r2"], marker="o", label="height R2")
    if "legal_next_accuracy" in df:
        axes[1].plot(df["bracket_tokens"], df["legal_next_accuracy"], marker="o", label="legal-next acc")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("bracket tokens in seq_len=2000")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("behavior accuracy")
    axes[1].set_ylabel("probe score")
    fig.suptitle("Density curve proxy from Task A sparse ladder")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "experiment_7_density_curve.png", dpi=180)
    plt.close(fig)


def write_table(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(OUT_DIR / filename, index=False)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def behavior_log(metrics: dict) -> pd.DataFrame:
    train = metrics.get("train", {})
    return pd.DataFrame(
        {
            "step": train.get("step", []),
            "eval_loss": train.get("eval_loss", []),
            "eval_acc": train.get("eval_acc", []),
            "eval_dyck_acc": train.get("eval_dyck_acc", []),
        }
    )


def best_probe_summary(path: Path, config: dict) -> dict:
    df = pd.read_csv(path)
    training_steps = int(config.get("training_steps", 0))
    df["step"] = df["checkpoint"].map(lambda value: checkpoint_to_step(value, training_steps))
    final = df[df["step"].eq(df["step"].max())]
    if final.empty:
        final = df
    row = final.loc[final["height_r2"].astype(float).idxmax()]
    ready = df[(df["height_r2"] >= 0.8) & (df.get("height_class_accuracy", 1.0) >= 0.5)].copy()
    ready["step"] = ready["checkpoint"].map(lambda value: checkpoint_to_step(value, training_steps))
    return {
        "final_best_layer": int(row["layer"]),
        "final_height_r2": float(row["height_r2"]),
        "final_height_class_accuracy": float(row.get("height_class_accuracy", np.nan)),
        "probe_emergence_step": int(ready["step"].min()) if not ready.empty else np.nan,
    }


def best_final_layer(run: RunSpec) -> int:
    path = run.run_dir / "probes" / "layerwise_probe.csv"
    df = pd.read_csv(path)
    final = df[df["checkpoint"].eq("final")]
    if final.empty:
        final = df[df["checkpoint"].eq(df["checkpoint"].iloc[-1])]
    return int(final.loc[final["height_r2"].astype(float).idxmax(), "layer"])


def best_layer_at_checkpoint(run: RunSpec, checkpoint: str) -> int:
    df = pd.read_csv(run.run_dir / "probes" / "layerwise_probe.csv")
    sub = df[df["checkpoint"].eq(checkpoint)]
    if sub.empty:
        sub = df[df["checkpoint"].eq("final")]
    return int(sub.loc[sub["height_r2"].astype(float).idxmax(), "layer"])


def probe_metric(run: RunSpec, checkpoint: str, layer: int, metric: str) -> float:
    df = pd.read_csv(run.run_dir / "probes" / "layerwise_probe.csv")
    sub = df[df["checkpoint"].eq(checkpoint) & df["layer"].eq(layer)]
    if sub.empty or metric not in sub:
        return float("nan")
    return float(sub.iloc[0][metric])


def available_checkpoints(run: RunSpec) -> list[str]:
    hidden_root = run.run_dir / "hidden_states"
    checkpoints = [path.name for path in hidden_root.iterdir() if path.is_dir() and (path / "labels.parquet").exists()]
    return sorted(checkpoints, key=lambda name: checkpoint_order(name, run.training_steps))


def available_layers(run: RunSpec, checkpoint: str) -> list[int]:
    path = run.run_dir / "hidden_states" / checkpoint
    return sorted(int(layer_path.stem.removeprefix("layer_")) for layer_path in path.glob("layer_*.pt"))


def checkpoint_to_step(checkpoint: object, training_steps: int) -> int:
    name = str(checkpoint)
    if name == "final":
        return int(training_steps)
    match = CHECKPOINT_RE.match(name)
    if match:
        return int(match.group(1))
    if name.isdigit():
        return int(name)
    raise ValueError(f"Cannot parse checkpoint {checkpoint!r}")


def checkpoint_order(checkpoint: object, training_steps: int) -> float:
    step = checkpoint_to_step(checkpoint, training_steps)
    return float(step) + (0.5 if str(checkpoint) == "final" else 0.0)


def load_labels(run: RunSpec, checkpoint: str) -> pd.DataFrame:
    return pd.read_parquet(run.run_dir / "hidden_states" / checkpoint / "labels.parquet").sort_values(["example_id", "position"]).reset_index(drop=True)


def load_hidden(run: RunSpec, checkpoint: str, layer: int, indices: np.ndarray | None = None) -> np.ndarray:
    tensor = torch.load(run.run_dir / "hidden_states" / checkpoint / f"layer_{layer}.pt", map_location="cpu").float()
    if indices is not None:
        tensor = tensor[torch.as_tensor(indices, dtype=torch.long)]
    return tensor.numpy()


def load_direction(run: RunSpec, checkpoint: str, layer: int, target: str) -> np.ndarray:
    path = run.run_dir / "probes" / "directions" / f"{checkpoint}_layer_{layer}_{target}.pt"
    direction = torch.load(path, map_location="cpu").float().reshape(-1).numpy()
    return normalize(direction)


def load_output_head(run: RunSpec, checkpoint: str) -> dict[str, np.ndarray]:
    ckpt = torch.load(checkpoint_path(run, checkpoint), map_location="cpu")
    state = ckpt["model"]
    return {
        "weight": state["output.weight"].detach().float().numpy(),
        "bias": state["output.bias"].detach().float().numpy(),
    }


def checkpoint_path(run: RunSpec, checkpoint: str) -> Path:
    if checkpoint == "final":
        return run.run_dir / "checkpoints" / "model_final.pt"
    step = checkpoint_to_step(checkpoint, run.training_steps)
    return run.run_dir / "checkpoints" / f"model_step_{step}.pt"


def prepare_target_labels(labels: pd.DataFrame, config: dict) -> pd.DataFrame:
    labels = labels.copy()
    grouped = labels.groupby("example_id", sort=False)
    next_position = grouped["position"].shift(-1)
    labels["target_position"] = labels["position"] + 1
    labels["has_target_label"] = next_position.eq(labels["target_position"])
    labels["target_token"] = grouped["token"].shift(-1)
    labels["target_is_dyck_position"] = grouped["is_dyck_position"].shift(-1).fillna(False).astype(bool)
    total_length = int(config["task"]["total_length"])
    max_opens = total_length // 2
    remaining_dyck = total_length - labels["dyck_seen"].to_numpy()
    remaining_opens = max_opens - labels["left"].to_numpy()
    height = labels["height"].to_numpy()
    must_open = height <= 0
    must_close = (remaining_opens <= 0) | (remaining_dyck <= height)
    labels["forced_state"] = np.select([must_open, must_close], ["must_open", "must_close"], default="free")
    return labels


def split_masks(state: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "all_dyck_targets": np.ones(len(state), dtype=bool),
        "forced": state != "free",
        "free": state == "free",
        "must_open": state == "must_open",
        "must_close": state == "must_close",
    }


def one_hot(values: np.ndarray) -> np.ndarray:
    values = values.astype(int)
    out = np.zeros((len(values), int(values.max()) + 1), dtype=np.float32)
    out[np.arange(len(values)), values] = 1.0
    return out


def score_control(run: RunSpec, control: str, X: np.ndarray, y: np.ndarray) -> dict[str, object]:
    train_idx, test_idx = split_indices(len(y), seed=stable_seed(run.run_key + control, 20))
    model = fit_ridge(X[train_idx], y[train_idx])
    pred = predict_ridge(model, X[test_idx])
    rounded = np.rint(pred)
    return {
        "experiment": run.experiment,
        "run_key": run.run_key,
        "control": control,
        "n": int(len(y)),
        "height_r2": r2_score(y[test_idx], pred),
        "height_mae": float(np.abs(y[test_idx] - pred).mean()),
        "rounded_height_accuracy": float((rounded == y[test_idx]).mean()),
    }


def fit_ridge(X: np.ndarray, y: np.ndarray, *, alpha: float = 1.0) -> dict[str, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    Xs = (X - mean) / std
    X_aug = np.concatenate([Xs, np.ones((len(Xs), 1))], axis=1)
    eye = np.eye(X_aug.shape[1])
    eye[-1, -1] = 0.0
    coef = np.linalg.solve(X_aug.T @ X_aug + alpha * eye, X_aug.T @ y)
    return {"coef": coef[:-1], "bias": np.array([coef[-1]]), "mean": mean.squeeze(), "std": std.squeeze()}


def feature_stats(X: np.ndarray) -> dict[str, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    return {"mean": mean, "std": std}


def fit_ridge_fixed_stats(
    X: np.ndarray,
    y: np.ndarray,
    *,
    stats: dict[str, np.ndarray],
    alpha: float = 1.0,
) -> dict[str, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Xs = (X - stats["mean"]) / stats["std"]
    X_aug = np.concatenate([Xs, np.ones((len(Xs), 1))], axis=1)
    eye = np.eye(X_aug.shape[1])
    eye[-1, -1] = 0.0
    coef = np.linalg.solve(X_aug.T @ X_aug + alpha * eye, X_aug.T @ y)
    return {
        "coef": coef[:-1],
        "bias": np.array([coef[-1]]),
        "mean": stats["mean"],
        "std": stats["std"],
    }


def predict_ridge(model: dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    Xs = (np.asarray(X, dtype=np.float64) - model["mean"]) / model["std"]
    return Xs @ model["coef"] + float(model["bias"][0])


def predict_ridge_fixed_stats(model: dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    return predict_ridge(model, X)


def raw_direction(model: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(model["coef"], dtype=np.float64) / np.asarray(model["std"], dtype=np.float64)


def remove_direction_component(X: np.ndarray, direction: np.ndarray, mean: np.ndarray) -> np.ndarray:
    direction = normalize(direction)
    centered = np.asarray(X, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    return centered - np.outer(centered @ direction, direction)


def remove_vector_component(vector: np.ndarray, direction: np.ndarray) -> np.ndarray:
    direction = normalize(direction)
    vector = np.asarray(vector, dtype=np.float64)
    return vector - float(vector @ direction) * direction


def fit_predict_scalar_classifier(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=np.float64).reshape(-1, 1)
    x_test = np.asarray(x_test, dtype=np.float64).reshape(-1, 1)
    model = fit_ridge(x_train, y_train)
    return predict_ridge(model, x_test)


def balanced_accuracy(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    pred = np.where(np.asarray(scores, dtype=float) >= 0.0, 1.0, -1.0)
    values = []
    for label in [-1.0, 1.0]:
        mask = y_true == label
        if mask.any():
            values.append(float((pred[mask] == y_true[mask]).mean()))
    return float(np.mean(values)) if values else float("nan")


def split_indices(n: int, *, seed: int, test_fraction: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * test_fraction)))
    return perm[n_test:], perm[:n_test]


def sample_indices(n: int, *, max_rows: int, seed: int) -> np.ndarray:
    if n <= max_rows:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_rows, replace=False))


def choose_rows(mask: np.ndarray, *, max_rows: int, seed: int) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if len(idx) <= max_rows:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=max_rows, replace=False))


def centered_projection(X: np.ndarray, direction: np.ndarray) -> np.ndarray:
    axis = X @ normalize(direction)
    return axis - axis.mean()


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        return vector
    return vector / norm


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = normalize(left)
    right = normalize(right)
    return float(np.dot(left, right))


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def length_eval_tasks(task: dict) -> dict[str, dict]:
    total_length = int(task["total_length"])
    base_seq_len = int(task["seq_len"])
    candidates = sorted({base_seq_len, 120, 500, 1000})
    out = {}
    for seq_len in candidates:
        if seq_len < total_length:
            continue
        repeat_prob = float(task["repeat_prob"]) if seq_len == base_seq_len else min(1.0, total_length / float(seq_len))
        task_kwargs = {key: value for key, value in task.items() if key != "device"}
        task_kwargs["seq_len"] = int(seq_len)
        task_kwargs["repeat_prob"] = float(repeat_prob)
        out[f"seq{seq_len}_p{repeat_prob:.3f}"] = task_kwargs
    return out


def select_checkpoints_for_eval(run: RunSpec) -> list[str]:
    available = set(available_checkpoints(run))
    preferred = ["step_0", "step_500", "step_1000", "step_2000", "final"]
    return [checkpoint for checkpoint in preferred if checkpoint in available]


def load_model(run: RunSpec, checkpoint: str, device: str):
    task_name = task_name_from_run_config(run.config)
    task_kwargs = {key: value for key, value in run.config["task"].items() if key != "device"}
    sampler = build_sampler(task_name, task_kwargs, device="cpu", seed=int(run.config["seed"]))
    spec = run.config["model"]
    kwargs = {key: value for key, value in spec.items() if key not in {"name", "state_kind"}}
    model = build_model(model_name=run.config["model_name"], vocab_size=sampler.vocab_size, **kwargs).to(device)
    ckpt = torch.load(checkpoint_path(run, checkpoint), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_model(model, sampler, *, num_examples: int, batch_size: int, device: str) -> dict[str, float]:
    seen = 0
    correct = 0
    total = 0
    dyck_correct = 0
    dyck_total = 0
    while seen < num_examples:
        current = min(batch_size, num_examples - seen)
        batch = sampler.sample(current)
        tokens = batch.tokens.to(device)
        logits = model(tokens)
        pred = logits[:, :-1].argmax(dim=-1).cpu()
        target = batch.tokens[:, 1:].cpu()
        all_mask = torch.ones_like(target, dtype=torch.bool)
        dyck_mask = batch.dyck_mask[:, 1:].cpu().bool()
        correct += int((pred[all_mask] == target[all_mask]).sum().item())
        total += int(all_mask.sum().item())
        if bool(dyck_mask.any()):
            dyck_correct += int((pred[dyck_mask] == target[dyck_mask]).sum().item())
            dyck_total += int(dyck_mask.sum().item())
        seen += current
    return {
        "accuracy": correct / max(total, 1),
        "dyck_accuracy": dyck_correct / max(dyck_total, 1),
        "num_examples": int(num_examples),
    }


def best_layer_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in df.groupby(["experiment", "checkpoint"], sort=False):
        idx = group["height_r2"].astype(float).idxmax()
        rows.append(group.loc[idx])
    return pd.DataFrame(rows)


def short_name(experiment: str) -> str:
    return str(experiment).removeprefix("dyck_counter_task_b_").removesuffix("_smoke").removesuffix("_5k")


def stable_seed(text: str, offset: int) -> int:
    value = int(offset)
    for char in text:
        value = (value * 131 + ord(char)) % 2_147_483_647
    return value


@property
def run_training_steps(self: RunSpec) -> int:
    return int(self.config.get("training_steps", 0))


RunSpec.training_steps = run_training_steps  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
