from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
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


CHECKPOINT_RE = re.compile(r"^step_(\d+)$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=[
            "dyck_counter_task_b_clean_short_smoke",
            "dyck_counter_task_b_noisy_short_smoke",
        ],
        help="Experiment result directory names under results/.",
    )
    parser.add_argument("--results-root", default=str(ROOT / "results"))
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "dyck_counter_task_b_training_dynamics"))
    parser.add_argument("--figure-dir", default=str(ROOT / "figures" / "dyck_counter_task_b_training_dynamics"))
    parser.add_argument("--probe-threshold", type=float, default=0.8)
    parser.add_argument("--count-acc-threshold", type=float, default=0.5)
    parser.add_argument("--behavior-metric", choices=["forced_acc", "raw_eval_dyck_acc"], default="forced_acc")
    parser.add_argument("--behavior-threshold", type=float, default=None)
    parser.add_argument("--stable-fraction", type=float, default=0.8)
    args = parser.parse_args()
    behavior_threshold = args.behavior_threshold
    if behavior_threshold is None:
        behavior_threshold = 0.95 if args.behavior_metric == "forced_acc" else 0.8

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    figure_dir = Path(args.figure_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    layerwise_tables = []
    best_tables = []
    behavior_tables = []
    forced_free_tables = []
    summary_rows = []
    figure_rows = []

    for experiment in args.experiments:
        exp_dir = results_root / experiment
        if not exp_dir.exists():
            raise FileNotFoundError(f"Missing experiment directory: {exp_dir}")
        for run_dir in sorted(path for path in exp_dir.iterdir() if path.is_dir()):
            config_path = run_dir / "config.json"
            probe_path = run_dir / "probes" / "layerwise_probe.csv"
            metrics_path = run_dir / "metrics.json"
            if not config_path.exists() or not probe_path.exists() or not metrics_path.exists():
                continue

            config = load_json(config_path)
            metrics = load_json(metrics_path)
            run_key = f"{experiment}/{run_dir.name}"
            training_steps = int(config.get("training_steps", config.get("training", {}).get("steps", 0)))
            layerwise = pd.read_csv(probe_path)
            layerwise.insert(0, "run_key", run_key)
            layerwise.insert(0, "experiment", experiment)
            layerwise.insert(1, "run_dir", str(run_dir.relative_to(ROOT)))
            layerwise["step"] = layerwise["checkpoint"].map(lambda value: checkpoint_to_step(value, training_steps))
            layerwise["checkpoint_order"] = layerwise["checkpoint"].map(lambda value: checkpoint_order(value, training_steps))
            layerwise = attach_height_direction_cosines(layerwise, run_dir)

            behavior = behavior_log_table(metrics, run_key=run_key, experiment=experiment, run_dir=run_dir, training_steps=training_steps)
            forced_free = checkpoint_forced_free_behavior(
                run_dir,
                run_key=run_key,
                experiment=experiment,
                training_steps=training_steps,
            )
            best = best_probe_by_checkpoint(layerwise)
            best = best.merge(
                behavior[["step", "eval_loss", "eval_acc", "eval_dyck_acc"]],
                on="step",
                how="left",
            )

            summary = summarize_run(
                best,
                behavior,
                run_key=run_key,
                experiment=experiment,
                run_dir=run_dir,
                probe_threshold=args.probe_threshold,
                count_acc_threshold=args.count_acc_threshold,
                behavior_metric=args.behavior_metric,
                behavior_threshold=behavior_threshold,
                stable_fraction=args.stable_fraction,
                forced_free=forced_free,
            )

            figure_path = figure_dir / f"{safe_name(run_key)}.png"
            plot_run(layerwise, best, behavior, forced_free, summary, figure_path)
            figure_rows.append(
                {
                    "run_key": run_key,
                    "figure": str(figure_path.relative_to(ROOT)),
                }
            )

            layerwise_tables.append(layerwise)
            best_tables.append(best)
            behavior_tables.append(behavior)
            if not forced_free.empty:
                forced_free_tables.append(forced_free)
            summary_rows.append(summary)

    if not summary_rows:
        raise RuntimeError("No completed runs with config.json, metrics.json, and probes/layerwise_probe.csv were found.")

    layerwise_all = pd.concat(layerwise_tables, ignore_index=True)
    best_all = pd.concat(best_tables, ignore_index=True)
    behavior_all = pd.concat(behavior_tables, ignore_index=True)
    forced_free_all = pd.concat(forced_free_tables, ignore_index=True) if forced_free_tables else pd.DataFrame()
    summary_all = pd.DataFrame(summary_rows)
    figures = pd.DataFrame(figure_rows)

    layerwise_all.to_csv(out_dir / "layerwise_checkpoint_probe.csv", index=False)
    best_all.to_csv(out_dir / "checkpoint_best_probe.csv", index=False)
    behavior_all.to_csv(out_dir / "behavior_log.csv", index=False)
    forced_free_all.to_csv(out_dir / "checkpoint_forced_free_behavior.csv", index=False)
    summary_all.to_csv(out_dir / "emergence_summary.csv", index=False)
    summary_all.to_csv(out_dir / "summary.csv", index=False)
    figures.to_csv(out_dir / "figures.csv", index=False)

    first_figure = figure_dir / f"{safe_name(str(summary_all.iloc[0]['run_key']))}.png"
    if first_figure.exists():
        shutil.copyfile(first_figure, figure_dir / "task_b_training_dynamics.png")

    print(f"saved Task B summaries to {out_dir}")
    print(f"saved Task B figures to {figure_dir}")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def checkpoint_to_step(checkpoint: object, training_steps: int) -> int:
    name = str(checkpoint)
    if name == "final":
        return int(training_steps)
    match = CHECKPOINT_RE.match(name)
    if match:
        return int(match.group(1))
    if name.isdigit():
        return int(name)
    raise ValueError(f"Cannot parse checkpoint step from {name!r}")


def checkpoint_order(checkpoint: object, training_steps: int) -> float:
    name = str(checkpoint)
    step = checkpoint_to_step(name, training_steps)
    return float(step) + (0.5 if name == "final" else 0.0)


def behavior_log_table(
    metrics: dict,
    *,
    run_key: str,
    experiment: str,
    run_dir: Path,
    training_steps: int,
) -> pd.DataFrame:
    train = metrics.get("train", {})
    frame = pd.DataFrame(
        {
            "step": train.get("step", []),
            "train_loss": train.get("loss", []),
            "eval_loss": train.get("eval_loss", []),
            "eval_acc": train.get("eval_acc", []),
            "eval_dyck_acc": train.get("eval_dyck_acc", []),
        }
    )
    if not frame.empty:
        frame["step"] = frame["step"].astype(int)
    final_eval = metrics.get("eval", {})
    if final_eval and (frame.empty or int(training_steps) not in set(frame["step"].astype(int))):
        frame = pd.concat(
            [
                frame,
                pd.DataFrame(
                    [
                        {
                            "step": int(training_steps),
                            "train_loss": np.nan,
                            "eval_loss": final_eval.get("loss", np.nan),
                            "eval_acc": final_eval.get("accuracy", np.nan),
                            "eval_dyck_acc": final_eval.get("dyck_accuracy", np.nan),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    frame.insert(0, "run_key", run_key)
    frame.insert(0, "experiment", experiment)
    frame.insert(1, "run_dir", str(run_dir.relative_to(ROOT)))
    return frame.sort_values("step").reset_index(drop=True)


def checkpoint_forced_free_behavior(
    run_dir: Path,
    *,
    run_key: str,
    experiment: str,
    training_steps: int,
    max_rows: int = 200_000,
) -> pd.DataFrame:
    config = load_json(run_dir / "config.json")
    task = config.get("task", {})
    hidden_root = run_dir / "hidden_states"
    if not hidden_root.exists():
        return pd.DataFrame()

    rows = []
    checkpoints = sorted(
        [path.name for path in hidden_root.iterdir() if path.is_dir() and (path / "labels.parquet").exists()],
        key=lambda name: checkpoint_order(name, training_steps),
    )
    for checkpoint in checkpoints:
        checkpoint_dir = hidden_root / checkpoint
        head_path = model_checkpoint_path(run_dir, checkpoint)
        if not head_path.exists():
            continue
        labels = prepare_dyck_target_labels(pd.read_parquet(checkpoint_dir / "labels.parquet"), task)
        target_mask = labels["has_target_label"].to_numpy(dtype=bool) & labels["target_is_dyck_position"].to_numpy(dtype=bool)
        eval_idx = choose_indices(target_mask, max_rows=max_rows, seed=123)
        if len(eval_idx) == 0:
            continue

        layer = max_available_layer(checkpoint_dir)
        hidden = torch.load(checkpoint_dir / f"layer_{layer}.pt", map_location="cpu")
        X = hidden[eval_idx].detach().float().numpy()
        del hidden
        head = load_output_head_at(head_path)
        logits = X @ head["weight"].T + head["bias"]
        pred = logits.argmax(axis=1)
        target = labels.iloc[eval_idx]["target_token"].to_numpy(dtype=int)
        correct = pred == target
        eval_labels = labels.iloc[eval_idx].reset_index(drop=True)

        step = checkpoint_to_step(checkpoint, training_steps)
        order = checkpoint_order(checkpoint, training_steps)
        for split_name, split_mask in split_masks(eval_labels).items():
            if split_mask.sum() == 0:
                continue
            rows.append(
                {
                    "experiment": experiment,
                    "run_key": run_key,
                    "run_dir": str(run_dir.relative_to(ROOT)),
                    "checkpoint": checkpoint,
                    "step": int(step),
                    "checkpoint_order": float(order),
                    "split": split_name,
                    "n": int(split_mask.sum()),
                    "fraction": float(split_mask.mean()),
                    "model_acc": float(correct[split_mask].mean()),
                    "oracle_acc": float(eval_labels.loc[split_mask, "oracle_next_dyck_acc"].mean()),
                    "gap_model_minus_oracle": float(
                        correct[split_mask].mean() - eval_labels.loc[split_mask, "oracle_next_dyck_acc"].mean()
                    ),
                }
            )
        del X, logits

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["run_key", "checkpoint_order", "split"]).reset_index(drop=True)


def prepare_dyck_target_labels(labels: pd.DataFrame, task: dict) -> pd.DataFrame:
    labels = labels.sort_values(["example_id", "position"]).reset_index(drop=True).copy()
    grouped = labels.groupby("example_id", sort=False)
    next_position = grouped["position"].shift(-1)
    labels["target_position"] = labels["position"] + 1
    labels["has_target_label"] = next_position.eq(labels["target_position"])
    labels["target_token"] = grouped["token"].shift(-1)
    labels["target_is_dyck_position"] = grouped["is_dyck_position"].shift(-1).fillna(False).astype(bool)

    total_length = int(task["total_length"])
    max_opens = total_length // 2
    remaining_dyck = total_length - labels["dyck_seen"].to_numpy(dtype=float)
    remaining_opens = max_opens - labels["left"].to_numpy(dtype=float)
    height = labels["height"].to_numpy(dtype=float)
    must_open = height <= 0
    must_close = (remaining_opens <= 0) | (remaining_dyck <= height)
    labels["forced_state"] = np.select([must_open, must_close], ["must_open", "must_close"], default="free")
    labels["oracle_next_dyck_acc"] = np.where(labels["forced_state"].eq("free"), 0.5, 1.0)
    return labels


def split_masks(labels: pd.DataFrame) -> dict[str, np.ndarray]:
    state = labels["forced_state"].to_numpy()
    return {
        "all_dyck_targets": np.ones(len(labels), dtype=bool),
        "forced": state != "free",
        "must_open": state == "must_open",
        "must_close": state == "must_close",
        "free": state == "free",
    }


def choose_indices(mask: np.ndarray, *, max_rows: int, seed: int) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if len(idx) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_rows, replace=False)
    return np.sort(idx)


def max_available_layer(checkpoint_dir: Path) -> int:
    layers = [int(path.stem.removeprefix("layer_")) for path in checkpoint_dir.glob("layer_*.pt")]
    if not layers:
        raise FileNotFoundError(f"No layer_*.pt files found under {checkpoint_dir}")
    return max(layers)


def model_checkpoint_path(run_dir: Path, checkpoint: str) -> Path:
    if checkpoint == "final":
        return run_dir / "checkpoints" / "model_final.pt"
    return run_dir / "checkpoints" / f"model_{checkpoint}.pt"


def load_output_head_at(checkpoint_path: Path) -> dict[str, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["model"]
    return {
        "weight": state["output.weight"].detach().float().numpy(),
        "bias": state["output.bias"].detach().float().numpy(),
    }


def best_probe_by_checkpoint(layerwise: pd.DataFrame) -> pd.DataFrame:
    if "height_r2" not in layerwise:
        raise ValueError("layerwise_probe.csv must contain height_r2 for Task B.")
    rows = []
    sort_cols = ["checkpoint_order", "layer"]
    for _, group in layerwise.sort_values(sort_cols).groupby(["run_key", "checkpoint"], sort=False):
        idx = group["height_r2"].astype(float).idxmax()
        row = group.loc[idx].copy()
        row["best_layer"] = int(row["layer"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["run_key", "checkpoint_order"]).reset_index(drop=True)


def summarize_run(
    best: pd.DataFrame,
    behavior: pd.DataFrame,
    *,
    run_key: str,
    experiment: str,
    run_dir: Path,
    probe_threshold: float,
    count_acc_threshold: float,
    behavior_metric: str,
    behavior_threshold: float,
    stable_fraction: float,
    forced_free: pd.DataFrame,
) -> dict:
    ordered = best.sort_values("checkpoint_order").reset_index(drop=True)
    probe_ready = ordered["height_r2"].astype(float) >= probe_threshold
    if "height_class_accuracy" in ordered:
        count_acc = ordered["height_class_accuracy"].astype(float)
        probe_ready &= count_acc.notna() & (count_acc >= count_acc_threshold)

    probe_step, early_layer = first_probe_step(ordered, probe_ready)
    stable_step = stable_probe_step(ordered, probe_ready, stable_fraction=stable_fraction)
    behavior_step = first_behavior_step(
        behavior,
        forced_free,
        metric=behavior_metric,
        threshold=behavior_threshold,
    )
    lag = np.nan if math.isnan(probe_step) or math.isnan(behavior_step) else behavior_step - probe_step

    final_rows = ordered.loc[ordered["step"] == ordered["step"].max()]
    final_best = final_rows.loc[final_rows["height_r2"].astype(float).idxmax()]
    final_forced = final_split_metric(forced_free, "forced", "model_acc")
    final_free = final_split_metric(forced_free, "free", "model_acc")
    final_oracle = final_split_metric(forced_free, "all_dyck_targets", "oracle_acc")
    final_oracle_gap = final_split_metric(forced_free, "all_dyck_targets", "gap_model_minus_oracle")

    return {
        "experiment": experiment,
        "run_key": run_key,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "probe_threshold": probe_threshold,
        "count_acc_threshold": count_acc_threshold,
        "behavior_metric": behavior_metric,
        "behavior_threshold": behavior_threshold,
        "probe_emergence_step": none_if_nan(probe_step),
        "stable_probe_step": none_if_nan(stable_step),
        "behavior_emergence_step": none_if_nan(behavior_step),
        "verbalization_lag": none_if_nan(lag),
        "early_feature_layer": none_if_nan(early_layer),
        "best_feature_layer": int(final_best["best_layer"]),
        "final_height_r2": float(final_best["height_r2"]),
        "final_height_class_accuracy": metric_value(final_best, "height_class_accuracy"),
        "final_legal_next_class_accuracy": metric_value(final_best, "legal_next_class_accuracy"),
        "final_eval_dyck_acc": float(behavior["eval_dyck_acc"].dropna().iloc[-1]) if not behavior["eval_dyck_acc"].dropna().empty else np.nan,
        "final_forced_acc": final_forced,
        "final_free_acc": final_free,
        "final_oracle_acc": final_oracle,
        "final_gap_model_minus_oracle": final_oracle_gap,
        "num_checkpoints_probed": int(ordered["checkpoint"].nunique()),
    }


def first_probe_step(ordered: pd.DataFrame, ready: pd.Series) -> tuple[float, float]:
    if not bool(ready.any()):
        return np.nan, np.nan
    row = ordered.loc[ready].sort_values("checkpoint_order").iloc[0]
    return float(row["step"]), float(row["best_layer"])


def stable_probe_step(ordered: pd.DataFrame, ready: pd.Series, *, stable_fraction: float) -> float:
    ready_values = ready.to_numpy(dtype=bool)
    for index, is_ready in enumerate(ready_values):
        if not is_ready:
            continue
        if ready_values[index:].mean() >= stable_fraction:
            return float(ordered.iloc[index]["step"])
    return np.nan


def first_behavior_step(
    behavior: pd.DataFrame,
    forced_free: pd.DataFrame,
    *,
    metric: str,
    threshold: float,
) -> float:
    if metric == "forced_acc":
        if forced_free.empty:
            return np.nan
        forced = forced_free.loc[forced_free["split"].eq("forced")].copy()
        if forced.empty:
            return np.nan
        ready = forced["model_acc"].astype(float) >= threshold
        if not bool(ready.any()):
            return np.nan
        return float(forced.loc[ready].sort_values("checkpoint_order").iloc[0]["step"])
    if metric != "raw_eval_dyck_acc":
        raise ValueError(f"Unknown behavior metric: {metric}")
    if behavior.empty or "eval_dyck_acc" not in behavior:
        return np.nan
    ready = behavior["eval_dyck_acc"].astype(float) >= threshold
    if not bool(ready.any()):
        return np.nan
    return float(behavior.loc[ready].sort_values("step").iloc[0]["step"])


def final_split_metric(forced_free: pd.DataFrame, split: str, column: str) -> float:
    if forced_free.empty:
        return float("nan")
    final_order = forced_free["checkpoint_order"].max()
    rows = forced_free.loc[forced_free["checkpoint_order"].eq(final_order) & forced_free["split"].eq(split)]
    if rows.empty or column not in rows:
        return float("nan")
    return float(rows.iloc[0][column])


def attach_height_direction_cosines(layerwise: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    directions_dir = run_dir / "probes" / "directions"
    if not directions_dir.exists():
        layerwise["height_direction_cosine_to_final_same_layer"] = np.nan
        return layerwise

    refs: dict[int, torch.Tensor] = {}
    max_order = layerwise["checkpoint_order"].max()
    for _, row in layerwise.loc[layerwise["checkpoint_order"] == max_order].iterrows():
        layer = int(row["layer"])
        weight = load_direction(directions_dir, str(row["checkpoint"]), layer)
        if weight is not None:
            refs[layer] = weight

    cosines = []
    for _, row in layerwise.iterrows():
        layer = int(row["layer"])
        weight = load_direction(directions_dir, str(row["checkpoint"]), layer)
        ref = refs.get(layer)
        if weight is None or ref is None:
            cosines.append(np.nan)
        else:
            cosines.append(cosine(weight, ref))
    layerwise["height_direction_cosine_to_final_same_layer"] = cosines
    return layerwise


def load_direction(directions_dir: Path, checkpoint: str, layer: int) -> torch.Tensor | None:
    path = directions_dir / f"{checkpoint}_layer_{layer}_height.pt"
    if not path.exists():
        return None
    weight = torch.load(path, map_location="cpu")
    return torch.as_tensor(weight, dtype=torch.float32).reshape(-1)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = torch.linalg.norm(left) * torch.linalg.norm(right)
    if float(denom) <= 0.0:
        return float("nan")
    return float(torch.dot(left, right) / denom)


def plot_run(
    layerwise: pd.DataFrame,
    best: pd.DataFrame,
    behavior: pd.DataFrame,
    forced_free: pd.DataFrame,
    summary: dict,
    figure_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Task B training dynamics: {summary['run_key']}", fontsize=14)

    plot_layer_heatmap(layerwise, axes[0, 0])
    plot_probe_vs_behavior(best, behavior, forced_free, axes[0, 1], summary)
    plot_legal_probe(best, behavior, forced_free, axes[1, 0])
    plot_direction_stability(layerwise, axes[1, 1])

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def plot_layer_heatmap(layerwise: pd.DataFrame, ax) -> None:
    table = layerwise.pivot_table(index="layer", columns="step", values="height_r2", aggfunc="max").sort_index()
    image = ax.imshow(table.to_numpy(dtype=float), aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title("Height R2 by checkpoint and layer")
    ax.set_xlabel("training step")
    ax.set_ylabel("layer")
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels([str(int(value)) for value in table.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels([str(int(value)) for value in table.index])
    for y in range(table.shape[0]):
        for x in range(table.shape[1]):
            value = table.iloc[y, x]
            if pd.notna(value):
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def plot_probe_vs_behavior(best: pd.DataFrame, behavior: pd.DataFrame, forced_free: pd.DataFrame, ax, summary: dict) -> None:
    by_step = best_by_step(best)
    ax.plot(by_step["step"], by_step["height_r2"], marker="o", label="best-layer height R2")
    if "height_class_accuracy" in by_step:
        ax.plot(by_step["step"], by_step["height_class_accuracy"], marker="o", label="height-class probe acc")
    ax.plot(behavior["step"], behavior["eval_dyck_acc"], marker="s", label="Dyck next-token acc")
    forced = split_curve(forced_free, "forced")
    if not forced.empty:
        ax.plot(forced["step"], forced["model_acc"], marker="^", label="forced acc")
    ax.axhline(summary["probe_threshold"], color="gray", linestyle="--", linewidth=1, label="probe threshold")
    ax.axhline(summary["behavior_threshold"], color="#059669", linestyle=":", linewidth=1, label="behavior threshold")
    ax.set_title("Probe emergence vs behavior")
    ax.set_xlabel("training step")
    ax.set_ylabel("score")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def plot_legal_probe(best: pd.DataFrame, behavior: pd.DataFrame, forced_free: pd.DataFrame, ax) -> None:
    by_step = best_by_step(best)
    if "legal_next_class_accuracy" in by_step:
        ax.plot(by_step["step"], by_step["legal_next_class_accuracy"], marker="o", label="legal-next probe acc")
    ax.plot(behavior["step"], behavior["eval_dyck_acc"], marker="s", label="Dyck next-token acc")
    forced = split_curve(forced_free, "forced")
    free = split_curve(forced_free, "free")
    if not forced.empty:
        ax.plot(forced["step"], forced["model_acc"], marker="^", label="forced acc")
    if not free.empty:
        ax.plot(free["step"], free["model_acc"], marker="v", label="free acc")
    ax.set_title("Decision-relevant probe vs behavior")
    ax.set_xlabel("training step")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def plot_direction_stability(layerwise: pd.DataFrame, ax) -> None:
    if "height_direction_cosine_to_final_same_layer" not in layerwise:
        ax.set_axis_off()
        return
    for layer, group in layerwise.sort_values("step").groupby("layer"):
        by_step = group.groupby("step", as_index=False)["height_direction_cosine_to_final_same_layer"].mean()
        ax.plot(by_step["step"], by_step["height_direction_cosine_to_final_same_layer"], marker="o", label=f"layer {int(layer)}")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Height direction cosine to final")
    ax.set_xlabel("training step")
    ax.set_ylabel("cosine")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def best_by_step(best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in best.groupby("step", sort=True):
        idx = group["height_r2"].astype(float).idxmax()
        rows.append(group.loc[idx])
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def split_curve(forced_free: pd.DataFrame, split: str) -> pd.DataFrame:
    if forced_free.empty:
        return pd.DataFrame()
    curve = forced_free.loc[forced_free["split"].eq(split), ["step", "checkpoint_order", "model_acc"]].copy()
    if curve.empty:
        return curve
    return curve.sort_values(["checkpoint_order", "step"]).reset_index(drop=True)


def metric_value(row: pd.Series, key: str) -> float:
    if key not in row or pd.isna(row[key]):
        return float("nan")
    return float(row[key])


def none_if_nan(value):
    if value is None:
        return np.nan
    try:
        if math.isnan(float(value)):
            return np.nan
    except Exception:
        pass
    if float(value).is_integer():
        return int(value)
    return float(value)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip("/"))


if __name__ == "__main__":
    main()
