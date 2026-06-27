from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.task_a_extra_probes import (
    available_layers,
    choose_indices,
    load_hidden_rows,
    load_labels,
    load_output_head,
    load_probe_direction,
    prepare_labels,
    setting_seed,
    split_masks,
)


SUMMARY_PATH = ROOT / "results" / "dyck_counter_task_a_summary.csv"
OUT_DIR = ROOT / "results" / "dyck_counter_task_a_ablation"
FIG_DIR = ROOT / "figures" / "dyck_counter_task_a_ablation"
SETTING_ORDER = ["tiny_extreme_long", "clean_short", "noisy_short", "sparse_medium", "sparse_long", "extreme_long"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(SUMMARY_PATH)
    summary["run_dir_abs"] = summary["run_dir"].map(lambda path: ROOT / path)
    summary = summary.set_index("setting").reindex(SETTING_ORDER).dropna(subset=["run_dir"]).reset_index()

    rows = []
    for run in summary.itertuples(index=False):
        rows.extend(run_setting_ablation(run))
        gc.collect()

    out = pd.DataFrame(rows)
    out = add_baseline_deltas(out)
    out_path = OUT_DIR / "height_direction_ablation.csv"
    out.to_csv(out_path, index=False)
    fig_path = plot_ablation(out)
    print(f"wrote {out_path}")
    print(f"wrote {fig_path}")


def run_setting_ablation(run) -> list[dict]:
    run_dir = Path(run.run_dir_abs)
    labels = prepare_labels(load_labels(run_dir), run)
    mask = labels["has_target_label"].to_numpy(dtype=bool) & labels["target_is_dyck_position"].to_numpy(dtype=bool)
    eval_idx = choose_indices(mask, max_rows=160_000, seed=setting_seed(run.setting, 70))
    if len(eval_idx) == 0:
        return []

    final_layer = max(available_layers(run_dir))
    X = load_hidden_rows(run_dir, final_layer, eval_idx)
    y = labels.iloc[eval_idx]["target_token"].to_numpy(dtype=int)
    eval_labels = labels.iloc[eval_idx].reset_index(drop=True)
    head = load_output_head(run_dir)
    height_dir = normalize(load_probe_direction(run_dir, final_layer, "height"))
    random_dir = matched_random_direction(height_dir, seed=setting_seed(run.setting, 71))

    variants = ablated_variants(X, height_dir, random_dir, seed=setting_seed(run.setting, 72))
    rows = []
    for ablation_name, X_variant in variants.items():
        logits = X_variant @ head["weight"].T + head["bias"]
        pred = logits.argmax(axis=1)
        nll = token_nll(logits, y)
        close_token = int(run.noise_vocab)
        open_token = close_token + 1
        p_close_given_bracket = sigmoid(logits[:, close_token] - logits[:, open_token])
        margin = logits[:, close_token] - logits[:, open_token]
        correct = pred == y
        for split, split_mask in split_masks(eval_labels).items():
            if split_mask.sum() == 0:
                continue
            rows.append(
                {
                    "setting": run.setting,
                    "ablation": ablation_name,
                    "layer": int(final_layer),
                    "split": split,
                    "n": int(split_mask.sum()),
                    "accuracy": float(correct[split_mask].mean()),
                    "target_nll": float(nll[split_mask].mean()),
                    "mean_close_minus_open_margin": float(margin[split_mask].mean()),
                    "mean_p_close_given_bracket": float(p_close_given_bracket[split_mask].mean()),
                    "mean_height": float(eval_labels.loc[split_mask, "height"].mean()),
                }
            )
        del logits
    del X
    return rows


def ablated_variants(X: np.ndarray, height_dir: np.ndarray, random_dir: np.ndarray, *, seed: int) -> dict[str, np.ndarray]:
    mean = X.mean(axis=0, keepdims=True)
    X_centered = X - mean
    height_axis = X_centered @ height_dir
    random_axis = X_centered @ random_dir
    rng = np.random.default_rng(seed)
    shuffled_height_axis = rng.permutation(height_axis)
    return {
        "baseline": X,
        "remove_height_direction": X - height_axis[:, None] * height_dir[None, :],
        "shuffle_height_axis": X - height_axis[:, None] * height_dir[None, :] + shuffled_height_axis[:, None] * height_dir[None, :],
        "remove_random_direction": X - random_axis[:, None] * random_dir[None, :],
    }


def add_baseline_deltas(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    baseline = (
        df[df["ablation"].eq("baseline")]
        .set_index(["setting", "split"])[["accuracy", "target_nll", "mean_close_minus_open_margin", "mean_p_close_given_bracket"]]
        .rename(
            columns={
                "accuracy": "baseline_accuracy",
                "target_nll": "baseline_target_nll",
                "mean_close_minus_open_margin": "baseline_mean_close_minus_open_margin",
                "mean_p_close_given_bracket": "baseline_mean_p_close_given_bracket",
            }
        )
    )
    out = df.join(baseline, on=["setting", "split"])
    out["delta_accuracy_vs_baseline"] = out["accuracy"] - out["baseline_accuracy"]
    out["delta_target_nll_vs_baseline"] = out["target_nll"] - out["baseline_target_nll"]
    out["delta_margin_vs_baseline"] = out["mean_close_minus_open_margin"] - out["baseline_mean_close_minus_open_margin"]
    out["delta_p_close_vs_baseline"] = out["mean_p_close_given_bracket"] - out["baseline_mean_p_close_given_bracket"]
    return out


def plot_ablation(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
    focused = df[df["ablation"].isin(["remove_height_direction", "shuffle_height_axis", "remove_random_direction"])].copy()
    for ax, split, title in [
        (axes[0], "all_dyck_targets", "All Dyck targets"),
        (axes[1], "forced", "Forced targets"),
        (axes[2], "free", "Free targets"),
    ]:
        sub = focused[focused["split"].eq(split)]
        pivot = sub.pivot(index="setting", columns="ablation", values="delta_accuracy_vs_baseline").reindex(SETTING_ORDER)
        pivot.plot(kind="bar", ax=ax, color=["#2ca25f", "#8856a7", "#3182bd"])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(title)
        ax.set_ylabel("accuracy change")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    out = FIG_DIR / "height_direction_ablation.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def matched_random_direction(height_dir: np.ndarray, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    random_dir = rng.normal(size=height_dir.shape)
    random_dir = random_dir - np.dot(random_dir, height_dir) * height_dir
    return normalize(random_dir)


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    return x / (np.linalg.norm(x) + 1e-12)


def token_nll(logits: np.ndarray, target: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1)) + logits.max(axis=1)
    return logsumexp - logits[np.arange(len(target)), target]


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


if __name__ == "__main__":
    main()
