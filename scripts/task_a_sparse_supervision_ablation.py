from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hse.utils import load_json
from scripts.task_a_extra_probes import (
    choose_indices,
    load_labels,
    model_predictions,
    prepare_labels,
    setting_seed,
    split_masks,
)


OUT_DIR = ROOT / "results" / "dyck_counter_sparse_supervision_ablation"
FIG_DIR = ROOT / "figures" / "dyck_counter_sparse_supervision_ablation"
BRACKET_SWEEP = [20, 24, 28, 32, 34, 36, 40, 44, 48, 56, 64, 80, 100, 200, 400]


def run_spec(bracket_tokens: int) -> dict:
    if bracket_tokens == 20:
        source = "tiny_extreme_long"
        run_dir = ROOT / "results" / "dyck_counter_task_a_tiny_extreme_long" / "transformer_seed0"
    elif bracket_tokens == 400:
        source = "extreme_long"
        run_dir = ROOT / "results" / "dyck_counter_task_a_extreme_long" / "transformer_seed0"
    else:
        source = f"sparse_len2000_b{bracket_tokens}"
        run_dir = ROOT / "results" / f"dyck_counter_task_a_sparse_len2000_b{bracket_tokens}" / "transformer_seed0"
    return {
        "setting": f"b{bracket_tokens}",
        "bracket_tokens": bracket_tokens,
        "source": source,
        "run_dir": run_dir,
    }


RUNS = [run_spec(bracket_tokens) for bracket_tokens in BRACKET_SWEEP]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    oracle_rows = []
    layerwise_rows = []
    for spec in RUNS:
        run_dir = Path(spec["run_dir"])
        required = [
            run_dir / "config.json",
            run_dir / "metrics.json",
            run_dir / "hidden_states" / "final" / "labels.parquet",
            run_dir / "probes" / "layerwise_probe.csv",
        ]
        if not all(path.exists() for path in required):
            missing = [path.name for path in required if not path.exists()]
            print(f"skip {spec['setting']}: missing {missing}")
            continue
        run_summary, run_oracle, run_layerwise = collect_run(spec)
        summary_rows.append(run_summary)
        oracle_rows.extend(run_oracle)
        layerwise_rows.append(run_layerwise)
        gc.collect()

    if not summary_rows:
        raise RuntimeError("No complete sparse-supervision runs found.")
    summary = pd.DataFrame(summary_rows).sort_values("bracket_tokens").reset_index(drop=True)
    oracle = pd.DataFrame(oracle_rows).sort_values(["bracket_tokens", "split"]).reset_index(drop=True)
    layerwise = pd.concat(layerwise_rows, ignore_index=True).sort_values(["bracket_tokens", "layer"]).reset_index(drop=True)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    oracle.to_csv(OUT_DIR / "oracle_forced_free.csv", index=False)
    layerwise.to_csv(OUT_DIR / "layerwise_probe.csv", index=False)
    fig_path = plot_sparse_ablation(summary, oracle, layerwise)
    print(f"wrote {OUT_DIR / 'summary.csv'}")
    print(f"wrote {OUT_DIR / 'oracle_forced_free.csv'}")
    print(f"wrote {OUT_DIR / 'layerwise_probe.csv'}")
    print(f"wrote {fig_path}")


def collect_run(spec: dict) -> tuple[dict, list[dict], pd.DataFrame]:
    run_dir = Path(spec["run_dir"])
    cfg = load_json(run_dir / "config.json")
    task = cfg["task"]
    metrics = load_json(run_dir / "metrics.json")
    eval_metrics = metrics.get("eval", {})
    layerwise = pd.read_csv(run_dir / "probes" / "layerwise_probe.csv")
    best = layerwise.sort_values("height_r2", ascending=False).iloc[0].to_dict()
    layerwise.insert(0, "source", spec["source"])
    layerwise.insert(0, "setting", spec["setting"])
    layerwise.insert(0, "bracket_tokens", spec["bracket_tokens"])

    row_ns = SimpleNamespace(
        setting=spec["setting"],
        run_dir_abs=run_dir,
        total_length=int(task["total_length"]),
        noise_vocab=int(task["num_noise_tokens"]),
    )
    labels = prepare_labels(load_labels(run_dir), row_ns)
    mask = labels["has_target_label"].to_numpy(dtype=bool) & labels["target_is_dyck_position"].to_numpy(dtype=bool)
    eval_idx = choose_indices(mask, max_rows=200_000, seed=setting_seed(spec["setting"], 80))
    target = labels.iloc[eval_idx]["target_token"].to_numpy(dtype=int)
    pred, logits = model_predictions(run_dir, int(task["num_noise_tokens"]), eval_idx)
    correct = pred == target
    eval_labels = labels.iloc[eval_idx].reset_index(drop=True)

    close_token = int(task["num_noise_tokens"])
    open_token = close_token + 1
    p_close_given_bracket = sigmoid(logits[:, close_token] - logits[:, open_token])
    oracle_rows = []
    split_results = {}
    for split_name, split_mask in split_masks(eval_labels).items():
        if split_mask.sum() == 0:
            continue
        model_acc = float(correct[split_mask].mean())
        oracle_acc = float(eval_labels.loc[split_mask, "oracle_next_dyck_acc"].mean())
        split_results[f"{split_name}_model_acc"] = model_acc
        split_results[f"{split_name}_oracle_acc"] = oracle_acc
        split_results[f"{split_name}_n"] = int(split_mask.sum())
        oracle_rows.append(
            {
                "setting": spec["setting"],
                "source": spec["source"],
                "bracket_tokens": spec["bracket_tokens"],
                "seq_len": int(task["seq_len"]),
                "repeat_prob": float(task["repeat_prob"]),
                "split": split_name,
                "n": int(split_mask.sum()),
                "model_acc": model_acc,
                "oracle_acc": oracle_acc,
                "gap_model_minus_oracle": model_acc - oracle_acc,
                "mean_height": float(eval_labels.loc[split_mask, "height"].mean()),
                "mean_p_close_given_bracket": float(p_close_given_bracket[split_mask].mean()),
            }
        )

    summary = {
        "setting": spec["setting"],
        "source": spec["source"],
        "run_dir": relpath(run_dir),
        "bracket_tokens": spec["bracket_tokens"],
        "bracket_density": spec["bracket_tokens"] / int(task["seq_len"]),
        "seq_len": int(task["seq_len"]),
        "repeat_prob": float(task["repeat_prob"]),
        "noise_vocab": int(task["num_noise_tokens"]),
        "train_eval_loss": float(eval_metrics.get("loss", np.nan)),
        "train_eval_accuracy": float(eval_metrics.get("accuracy", np.nan)),
        "train_eval_dyck_accuracy": float(eval_metrics.get("dyck_accuracy", np.nan)),
        "best_layer": int(best["layer"]),
        "height_r2": float(best["height_r2"]),
        "height_mae": float(best["height_mae"]),
        "left_r2": float(best.get("left_r2", np.nan)),
        "right_r2": float(best.get("right_r2", np.nan)),
        "legal_next_class_accuracy": float(best.get("legal_next_class_accuracy", np.nan)),
        **split_results,
    }
    del logits
    return summary, oracle_rows, layerwise


def plot_sparse_ablation(summary: pd.DataFrame, oracle: pd.DataFrame, layerwise: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.0), constrained_layout=True)
    x = summary["bracket_tokens"].to_numpy()

    axes[0].plot(x, summary["all_dyck_targets_model_acc"], marker="o", label="Dyck acc on extracted sample", color="#2563eb")
    axes[0].plot(x, summary["all_dyck_targets_oracle_acc"], marker="o", label="oracle", color="#64748b")
    axes[0].plot(x, summary["forced_model_acc"], marker="o", label="forced acc", color="#059669")
    axes[0].plot(x, summary["free_model_acc"], marker="o", label="free acc", color="#dc2626")
    axes[0].set_xscale("log")
    axes[0].set_ylim(0, 1.03)
    axes[0].set_xlabel("bracket tokens in seq_len=2000")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Behavior as Dyck supervision gets denser")
    axes[0].grid(alpha=0.22)
    axes[0].legend(fontsize=8)

    axes[1].plot(x, summary["height_r2"], marker="o", label="height R2", color="#7c3aed")
    axes[1].plot(x, summary["legal_next_class_accuracy"], marker="o", label="legal-next probe acc", color="#f97316")
    axes[1].set_xscale("log")
    axes[1].set_ylim(0, 1.03)
    axes[1].set_xlabel("bracket tokens in seq_len=2000")
    axes[1].set_ylabel("probe score")
    axes[1].set_title("Hidden counter remains readable")
    axes[1].grid(alpha=0.22)
    axes[1].legend(fontsize=8)

    heat = layerwise.pivot_table(index="setting", columns="layer", values="height_r2", aggfunc="mean").reindex(summary["setting"])
    im = axes[2].imshow(heat.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axes[2].set_xticks(np.arange(len(heat.columns)))
    axes[2].set_xticklabels([str(c) for c in heat.columns])
    axes[2].set_yticks(np.arange(len(heat.index)))
    axes[2].set_yticklabels([f"{row.bracket_tokens}" for row in summary.itertuples(index=False)])
    axes[2].set_xlabel("layer")
    axes[2].set_ylabel("bracket tokens")
    axes[2].set_title("Layerwise height R2")
    for yi in range(heat.shape[0]):
        for xi in range(heat.shape[1]):
            value = heat.iloc[yi, xi]
            axes[2].text(xi, yi, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.65 else "black", fontsize=8)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    out = FIG_DIR / "sparse_supervision_ablation.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def relpath(path: Path) -> str:
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


if __name__ == "__main__":
    main()
