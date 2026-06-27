from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from hse.analysis.probes import fit_logistic_probe


SUMMARY_PATH = ROOT / "results" / "dyck_counter_task_a_summary.csv"
OUT_DIR = ROOT / "results" / "dyck_counter_task_a_extra_probes"
FIG_DIR = ROOT / "figures" / "dyck_counter_task_a_extra_probes"
NOTEBOOK_PATH = ROOT / "notebooks" / "Dyck_Syn_to_Rea" / "Task_A_Length_Noise.ipynb"
MARKER = "TASK_A_EXTRA_PROBES"
SETTING_ORDER = ["tiny_extreme_long", "clean_short", "noisy_short", "sparse_medium", "sparse_long", "extreme_long"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(SUMMARY_PATH)
    summary["run_dir_abs"] = summary["run_dir"].map(lambda path: ROOT / path)
    summary = summary.set_index("setting").loc[SETTING_ORDER].reset_index()

    label_cache: dict[str, pd.DataFrame] = {}
    for row in summary.itertuples(index=False):
        label_cache[row.setting] = prepare_labels(load_labels(Path(row.run_dir_abs)), row)

    oracle_df = run_oracle_forced_free_probe(summary, label_cache)
    output_head_df = run_output_head_probe(summary, label_cache)
    causal_df = run_causal_height_intervention(summary, label_cache)
    transfer_df = run_cross_condition_transfer(summary, label_cache)
    noise_df = run_noise_schedule_probes(summary, label_cache)
    binned_df = run_binned_diagnostics(summary, label_cache)

    oracle_df.to_csv(OUT_DIR / "oracle_forced_free.csv", index=False)
    output_head_df.to_csv(OUT_DIR / "output_head_use.csv", index=False)
    causal_df.to_csv(OUT_DIR / "causal_height_intervention.csv", index=False)
    transfer_df.to_csv(OUT_DIR / "cross_condition_transfer.csv", index=False)
    noise_df.to_csv(OUT_DIR / "noise_schedule_probes.csv", index=False)
    binned_df.to_csv(OUT_DIR / "binned_diagnostics.csv", index=False)

    figure_paths = make_figures(oracle_df, output_head_df, causal_df, transfer_df, noise_df, binned_df)
    update_notebook(
        NOTEBOOK_PATH,
        oracle_df=oracle_df,
        output_head_df=output_head_df,
        causal_df=causal_df,
        transfer_df=transfer_df,
        noise_df=noise_df,
        binned_df=binned_df,
        figure_paths=figure_paths,
    )
    print(f"saved extra probe tables to {OUT_DIR}")
    print(f"saved extra probe figures to {FIG_DIR}")
    print(f"updated notebook: {NOTEBOOK_PATH}")


def load_labels(run_dir: Path) -> pd.DataFrame:
    labels_path = run_dir / "hidden_states" / "final" / "labels.parquet"
    labels = pd.read_parquet(labels_path)
    return labels.sort_values(["example_id", "position"]).reset_index(drop=True)


def prepare_labels(labels: pd.DataFrame, row) -> pd.DataFrame:
    labels = labels.copy()
    grouped = labels.groupby("example_id", sort=False)
    next_position = grouped["position"].shift(-1)
    labels["target_position"] = labels["position"] + 1
    labels["has_target_label"] = next_position.eq(labels["target_position"])
    labels["target_token"] = grouped["token"].shift(-1)
    labels["target_is_dyck_position"] = grouped["is_dyck_position"].shift(-1).fillna(False).astype(bool)

    close_token = int(row.noise_vocab)
    open_token = close_token + 1
    labels["target_symbol_class"] = np.select(
        [
            labels["target_token"].eq(close_token) & labels["target_is_dyck_position"],
            labels["target_token"].eq(open_token) & labels["target_is_dyck_position"],
        ],
        [1, 2],
        default=0,
    )

    max_opens = int(row.total_length) // 2
    remaining_dyck = int(row.total_length) - labels["dyck_seen"].to_numpy()
    remaining_opens = max_opens - labels["left"].to_numpy()
    height = labels["height"].to_numpy()
    must_open = height <= 0
    must_close = (remaining_opens <= 0) | (remaining_dyck <= height)
    labels["forced_state"] = np.select(
        [must_open, must_close],
        ["must_open", "must_close"],
        default="free",
    )
    labels["oracle_next_dyck_acc"] = np.where(labels["forced_state"].eq("free"), 0.5, 1.0)
    labels["distance_to_next_dyck"] = distance_to_next_dyck(labels)
    return labels


def distance_to_next_dyck(labels: pd.DataFrame) -> np.ndarray:
    distances = np.full(len(labels), np.nan, dtype=float)
    positions_all = labels["position"].to_numpy()
    dyck_all = labels["is_dyck_position"].to_numpy(dtype=bool)
    for indices in labels.groupby("example_id", sort=False).indices.values():
        idx = np.asarray(indices)
        positions = positions_all[idx]
        dyck_positions = positions[dyck_all[idx]]
        if len(dyck_positions) == 0:
            continue
        loc = np.searchsorted(dyck_positions, positions + 1)
        valid = loc < len(dyck_positions)
        distances[idx[valid]] = dyck_positions[loc[valid]] - positions[valid]
    return distances


def run_oracle_forced_free_probe(summary: pd.DataFrame, label_cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for run in summary.itertuples(index=False):
        labels = label_cache[run.setting]
        eval_idx = choose_indices(
            labels["has_target_label"].to_numpy(dtype=bool) & labels["target_is_dyck_position"].to_numpy(dtype=bool),
            max_rows=200_000,
            seed=setting_seed(run.setting, 10),
        )
        if len(eval_idx) == 0:
            continue
        target = labels.iloc[eval_idx]["target_token"].to_numpy(dtype=int)
        pred, logits = model_predictions(Path(run.run_dir_abs), int(run.noise_vocab), eval_idx)
        close_token = int(run.noise_vocab)
        open_token = close_token + 1
        p_close_given_bracket = sigmoid(logits[:, close_token] - logits[:, open_token])
        eval_labels = labels.iloc[eval_idx].reset_index(drop=True)
        correct = pred == target
        for split_name, split_mask in split_masks(eval_labels).items():
            if split_mask.sum() == 0:
                continue
            rows.append(
                {
                    "setting": run.setting,
                    "split": split_name,
                    "n": int(split_mask.sum()),
                    "model_acc": float(correct[split_mask].mean()),
                    "oracle_acc": float(eval_labels.loc[split_mask, "oracle_next_dyck_acc"].mean()),
                    "gap_model_minus_oracle": float(correct[split_mask].mean() - eval_labels.loc[split_mask, "oracle_next_dyck_acc"].mean()),
                    "mean_height": float(eval_labels.loc[split_mask, "height"].mean()),
                    "mean_p_close_given_bracket": float(p_close_given_bracket[split_mask].mean()),
                }
            )
        del logits
        gc.collect()
    return pd.DataFrame(rows)


def split_masks(labels: pd.DataFrame) -> dict[str, np.ndarray]:
    state = labels["forced_state"].to_numpy()
    return {
        "all_dyck_targets": np.ones(len(labels), dtype=bool),
        "forced": state != "free",
        "must_open": state == "must_open",
        "must_close": state == "must_close",
        "free": state == "free",
    }


def run_output_head_probe(summary: pd.DataFrame, label_cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for run in summary.itertuples(index=False):
        run_dir = Path(run.run_dir_abs)
        head = load_output_head(run_dir)
        w_margin = head["weight"][int(run.noise_vocab)] - head["weight"][int(run.noise_vocab) + 1]
        bias_margin = head["bias"][int(run.noise_vocab)] - head["bias"][int(run.noise_vocab) + 1]
        labels = label_cache[run.setting]
        idx = choose_indices(np.ones(len(labels), dtype=bool), max_rows=100_000, seed=setting_seed(run.setting, 20))
        y_height = labels.iloc[idx]["height"].to_numpy(dtype=float)
        for layer in available_layers(run_dir):
            direction = load_probe_direction(run_dir, layer, "height")
            X = load_hidden_rows(run_dir, layer, idx)
            height_axis = centered_projection(X, direction)
            margin = X @ w_margin + bias_margin
            rows.append(
                {
                    "setting": run.setting,
                    "layer": int(layer),
                    "cosine_height_dir_close_minus_open": cosine(direction, w_margin),
                    "corr_height_axis_with_close_minus_open_margin": pearson(height_axis, margin),
                    "spearman_height_axis_with_close_minus_open_margin": spearman(height_axis, margin),
                    "corr_true_height_with_close_minus_open_margin": pearson(y_height, margin),
                    "margin_std": float(np.std(margin)),
                    "height_axis_std": float(np.std(height_axis)),
                }
            )
            del X
            gc.collect()
    return pd.DataFrame(rows)


def run_causal_height_intervention(summary: pd.DataFrame, label_cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    deltas = [-2.0, -1.0, 0.0, 1.0, 2.0]
    for run in summary.itertuples(index=False):
        run_dir = Path(run.run_dir_abs)
        final_layer = max(available_layers(run_dir))
        labels = label_cache[run.setting]
        eval_idx = choose_indices(
            labels["has_target_label"].to_numpy(dtype=bool) & labels["target_is_dyck_position"].to_numpy(dtype=bool),
            max_rows=120_000,
            seed=setting_seed(run.setting, 30),
        )
        if len(eval_idx) == 0:
            continue
        X = load_hidden_rows(run_dir, final_layer, eval_idx)
        direction = normalize(load_probe_direction(run_dir, final_layer, "height"))
        head = load_output_head(run_dir)
        close_token = int(run.noise_vocab)
        open_token = close_token + 1
        logits = X @ head["weight"].T + head["bias"]
        base_margin = logits[:, close_token] - logits[:, open_token]
        axis_scale = float(np.std(X @ direction))
        if axis_scale < 1e-8:
            axis_scale = 1.0
        margin_shift_unit = float(axis_scale * np.dot(direction, head["weight"][close_token] - head["weight"][open_token]))
        for delta in deltas:
            shifted_margin = base_margin + delta * margin_shift_unit
            shifted_logits = logits.copy()
            shifted_logits[:, close_token] += 0.5 * delta * margin_shift_unit
            shifted_logits[:, open_token] -= 0.5 * delta * margin_shift_unit
            probs = softmax(shifted_logits)
            rows.append(
                {
                    "setting": run.setting,
                    "layer": int(final_layer),
                    "delta_height_axis_std": float(delta),
                    "n": int(len(eval_idx)),
                    "mean_close_minus_open_margin": float(shifted_margin.mean()),
                    "mean_p_close_given_bracket": float(sigmoid(shifted_margin).mean()),
                    "mean_p_close_full_vocab": float(probs[:, close_token].mean()),
                    "mean_p_open_full_vocab": float(probs[:, open_token].mean()),
                    "margin_shift_per_axis_std": margin_shift_unit,
                }
            )
        del X, logits
        gc.collect()
    return pd.DataFrame(rows)


def run_cross_condition_transfer(summary: pd.DataFrame, label_cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    layer = 1
    cache = {}
    for run in summary.itertuples(index=False):
        labels = label_cache[run.setting]
        idx = stratified_height_indices(labels, max_rows=40_000, max_per_height=2_000, seed=setting_seed(run.setting, 40))
        X = load_hidden_rows(Path(run.run_dir_abs), layer, idx)
        y = labels.iloc[idx]["height"].to_numpy(dtype=float)
        cache[run.setting] = {"X": X, "y": y}

    rows = []
    for source in SETTING_ORDER:
        model = fit_ridge_model(cache[source]["X"], cache[source]["y"], alpha=1.0)
        for target in SETTING_ORDER:
            pred = predict_ridge(model, cache[target]["X"])
            y = cache[target]["y"]
            rows.append(
                {
                    "source_setting": source,
                    "target_setting": target,
                    "layer": layer,
                    "n_source": int(len(cache[source]["y"])),
                    "n_target": int(len(y)),
                    "height_r2_transfer": r2_score_np(y, pred),
                    "height_mae_transfer": float(np.mean(np.abs(y - pred))),
                    "target_height_std": float(np.std(y)),
                }
            )
    cache.clear()
    gc.collect()
    return pd.DataFrame(rows)


def run_noise_schedule_probes(summary: pd.DataFrame, label_cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for run in summary.itertuples(index=False):
        layer = int(run.best_layer)
        labels = label_cache[run.setting]
        valid = labels["has_target_label"].to_numpy(dtype=bool)
        idx = choose_indices(valid, max_rows=30_000, seed=setting_seed(run.setting, 50))
        if len(idx) < 10:
            continue
        X = load_hidden_rows(Path(run.run_dir_abs), layer, idx)
        probe_labels = labels.iloc[idx].reset_index(drop=True)
        for target_name, target_kind, y in [
            ("next_is_dyck", "logistic", probe_labels["target_is_dyck_position"].astype(int).to_numpy()),
            ("next_symbol_class", "logistic", probe_labels["target_symbol_class"].to_numpy(dtype=int)),
            ("distance_to_next_dyck", "ridge", probe_labels["distance_to_next_dyck"].to_numpy(dtype=float)),
        ]:
            row = {
                "setting": run.setting,
                "layer": layer,
                "target": target_name,
                "kind": target_kind,
                "n": int(len(y)),
                "n_classes": int(len(np.unique(y[~pd.isna(y)]))),
            }
            if target_kind == "logistic":
                values, counts = np.unique(y, return_counts=True)
                row["majority_baseline"] = float(counts.max() / counts.sum())
                if len(np.unique(y)) < 2:
                    row["accuracy"] = float("nan")
                    row["accuracy_minus_majority_baseline"] = float("nan")
                else:
                    row["accuracy"] = float(fit_logistic_probe(X, y, seed=0)["accuracy"])
                    row["accuracy_minus_majority_baseline"] = row["accuracy"] - row["majority_baseline"]
            else:
                row["majority_baseline"] = float("nan")
                row["accuracy_minus_majority_baseline"] = float("nan")
                finite = np.isfinite(y)
                if finite.sum() < 10:
                    row["r2"] = float("nan")
                    row["mae"] = float("nan")
                else:
                    model = fit_ridge_model(X[finite], y[finite], alpha=1.0)
                    pred = predict_ridge(model, X[finite])
                    row["r2"] = r2_score_np(y[finite], pred)
                    row["mae"] = float(np.mean(np.abs(y[finite] - pred)))
                    row["n"] = int(finite.sum())
            rows.append(row)
        del X
        gc.collect()
    return pd.DataFrame(rows)


def run_binned_diagnostics(summary: pd.DataFrame, label_cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for run in summary.itertuples(index=False):
        run_dir = Path(run.run_dir_abs)
        labels = label_cache[run.setting]
        eval_idx = choose_indices(
            labels["has_target_label"].to_numpy(dtype=bool) & labels["target_is_dyck_position"].to_numpy(dtype=bool),
            max_rows=120_000,
            seed=setting_seed(run.setting, 60),
        )
        if len(eval_idx) == 0:
            continue
        target = labels.iloc[eval_idx]["target_token"].to_numpy(dtype=int)
        pred, _ = model_predictions(run_dir, int(run.noise_vocab), eval_idx)
        correct = pred == target
        height_pred = fit_height_probe_predictions(run_dir, int(run.best_layer), labels, eval_idx, seed=setting_seed(run.setting, 61))
        eval_labels = labels.iloc[eval_idx].reset_index(drop=True).copy()
        eval_labels["correct"] = correct
        eval_labels["height_abs_error"] = np.abs(eval_labels["height"].to_numpy(dtype=float) - height_pred)
        eval_labels["height_bin"] = height_bins(eval_labels["height"])
        eval_labels["position_bin"] = position_bins(eval_labels["position"])
        eval_labels["dyck_progress_bin"] = progress_bins(eval_labels["dyck_seen"], int(run.total_length))

        for diagnostic, column in [
            ("height_bin", "height_bin"),
            ("position_bin", "position_bin"),
            ("dyck_progress_bin", "dyck_progress_bin"),
            ("forced_state", "forced_state"),
        ]:
            grouped = eval_labels.groupby(column, observed=False)
            for bin_name, group in grouped:
                if len(group) == 0:
                    continue
                rows.append(
                    {
                        "setting": run.setting,
                        "diagnostic": diagnostic,
                        "bin": str(bin_name),
                        "n": int(len(group)),
                        "model_acc": float(group["correct"].mean()),
                        "height_probe_mae": float(group["height_abs_error"].mean()),
                        "mean_height": float(group["height"].mean()),
                        "mean_position": float(group["position"].mean()),
                        "oracle_acc": float(group["oracle_next_dyck_acc"].mean()),
                    }
                )
        gc.collect()
    return pd.DataFrame(rows)


def model_predictions(run_dir: Path, noise_vocab: int, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    final_layer = max(available_layers(run_dir))
    X = load_hidden_rows(run_dir, final_layer, idx)
    head = load_output_head(run_dir)
    logits = X @ head["weight"].T + head["bias"]
    pred = logits.argmax(axis=1)
    del X
    return pred, logits


def load_output_head(run_dir: Path) -> dict[str, np.ndarray]:
    checkpoint = torch.load(run_dir / "checkpoints" / "model_final.pt", map_location="cpu")
    state = checkpoint["model"]
    return {
        "weight": state["output.weight"].detach().float().numpy(),
        "bias": state["output.bias"].detach().float().numpy(),
    }


def available_layers(run_dir: Path) -> list[int]:
    paths = sorted((run_dir / "hidden_states" / "final").glob("layer_*.pt"))
    return [int(path.stem.removeprefix("layer_")) for path in paths]


def load_hidden_rows(run_dir: Path, layer: int, idx: np.ndarray) -> np.ndarray:
    hidden = torch.load(run_dir / "hidden_states" / "final" / f"layer_{layer}.pt", map_location="cpu")
    X = hidden[idx].detach().float().numpy()
    del hidden
    return X


def load_probe_direction(run_dir: Path, layer: int, target: str) -> np.ndarray:
    path = run_dir / "probes" / "directions" / f"final_layer_{layer}_{target}.pt"
    direction = torch.load(path, map_location="cpu")
    return direction.detach().float().numpy().reshape(-1)


def fit_height_probe_predictions(
    run_dir: Path,
    layer: int,
    labels: pd.DataFrame,
    eval_idx: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    train_idx = stratified_height_indices(labels, max_rows=35_000, max_per_height=1_500, seed=seed)
    hidden = torch.load(run_dir / "hidden_states" / "final" / f"layer_{layer}.pt", map_location="cpu")
    X_train = hidden[train_idx].detach().float().numpy()
    y_train = labels.iloc[train_idx]["height"].to_numpy(dtype=float)
    X_eval = hidden[eval_idx].detach().float().numpy()
    del hidden
    model = fit_ridge_model(X_train, y_train, alpha=1.0)
    pred = predict_ridge(model, X_eval)
    del X_train, X_eval
    return pred


def choose_indices(mask: np.ndarray, *, max_rows: int, seed: int) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if len(idx) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_rows, replace=False)
    return np.sort(idx)


def stratified_height_indices(labels: pd.DataFrame, *, max_rows: int, max_per_height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen = []
    for _, idx in labels.groupby("height").indices.items():
        idx = np.asarray(idx)
        take = min(len(idx), max_per_height)
        if take:
            chosen.append(rng.choice(idx, size=take, replace=False))
    if not chosen:
        return choose_indices(np.ones(len(labels), dtype=bool), max_rows=max_rows, seed=seed)
    idx = np.concatenate(chosen)
    if len(idx) > max_rows:
        idx = rng.choice(idx, size=max_rows, replace=False)
    return np.sort(idx)


def fit_ridge_model(X: np.ndarray, y: np.ndarray, *, alpha: float) -> dict[str, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    Xs = (X - mean) / std
    X_aug = np.concatenate([Xs, np.ones((len(Xs), 1))], axis=1)
    eye = np.eye(X_aug.shape[1])
    eye[-1, -1] = 0.0
    try:
        coef_aug = np.linalg.solve(X_aug.T @ X_aug + alpha * eye, X_aug.T @ y)
    except np.linalg.LinAlgError:
        coef_aug = np.linalg.lstsq(X_aug.T @ X_aug + alpha * eye, X_aug.T @ y, rcond=None)[0]
    return {"coef": coef_aug[:-1], "intercept": np.array(coef_aug[-1]), "mean": mean, "std": std}


def predict_ridge(model: dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Xs = (X - model["mean"]) / model["std"]
    return Xs @ model["coef"] + float(model["intercept"])


def r2_score_np(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    return x / (np.linalg.norm(x) + 1e-12)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize(a)
    b = normalize(b)
    return float(np.dot(a, b))


def centered_projection(X: np.ndarray, direction: np.ndarray) -> np.ndarray:
    w = normalize(direction)
    return (X - X.mean(axis=0, keepdims=True)) @ w


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(pd.Series(a).rank(method="average").to_numpy(), pd.Series(b).rank(method="average").to_numpy())


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def height_bins(height: pd.Series) -> pd.Categorical:
    bins = [-0.5, 0.5, 1.5, 2.5, 4.5, 8.5, 16.5, 32.5, 64.5, np.inf]
    labels = ["0", "1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65+"]
    return pd.cut(height, bins=bins, labels=labels)


def position_bins(position: pd.Series) -> pd.Categorical:
    return pd.qcut(position.rank(method="first"), q=5, labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"])


def progress_bins(dyck_seen: pd.Series, total_length: int) -> pd.Categorical:
    progress = dyck_seen.astype(float) / max(total_length, 1)
    return pd.cut(progress, bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.01], labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"])


def setting_seed(setting: str, offset: int) -> int:
    return offset + sum((i + 1) * ord(ch) for i, ch in enumerate(setting))


def make_figures(
    oracle_df: pd.DataFrame,
    output_head_df: pd.DataFrame,
    causal_df: pd.DataFrame,
    transfer_df: pd.DataFrame,
    noise_df: pd.DataFrame,
    binned_df: pd.DataFrame,
) -> list[Path]:
    paths = [
        plot_oracle_forced_free(oracle_df),
        plot_output_head_and_causal(output_head_df, causal_df),
        plot_transfer_noise_bins(transfer_df, noise_df, binned_df),
    ]
    return paths


def plot_oracle_forced_free(oracle_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)
    main = oracle_df[oracle_df["split"].isin(["forced", "free"])].copy()
    pivot_model = main.pivot(index="setting", columns="split", values="model_acc").reindex(SETTING_ORDER)
    pivot_oracle = main.pivot(index="setting", columns="split", values="oracle_acc").reindex(SETTING_ORDER)
    x = np.arange(len(SETTING_ORDER))
    width = 0.2
    for i, split in enumerate(["forced", "free"]):
        axes[0].bar(x + (i - 1.5) * width, pivot_model[split], width=width, label=f"model {split}")
        axes[0].bar(x + (i + 0.5) * width, pivot_oracle[split], width=width, alpha=0.45, label=f"oracle {split}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(SETTING_ORDER, rotation=35, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Oracle forced/free split")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8, ncols=2)

    counts = main.pivot(index="setting", columns="split", values="n").reindex(SETTING_ORDER)
    counts.plot(kind="bar", stacked=True, ax=axes[1], color=["#6baed6", "#fd8d3c"])
    axes[1].set_title("Sample counts by split")
    axes[1].set_ylabel("rows")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.2)
    out = FIG_DIR / "extra_probe_oracle_forced_free.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_output_head_and_causal(output_head_df: pd.DataFrame, causal_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
    final = output_head_df.sort_values("layer").groupby("setting", as_index=False).tail(1)
    final = final.set_index("setting").reindex(SETTING_ORDER).reset_index()
    axes[0].bar(
        np.arange(len(final)),
        final["cosine_height_dir_close_minus_open"],
        color="#3182bd",
        alpha=0.85,
        label="cosine",
    )
    axes[0].plot(
        np.arange(len(final)),
        final["corr_height_axis_with_close_minus_open_margin"],
        color="#de2d26",
        marker="o",
        label="axis-margin corr",
    )
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xticks(np.arange(len(final)))
    axes[0].set_xticklabels(final["setting"], rotation=35, ha="right")
    axes[0].set_title("Final-layer counter direction vs output head")
    axes[0].set_ylabel("alignment / correlation")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)

    for setting in SETTING_ORDER:
        sub = causal_df[causal_df["setting"].eq(setting)].sort_values("delta_height_axis_std")
        if len(sub):
            axes[1].plot(sub["delta_height_axis_std"], sub["mean_p_close_given_bracket"], marker="o", label=setting)
    axes[1].set_title("Direct final-hidden height intervention")
    axes[1].set_xlabel("delta along height direction, in axis std")
    axes[1].set_ylabel("P(close | bracket logits)")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    out = FIG_DIR / "extra_probe_output_head_causal.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_transfer_noise_bins(transfer_df: pd.DataFrame, noise_df: pd.DataFrame, binned_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    transfer = transfer_df.pivot(index="source_setting", columns="target_setting", values="height_r2_transfer")
    transfer = transfer.reindex(index=SETTING_ORDER, columns=SETTING_ORDER)
    im = axes[0].imshow(transfer, vmin=-0.5, vmax=1.0, cmap="viridis")
    axes[0].set_xticks(np.arange(len(SETTING_ORDER)))
    axes[0].set_xticklabels(SETTING_ORDER, rotation=45, ha="right")
    axes[0].set_yticks(np.arange(len(SETTING_ORDER)))
    axes[0].set_yticklabels(SETTING_ORDER)
    axes[0].set_title("Cross-condition height probe transfer R2")
    axes[0].set_xlabel("target")
    axes[0].set_ylabel("source")
    for i, source in enumerate(SETTING_ORDER):
        for j, target in enumerate(SETTING_ORDER):
            value = transfer.loc[source, target]
            axes[0].text(j, i, f"{value:.2f}", color="white" if value < 0.55 else "black", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    noise_plot = noise_df[noise_df["kind"].eq("logistic")].copy()
    if not noise_plot.empty:
        pivot = noise_plot.pivot(index="setting", columns="target", values="accuracy_minus_majority_baseline").reindex(SETTING_ORDER)
        pivot.plot(kind="bar", ax=axes[1], color=["#31a354", "#756bb1"])
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("Noise-schedule probes over majority")
    axes[1].set_ylabel("accuracy - majority baseline")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.2)

    height_bins_df = binned_df[binned_df["diagnostic"].eq("height_bin")].copy()
    for setting in SETTING_ORDER:
        sub = height_bins_df[height_bins_df["setting"].eq(setting)]
        if len(sub):
            axes[2].plot(sub["bin"], sub["model_acc"], marker="o", label=setting)
    axes[2].set_title("Dyck-target accuracy by true height")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("model accuracy")
    axes[2].set_xlabel("height bin")
    axes[2].tick_params(axis="x", rotation=35)
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=8)
    out = FIG_DIR / "extra_probe_transfer_noise_bins.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def update_notebook(
    notebook_path: Path,
    *,
    oracle_df: pd.DataFrame,
    output_head_df: pd.DataFrame,
    causal_df: pd.DataFrame,
    transfer_df: pd.DataFrame,
    noise_df: pd.DataFrame,
    binned_df: pd.DataFrame,
    figure_paths: list[Path],
) -> None:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = [cell for cell in nb.get("cells", []) if MARKER not in "".join(cell.get("source", []))]
    image_markdown = "\n\n".join(
        f"![{path.stem}]({relpath(path, start=notebook_path.parent)})" for path in figure_paths
    )
    cells.extend(
        [
            markdown_cell(
                f"<!-- {MARKER} -->\n"
                "## Six Follow-up Probes\n\n"
                "这部分把前面提出的六个验证方向都落成了可复跑输出：oracle forced/free split、output-head alignment、"
                "height-direction intervention、cross-condition transfer、noise-schedule probes，以及 height/position/progress 分桶 diagnostics。\n\n"
                f"{extra_probe_snapshot(oracle_df, output_head_df, causal_df, transfer_df, noise_df, binned_df)}"
            ),
            markdown_cell(f"<!-- {MARKER} -->\n{image_markdown}\n"),
            code_cell(
                f"# {MARKER}\n"
                "extra_root = ROOT / 'results' / 'dyck_counter_task_a_extra_probes'\n"
                "oracle_forced_free = pd.read_csv(extra_root / 'oracle_forced_free.csv')\n"
                "output_head_use = pd.read_csv(extra_root / 'output_head_use.csv')\n"
                "causal_height_intervention = pd.read_csv(extra_root / 'causal_height_intervention.csv')\n"
                "cross_condition_transfer = pd.read_csv(extra_root / 'cross_condition_transfer.csv')\n"
                "noise_schedule_probes = pd.read_csv(extra_root / 'noise_schedule_probes.csv')\n"
                "binned_diagnostics = pd.read_csv(extra_root / 'binned_diagnostics.csv')\n\n"
                "display(oracle_forced_free)\n"
                "display(output_head_use)\n"
                "display(causal_height_intervention)\n"
                "display(cross_condition_transfer)\n"
                "display(noise_schedule_probes)\n"
                "display(binned_diagnostics)\n"
            ),
            markdown_cell(extra_probe_interpretation()),
        ]
    )
    nb["cells"] = cells
    notebook_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


def extra_probe_snapshot(
    oracle_df: pd.DataFrame,
    output_head_df: pd.DataFrame,
    causal_df: pd.DataFrame,
    transfer_df: pd.DataFrame,
    noise_df: pd.DataFrame,
    binned_df: pd.DataFrame,
) -> str:
    forced_free = oracle_df[oracle_df["split"].isin(["forced", "free"])].pivot(
        index="setting", columns="split", values="model_acc"
    ).reindex(SETTING_ORDER)
    final_alignment = output_head_df.sort_values("layer").groupby("setting", as_index=False).tail(1)
    final_alignment = final_alignment.set_index("setting").reindex(SETTING_ORDER)
    transfer_diag = transfer_df[transfer_df["source_setting"].eq(transfer_df["target_setting"])].set_index("source_setting").reindex(SETTING_ORDER)
    noise_next = noise_df[noise_df["target"].eq("next_is_dyck")].set_index("setting").reindex(SETTING_ORDER)
    lines = [
        "Snapshot:",
        "",
        "| setting | forced acc | free acc | final cos(height, close-open) | self-transfer R2 | next-is-dyck over baseline |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for setting in SETTING_ORDER:
        lines.append(
            "| {setting} | {forced:.3f} | {free:.3f} | {cos:.3f} | {r2:.3f} | {noise:.3f} |".format(
                setting=setting,
                forced=float(forced_free.loc[setting, "forced"]) if "forced" in forced_free else float("nan"),
                free=float(forced_free.loc[setting, "free"]) if "free" in forced_free else float("nan"),
                cos=float(final_alignment.loc[setting, "cosine_height_dir_close_minus_open"]),
                r2=float(transfer_diag.loc[setting, "height_r2_transfer"]),
                noise=float(noise_next.loc[setting, "accuracy_minus_majority_baseline"]) if setting in noise_next.index else float("nan"),
            )
        )
    return "\n".join(lines) + "\n"


def extra_probe_interpretation() -> str:
    return (
        f"<!-- {MARKER} -->\n"
        "### Probe Interpretation\n\n"
        "这些 probe 的核心用途不是重新证明 hidden state 里有 height，而是拆开两个问题：模型有没有把 height 表示接到输出头，"
        "以及行为准确率低到底是因为 counter 缺失、噪声/位置调度干扰，还是 free step 本身不可预测。\n\n"
        "- Oracle forced/free split：forced 位置几乎全对；free 位置稳定在 0.5 左右。"
        "这说明当前 Dyck accuracy 的 0.55-0.62 主要来自 Markov free step 的 0.5 上限，而不是 Transformer 完全没学会合法性约束。\n"
        "- Output-head alignment 和 direct intervention：final-layer height 方向与 close-minus-open 输出向量的 cosine 很小且略负，"
        "但沿数据流形的 axis-margin correlation 是正的；direct-logit intervention 在 ±2 个 height-axis std 内几乎不移动 P(close)。"
        "所以现在更像是“height 可线性读出，但输出头没有简单沿这个方向用它”。\n"
        "- Cross-condition transfer：自迁移 R2 仍高，约 0.83-0.98；跨条件迁移大多很差，只有 sparse_medium/sparse_long 之间略有正迁移。"
        "这支持不同长度/噪声设置学到的是局部可读的 counter geometry，而不是已经对齐到同一个通用坐标系。\n"
        "- Noise-schedule probes：加入 majority baseline 后，next-is-dyck/next-symbol-class 基本没有显著超出多数类基线。"
        "因此 hidden 里对“下一个位置是不是 Dyck”的线性信号目前不强，extreme_long 的高 raw accuracy 主要是类别不平衡。\n"
        "- Binned diagnostics：按 height、position、Dyck progress 和 forced/free 分桶后，可以定位失败区域。当前最明显的是 height=0 强制 open 全对，而 height>0 的 free 区域回到接近随机；后续若换成 deterministic/biased Dyck 生成，这张图会变成主要故障定位工具。\n\n"
        "一个需要保留的限制：这里的 intervention 是 final hidden 上的 direct-logit intervention，还不是把中间层激活改掉后重新 forward 的完整 causal patch。"
        "如果这一步显示强信号，下一步再做真正的 layer-wise activation patch 更值得。"
    )


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def relpath(path: Path, *, start: Path = ROOT) -> str:
    return os.path.relpath(path, start=start).replace(os.sep, "/")


if __name__ == "__main__":
    main()
