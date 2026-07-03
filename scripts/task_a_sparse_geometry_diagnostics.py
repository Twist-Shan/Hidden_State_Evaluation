from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from types import SimpleNamespace

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
import torch

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


SPARSE_SUMMARY = ROOT / "results" / "dyck_counter_sparse_supervision_ablation" / "summary.csv"
OUT_DIR = ROOT / "results" / "dyck_counter_task_a_sparse_geometry"
FIG_DIR = ROOT / "figures" / "dyck_counter_task_a_sparse_geometry"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", nargs="*", default=None)
    parser.add_argument("--max-rows", type=int, default=160_000)
    parser.add_argument("--from-existing", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_existing:
        token_loss = pd.read_csv(OUT_DIR / "sparse_token_loss.csv")
        geometry = pd.read_csv(OUT_DIR / "direction_geometry.csv")
        stability = pd.read_csv(OUT_DIR / "direction_stability.csv")
        ablation = pd.read_csv(OUT_DIR / "direction_ablation.csv")
    else:
        sparse = pd.read_csv(SPARSE_SUMMARY)
        if args.settings:
            sparse = sparse[sparse["setting"].isin(args.settings)].copy()
        token_rows = []
        geometry_rows = []
        ablation_rows = []
        direction_cache: dict[tuple[str, int, str], np.ndarray] = {}

        for run in sparse.sort_values("bracket_tokens").itertuples(index=False):
            print(f"sparse diagnostics {run.setting}: brackets={run.bracket_tokens}")
            result = collect_run(run, max_rows=args.max_rows)
            token_rows.extend(result["token_rows"])
            geometry_rows.extend(result["geometry_rows"])
            ablation_rows.extend(result["ablation_rows"])
            direction_cache.update(result["directions"])
            gc.collect()

        token_loss = pd.DataFrame(token_rows)
        geometry = pd.DataFrame(geometry_rows)
        ablation = add_ablation_deltas(pd.DataFrame(ablation_rows))
        stability = compute_stability(direction_cache, sparse)
        token_loss.to_csv(OUT_DIR / "sparse_token_loss.csv", index=False)
        geometry.to_csv(OUT_DIR / "direction_geometry.csv", index=False)
        stability.to_csv(OUT_DIR / "direction_stability.csv", index=False)
        ablation.to_csv(OUT_DIR / "direction_ablation.csv", index=False)

    make_figures(token_loss, geometry, stability, ablation)
    write_readme(token_loss, geometry, stability, ablation)
    print(f"wrote tables to {OUT_DIR}")
    print(f"wrote figures to {FIG_DIR}")


def collect_run(run, *, max_rows: int) -> dict[str, object]:
    run_dir = ROOT / str(run.run_dir)
    spec = SimpleNamespace(
        setting=run.setting,
        run_dir_abs=run_dir,
        total_length=int(run.bracket_tokens),
        noise_vocab=int(run.noise_vocab),
    )
    labels = prepare_labels(load_labels(run_dir), spec)
    layers = available_layers(run_dir)
    final_layer = max(layers)
    close_token = int(run.noise_vocab)
    open_token = close_token + 1
    output = load_output_head(run_dir)
    output_vectors = load_static_vectors(run_dir, output, close_token, open_token)

    direction_cache: dict[tuple[str, int, str], np.ndarray] = {}
    geometry_rows = []
    final_bank = None
    for layer in layers:
        bank = layer_direction_bank(
            run_dir,
            layer,
            labels,
            output_vectors,
            close_token=close_token,
            open_token=open_token,
            seed=setting_seed(run.setting, 100 + layer),
        )
        if layer == final_layer:
            final_bank = bank
        for name in ["height", "left", "right", "current_bracket", "forced_next"]:
            direction_cache[(run.setting, int(layer), name)] = bank[name]
        geometry_rows.extend(geometry_for_bank(run, layer, bank))

    if final_bank is None:
        raise RuntimeError(f"No final bank for {run.setting}")

    eval_mask = labels["has_target_label"].to_numpy(dtype=bool) & labels["target_is_dyck_position"].to_numpy(dtype=bool)
    eval_idx = choose_indices(eval_mask, max_rows=max_rows, seed=setting_seed(run.setting, 120))
    eval_labels = labels.iloc[eval_idx].reset_index(drop=True)
    target = eval_labels["target_token"].to_numpy(dtype=int)
    X = load_hidden_rows(run_dir, final_layer, eval_idx)

    token_rows = token_metric_rows(
        run,
        eval_labels,
        target,
        logits=X @ output["weight"].T + output["bias"],
        close_token=close_token,
        open_token=open_token,
        evaluation="baseline",
        ablation="baseline",
    )
    ablation_rows = direction_ablation_rows(run, X, eval_labels, target, output, final_bank, close_token, open_token)
    del X
    return {
        "token_rows": token_rows,
        "geometry_rows": geometry_rows,
        "ablation_rows": ablation_rows,
        "directions": direction_cache,
    }


def load_static_vectors(run_dir: Path, output: dict[str, np.ndarray], close_token: int, open_token: int) -> dict[str, np.ndarray]:
    checkpoint = torch.load(run_dir / "checkpoints" / "model_final.pt", map_location="cpu")
    state = checkpoint["model"]
    embed = state["embed.weight"].detach().float().numpy()
    weight = output["weight"]
    return {
        "output_close_open": normalize(weight[close_token] - weight[open_token]),
        "output_bracket_noise": normalize(weight[[close_token, open_token]].mean(axis=0) - weight[:close_token].mean(axis=0)),
        "input_close_open": normalize(embed[close_token] - embed[open_token]),
        "input_bracket_noise": normalize(embed[[close_token, open_token]].mean(axis=0) - embed[:close_token].mean(axis=0)),
    }


def layer_direction_bank(
    run_dir: Path,
    layer: int,
    labels: pd.DataFrame,
    static: dict[str, np.ndarray],
    *,
    close_token: int,
    open_token: int,
    seed: int,
) -> dict[str, np.ndarray]:
    hidden_axes = mean_axes_for_layer(
        run_dir,
        layer,
        labels,
        close_token=close_token,
        open_token=open_token,
        seed=seed,
    )
    height = normalize(load_probe_direction(run_dir, layer, "height"))
    left = normalize(load_probe_direction(run_dir, layer, "left"))
    right = normalize(load_probe_direction(run_dir, layer, "right"))
    bank = {
        **static,
        "height": height,
        "left": left,
        "right": right,
        "current_bracket": hidden_axes["current_bracket"],
        "forced_next": hidden_axes["forced_next"],
    }
    bank["left_right"] = orthonormal_basis([left, right])
    bank["height_left_right"] = orthonormal_basis([height, left, right])
    bank["output_2d"] = orthonormal_basis([static["output_close_open"], static["output_bracket_noise"]])
    return bank


def mean_axes_for_layer(
    run_dir: Path,
    layer: int,
    labels: pd.DataFrame,
    *,
    close_token: int,
    open_token: int,
    seed: int,
    max_rows_per_class: int = 50_000,
) -> dict[str, np.ndarray]:
    current_pos = labels["is_dyck_position"].to_numpy(dtype=bool) & labels["token"].eq(close_token).to_numpy(dtype=bool)
    current_neg = labels["is_dyck_position"].to_numpy(dtype=bool) & labels["token"].eq(open_token).to_numpy(dtype=bool)
    forced_base = (
        labels["has_target_label"].to_numpy(dtype=bool)
        & labels["target_is_dyck_position"].to_numpy(dtype=bool)
        & labels["forced_state"].ne("free").to_numpy(dtype=bool)
    )
    forced_pos = forced_base & labels["target_token"].eq(close_token).to_numpy(dtype=bool)
    forced_neg = forced_base & labels["target_token"].eq(open_token).to_numpy(dtype=bool)
    idx_by_name = {
        "current_pos": choose_indices(current_pos, max_rows=max_rows_per_class, seed=seed),
        "current_neg": choose_indices(current_neg, max_rows=max_rows_per_class, seed=seed + 1),
        "forced_pos": choose_indices(forced_pos, max_rows=max_rows_per_class, seed=seed + 2),
        "forced_neg": choose_indices(forced_neg, max_rows=max_rows_per_class, seed=seed + 3),
    }
    all_idx = np.unique(np.concatenate([idx for idx in idx_by_name.values() if len(idx)]))
    if len(all_idx) == 0:
        raise RuntimeError(f"No hidden rows for {run_dir} layer {layer}")
    X = load_hidden_rows(run_dir, layer, all_idx)

    def mean_for(name: str) -> np.ndarray:
        idx = idx_by_name[name]
        if len(idx) == 0:
            return np.zeros(X.shape[1], dtype=float)
        loc = np.searchsorted(all_idx, idx)
        return X[loc].mean(axis=0)

    current_axis = normalize(mean_for("current_pos") - mean_for("current_neg"))
    forced_axis = normalize(mean_for("forced_pos") - mean_for("forced_neg"))
    del X
    return {
        "current_bracket": current_axis,
        "forced_next": forced_axis,
    }


def geometry_for_bank(run, layer: int, bank: dict[str, np.ndarray]) -> list[dict[str, object]]:
    pairs = [
        ("height", "output_close_open"),
        ("height", "output_bracket_noise"),
        ("height", "input_close_open"),
        ("left", "output_close_open"),
        ("right", "output_close_open"),
        ("current_bracket", "output_close_open"),
        ("current_bracket", "input_close_open"),
        ("current_bracket", "output_bracket_noise"),
        ("forced_next", "output_close_open"),
        ("forced_next", "output_bracket_noise"),
        ("forced_next", "height"),
        ("left", "right"),
        ("height", "left"),
        ("height", "right"),
    ]
    rows = []
    for source, target in pairs:
        value = cosine(bank[source], bank[target])
        rows.append(
            {
                "setting": run.setting,
                "source": run.source,
                "bracket_tokens": int(run.bracket_tokens),
                "bracket_density": float(run.bracket_density),
                "layer": int(layer),
                "vector_a": source,
                "vector_b": target,
                "cosine": value,
                "abs_cosine": abs(value),
                "angle_deg": float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0)))),
            }
        )
    return rows


def token_metric_rows(
    run,
    labels: pd.DataFrame,
    target: np.ndarray,
    *,
    logits: np.ndarray,
    close_token: int,
    open_token: int,
    evaluation: str,
    ablation: str,
) -> list[dict[str, object]]:
    metrics = token_metrics(logits, target, close_token, open_token)
    rows = []
    for split, mask in split_masks(labels).items():
        if int(mask.sum()) == 0:
            continue
        row = {
            "setting": run.setting,
            "source": run.source,
            "bracket_tokens": int(run.bracket_tokens),
            "bracket_density": float(run.bracket_density),
            "seq_len": int(run.seq_len),
            "evaluation": evaluation,
            "ablation": ablation,
            "split": split,
            "n": int(mask.sum()),
            "train_eval_loss": float(run.train_eval_loss),
            "train_eval_accuracy": float(run.train_eval_accuracy),
            "summary_full_vocab_acc": float(run.all_dyck_targets_model_acc),
            "summary_forced_acc": float(run.forced_model_acc),
            "summary_free_acc": float(run.free_model_acc),
        }
        for metric_name, values in metrics.items():
            row[metric_name] = float(np.mean(values[mask]))
        rows.append(row)
    return rows


def token_metrics(logits: np.ndarray, target: np.ndarray, close_token: int, open_token: int) -> dict[str, np.ndarray]:
    pred = logits.argmax(axis=1)
    full_nll = token_nll(logits, target)
    bracket_logits = logits[:, [close_token, open_token]]
    bracket_target = np.where(target == close_token, 0, 1)
    bracket_pred = bracket_logits.argmax(axis=1)
    bracket_nll = token_nll(bracket_logits, bracket_target)
    log_denom = logsumexp(logits, axis=1)
    p_target = np.exp(logits[np.arange(len(target)), target] - log_denom)
    bracket_mass = np.exp(logits[:, close_token] - log_denom) + np.exp(logits[:, open_token] - log_denom)
    bracket_margin = logits[:, close_token] - logits[:, open_token]
    return {
        "full_vocab_acc": (pred == target).astype(float),
        "full_vocab_nll": full_nll,
        "full_vocab_target_prob": p_target,
        "bracket_acc": (bracket_pred == bracket_target).astype(float),
        "bracket_nll": bracket_nll,
        "bracket_mass": bracket_mass,
        "close_minus_open_margin": bracket_margin,
    }


def direction_ablation_rows(
    run,
    X: np.ndarray,
    labels: pd.DataFrame,
    target: np.ndarray,
    output: dict[str, np.ndarray],
    bank: dict[str, np.ndarray],
    close_token: int,
    open_token: int,
) -> list[dict[str, object]]:
    ablations: dict[str, np.ndarray] = {
        "baseline": np.empty((X.shape[1], 0), dtype=float),
        "remove_height": bank["height"][:, None],
        "remove_left": bank["left"][:, None],
        "remove_right": bank["right"][:, None],
        "remove_left_right": bank["left_right"],
        "remove_height_left_right": bank["height_left_right"],
        "remove_current_bracket": bank["current_bracket"][:, None],
        "remove_forced_next": bank["forced_next"][:, None],
        "remove_output_close_open": bank["output_close_open"][:, None],
        "remove_output_bracket_noise": bank["output_bracket_noise"][:, None],
        "remove_output_2d": bank["output_2d"],
        "remove_random": random_basis(X.shape[1], seed=setting_seed(run.setting, 140))[:, None],
    }
    rows = []
    for name, basis in ablations.items():
        if name == "baseline":
            X_variant = X
        else:
            X_variant = remove_projection(X, basis)
        logits = X_variant @ output["weight"].T + output["bias"]
        rows.extend(
            token_metric_rows(
                run,
                labels,
                target,
                logits=logits,
                close_token=close_token,
                open_token=open_token,
                evaluation="final_hidden_direction_ablation",
                ablation=name,
            )
        )
        del logits
        if name != "baseline":
            del X_variant
    return rows


def remove_projection(X: np.ndarray, basis: np.ndarray) -> np.ndarray:
    Q = orthonormal_basis([basis[:, i] for i in range(basis.shape[1])])
    mean = X.mean(axis=0, keepdims=True)
    centered = X - mean
    return X - (centered @ Q) @ Q.T


def add_ablation_deltas(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    metric_cols = [
        "full_vocab_acc",
        "full_vocab_nll",
        "full_vocab_target_prob",
        "bracket_acc",
        "bracket_nll",
        "bracket_mass",
        "close_minus_open_margin",
    ]
    base = (
        df[df["ablation"].eq("baseline")]
        .set_index(["setting", "split"])[metric_cols]
        .add_prefix("baseline_")
    )
    out = df.join(base, on=["setting", "split"])
    for col in metric_cols:
        out[f"delta_{col}_vs_baseline"] = out[col] - out[f"baseline_{col}"]
    return out


def compute_stability(
    direction_cache: dict[tuple[str, int, str], np.ndarray],
    sparse: pd.DataFrame,
) -> pd.DataFrame:
    sparse = sparse.sort_values("bracket_tokens")
    settings = sparse["setting"].tolist()
    bracket_map = sparse.set_index("setting")["bracket_tokens"].to_dict()
    ref_setting = sparse.iloc[-1]["setting"]
    rows = []
    names = sorted({key[2] for key in direction_cache})
    layers = sorted({key[1] for key in direction_cache})
    for name in names:
        for layer in layers:
            ref = direction_cache.get((ref_setting, layer, name))
            previous_setting = None
            for setting in settings:
                vector = direction_cache.get((setting, layer, name))
                if vector is None:
                    continue
                to_ref = cosine(vector, ref) if ref is not None else np.nan
                if previous_setting is None:
                    adjacent = np.nan
                else:
                    prev = direction_cache.get((previous_setting, layer, name))
                    adjacent = cosine(vector, prev) if prev is not None else np.nan
                rows.append(
                    {
                        "setting": setting,
                        "bracket_tokens": int(bracket_map[setting]),
                        "layer": int(layer),
                        "direction": name,
                        "reference_setting": ref_setting,
                        "cosine_to_reference": float(to_ref),
                        "abs_cosine_to_reference": float(abs(to_ref)) if np.isfinite(to_ref) else np.nan,
                        "cosine_to_previous_density": float(adjacent) if np.isfinite(adjacent) else np.nan,
                        "abs_cosine_to_previous_density": float(abs(adjacent)) if np.isfinite(adjacent) else np.nan,
                    }
                )
                previous_setting = setting
    return pd.DataFrame(rows)


def make_figures(
    token_loss: pd.DataFrame,
    geometry: pd.DataFrame,
    stability: pd.DataFrame,
    ablation: pd.DataFrame,
) -> None:
    plot_loss_behavior(token_loss, FIG_DIR / "sparse_loss_behavior.png")
    plot_alignment(geometry, FIG_DIR / "sparse_direction_alignment.png")
    plot_stability(stability, FIG_DIR / "sparse_direction_stability.png")
    plot_ablation(ablation, FIG_DIR / "sparse_direction_ablation.png")


def plot_loss_behavior(token_loss: pd.DataFrame, path: Path) -> None:
    base = token_loss[token_loss["evaluation"].eq("baseline")].copy()
    all_targets = base[base["split"].eq("all_dyck_targets")].sort_values("bracket_tokens")
    forced = base[base["split"].eq("forced")].sort_values("bracket_tokens")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    x = all_targets["bracket_tokens"].to_numpy()
    axes[0].plot(x, all_targets["full_vocab_acc"], marker="o", label="full-vocab Dyck acc")
    axes[0].plot(x, all_targets["bracket_acc"], marker="o", label="bracket-only Dyck acc")
    axes[0].plot(forced["bracket_tokens"], forced["full_vocab_acc"], marker="o", label="full-vocab forced acc")
    axes[0].set_xscale("log")
    axes[0].set_ylim(0, 1.03)
    axes[0].set_title("Accuracy depends on output space")
    axes[0].set_xlabel("bracket tokens in seq_len=2000")
    axes[0].set_ylabel("accuracy")
    axes[0].grid(alpha=0.22)
    axes[0].legend(fontsize=8)

    axes[1].plot(x, all_targets["full_vocab_nll"], marker="o", label="full-vocab Dyck NLL")
    axes[1].plot(x, all_targets["bracket_nll"], marker="o", label="bracket-only Dyck NLL")
    axes[1].plot(forced["bracket_tokens"], forced["full_vocab_nll"], marker="o", label="full-vocab forced NLL")
    axes[1].set_xscale("log")
    axes[1].set_title("Cross-entropy separates failure modes")
    axes[1].set_xlabel("bracket tokens in seq_len=2000")
    axes[1].set_ylabel("NLL")
    axes[1].grid(alpha=0.22)
    axes[1].legend(fontsize=8)

    axes[2].plot(x, all_targets["bracket_mass"], marker="o", label="P(open)+P(close)")
    axes2 = axes[2].twinx()
    axes2.plot(x, all_targets["train_eval_loss"], marker="o", color="#64748b", label="all-token eval loss")
    axes[2].set_xscale("log")
    axes[2].set_title("Sparse failure is mostly bracket mass")
    axes[2].set_xlabel("bracket tokens in seq_len=2000")
    axes[2].set_ylabel("mean bracket mass")
    axes2.set_ylabel("all-token eval loss")
    axes[2].grid(alpha=0.22)
    lines, labels = axes[2].get_legend_handles_labels()
    lines2, labels2 = axes2.get_legend_handles_labels()
    axes[2].legend(lines + lines2, labels + labels2, fontsize=8)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_alignment(geometry: pd.DataFrame, path: Path) -> None:
    final = geometry.sort_values("layer").groupby(["setting", "vector_a", "vector_b"], as_index=False).last()
    pairs = [
        ("height", "output_close_open"),
        ("forced_next", "output_close_open"),
        ("current_bracket", "input_close_open"),
        ("current_bracket", "output_bracket_noise"),
        ("height", "input_close_open"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for vector_a, vector_b in pairs:
        sub = final[(final["vector_a"].eq(vector_a)) & (final["vector_b"].eq(vector_b))].sort_values("bracket_tokens")
        if sub.empty:
            continue
        ax.plot(sub["bracket_tokens"], sub["abs_cosine"], marker="o", label=f"{vector_a} vs {vector_b}")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("bracket tokens in seq_len=2000")
    ax.set_ylabel("absolute cosine")
    ax.set_title("Final-layer direction alignment")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_stability(stability: pd.DataFrame, path: Path) -> None:
    final = stability.sort_values("layer").groupby(["setting", "direction"], as_index=False).last()
    directions = ["height", "left", "right", "current_bracket", "forced_next"]
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for direction in directions:
        sub = final[final["direction"].eq(direction)].sort_values("bracket_tokens")
        if sub.empty:
            continue
        ax.plot(sub["bracket_tokens"], sub["abs_cosine_to_reference"], marker="o", label=f"{direction} vs densest")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("bracket tokens in seq_len=2000")
    ax.set_ylabel("abs cosine to b400 direction")
    ax.set_title("Final-layer direction stability across sparsity")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(ablation: pd.DataFrame, path: Path) -> None:
    forced = ablation[
        ablation["split"].eq("forced")
        & ablation["ablation"].isin(
            [
                "remove_height",
                "remove_left_right",
                "remove_height_left_right",
                "remove_forced_next",
                "remove_output_close_open",
                "remove_output_bracket_noise",
                "remove_output_2d",
                "remove_random",
            ]
        )
    ].copy()
    if forced.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.0), constrained_layout=True)
    for ablation_name, sub in forced.groupby("ablation"):
        sub = sub.sort_values("bracket_tokens")
        axes[0].plot(sub["bracket_tokens"], sub["delta_full_vocab_nll_vs_baseline"], marker="o", label=short_ablation(ablation_name))
        axes[1].plot(sub["bracket_tokens"], sub["delta_bracket_nll_vs_baseline"], marker="o", label=short_ablation(ablation_name))
    for ax, title, ylabel in [
        (axes[0], "Forced targets: full-vocab NLL increase", "delta NLL"),
        (axes[1], "Forced targets: bracket-only NLL increase", "delta bracket NLL"),
    ]:
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("bracket tokens in seq_len=2000")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_readme(token_loss: pd.DataFrame, geometry: pd.DataFrame, stability: pd.DataFrame, ablation: pd.DataFrame) -> None:
    base = token_loss[(token_loss["evaluation"].eq("baseline")) & (token_loss["split"].eq("all_dyck_targets"))].copy()
    cols = [
        "setting",
        "bracket_tokens",
        "full_vocab_acc",
        "bracket_acc",
        "full_vocab_nll",
        "bracket_nll",
        "bracket_mass",
        "train_eval_loss",
    ]
    forced_ablation = ablation[
        ablation["split"].eq("forced")
        & ablation["ablation"].isin(["remove_height", "remove_left_right", "remove_forced_next", "remove_output_close_open"])
    ][
        ["setting", "bracket_tokens", "ablation", "delta_full_vocab_nll_vs_baseline", "delta_bracket_nll_vs_baseline"]
    ]
    text = (
        "# Task A Sparse Geometry Diagnostics\n\n"
        "Baseline token metrics:\n\n"
        + base.sort_values("bracket_tokens")[cols].to_string(index=False)
        + "\n\nForced-split direction ablations:\n\n"
        + forced_ablation.sort_values(["bracket_tokens", "ablation"]).to_string(index=False)
        + "\n"
    )
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(x)
    if norm < 1e-12:
        return np.zeros_like(x)
    return x / norm


def orthonormal_basis(vectors: list[np.ndarray]) -> np.ndarray:
    cols = []
    for vector in vectors:
        v = normalize(vector)
        for q in cols:
            v = v - np.dot(v, q) * q
        norm = np.linalg.norm(v)
        if norm > 1e-8:
            cols.append(v / norm)
    if not cols:
        raise ValueError("No nonzero vectors for basis")
    return np.stack(cols, axis=1)


def random_basis(dim: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return normalize(rng.normal(size=dim))


def cosine(a: np.ndarray, b: np.ndarray | None) -> float:
    if b is None:
        return np.nan
    a = normalize(a)
    b = normalize(b)
    return float(np.dot(a, b))


def token_nll(logits: np.ndarray, target: np.ndarray) -> np.ndarray:
    return logsumexp(logits, axis=1) - logits[np.arange(len(target)), target]


def logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    max_x = np.max(x, axis=axis, keepdims=True)
    out = np.log(np.exp(x - max_x).sum(axis=axis, keepdims=True)) + max_x
    return np.squeeze(out, axis=axis)


def short_ablation(name: str) -> str:
    return {
        "remove_height": "height",
        "remove_left_right": "left/right",
        "remove_height_left_right": "h/l/r",
        "remove_forced_next": "forced-next",
        "remove_output_close_open": "out close-open",
        "remove_output_bracket_noise": "out bracket-noise",
        "remove_output_2d": "out 2D",
        "remove_random": "random",
    }.get(name, name)


if __name__ == "__main__":
    main()
