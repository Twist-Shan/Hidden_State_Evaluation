from __future__ import annotations

import argparse
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
import torch

from hse.tasks.registry import batch_to_cpu, build_labels, build_sampler, task_name_from_run_config
from hse.utils import load_json
from scripts.task_a_countscope_online_patching import (
    DEFAULT_SETTINGS,
    aggregate,
    available_layers,
    build_local_pairs,
    continue_with_selected_patch,
    label_batch,
    load_model,
    load_specs,
    metric_rows,
    setting_seed,
)
from scripts.task_a_extra_probes import (
    choose_indices,
    load_hidden_rows,
    load_labels,
    load_output_head,
    load_probe_direction,
    prepare_labels,
)


OUT_DIR = ROOT / "results" / "dyck_counter_task_a_axis_span_patching"
FIG_DIR = ROOT / "figures" / "dyck_counter_task_a_axis_span_patching"

PATCH_MODES = [
    "target_self",
    "full_source",
    "full_source_shuffle",
    "output_close_open_scalar",
    "output_bracket_noise_scalar",
    "output_output2d_span",
    "height_scalar",
    "left_scalar",
    "right_scalar",
    "left_right_span",
    "height_left_right_span",
    "current_bracket_scalar",
    "forced_next_scalar",
    "mean_forced",
    "zero",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", nargs="*", default=DEFAULT_SETTINGS)
    parser.add_argument("--pairs-per-setting", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=260)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed-offset", type=int, default=91_000)
    parser.add_argument("--from-existing", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / "axis_span_patching_raw.csv"

    if args.from_existing:
        raw = pd.read_csv(raw_path)
    else:
        rows = []
        for setting_index, spec in enumerate(load_specs(args.settings)):
            print(f"axis/span patching {spec.setting}: brackets={spec.bracket_tokens}, seq_len={spec.seq_len}")
            rows.extend(
                run_setting(
                    spec,
                    setting_index=setting_index,
                    pairs_per_setting=args.pairs_per_setting,
                    batch_size=args.batch_size,
                    max_batches=args.max_batches,
                    device=args.device,
                    seed_offset=args.seed_offset,
                )
            )
            gc.collect()
            if str(args.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        raw = pd.DataFrame(rows)
        raw.to_csv(raw_path, index=False)

    summary = aggregate(raw)
    summary.to_csv(OUT_DIR / "axis_span_patching_summary.csv", index=False)
    make_figures(summary)
    write_readme(summary)
    print(f"wrote {raw_path}")
    print(f"wrote {OUT_DIR / 'axis_span_patching_summary.csv'}")
    print(f"wrote figures to {FIG_DIR}")


@torch.no_grad()
def run_setting(
    spec,
    *,
    setting_index: int,
    pairs_per_setting: int,
    batch_size: int,
    max_batches: int,
    device: str,
    seed_offset: int,
) -> list[dict[str, object]]:
    config = load_json(spec.run_dir / "config.json")
    task_name = task_name_from_run_config(config)
    task_kwargs = {key: value for key, value in config["task"].items() if key != "device"}
    sampler = build_sampler(
        task_name,
        task_kwargs,
        device="cpu",
        seed=int(config["seed"]) + seed_offset + setting_index,
    )
    model = load_model(spec.run_dir, config, sampler.vocab_size, device)
    layers = available_layers(spec.run_dir)
    axis_bank = load_axis_bank(spec, layers, device=device)

    rows: list[dict[str, object]] = []
    collected = 0
    batch_index = 0
    rng = np.random.default_rng(setting_seed(spec.setting, 23))

    while batch_index < max_batches and collected < pairs_per_setting:
        target_batch = sampler.sample(batch_size)
        source_batch = sampler.sample(batch_size)
        target_tokens = target_batch.tokens.to(device)
        source_tokens = source_batch.tokens.to(device)
        target_labels = label_batch(task_name, target_batch, sampler.config, spec)
        source_labels = label_batch(task_name, source_batch, sampler.config, spec)

        target_logits, target_traces = model(target_tokens, return_traces=True)
        _source_logits, source_traces = model(source_tokens, return_traces=True)
        pairs = build_local_pairs(
            target_labels,
            source_labels,
            max_pairs=pairs_per_setting - collected,
            rng=rng,
        )
        if pairs:
            rows.extend(
                run_pair_set(
                    spec=spec,
                    model=model,
                    target_logits=target_logits,
                    target_traces=target_traces,
                    source_traces=source_traces,
                    pairs=pairs,
                    layers=layers,
                    axis_bank=axis_bank,
                    batch_index=batch_index,
                    rng=rng,
                )
            )
            collected += len(pairs)

        batch_index += 1
        del target_logits, target_traces, source_traces
        gc.collect()

    print(f"  collected local={collected}, batches={batch_index}")
    del model
    return rows


@torch.no_grad()
def run_pair_set(
    *,
    spec,
    model,
    target_logits: torch.Tensor,
    target_traces: dict[str, torch.Tensor],
    source_traces: dict[str, torch.Tensor],
    pairs: list[dict[str, int]],
    layers: list[int],
    axis_bank: dict[int, dict[str, torch.Tensor]],
    batch_index: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    device = target_logits.device
    target_examples = torch.tensor([pair["target_example"] for pair in pairs], device=device, dtype=torch.long)
    patch_positions = torch.tensor([pair["patch_position"] for pair in pairs], device=device, dtype=torch.long)
    eval_positions = torch.tensor([pair["eval_position"] for pair in pairs], device=device, dtype=torch.long)
    source_examples = torch.tensor([pair["source_example"] for pair in pairs], device=device, dtype=torch.long)
    source_positions = torch.tensor([pair["source_position"] for pair in pairs], device=device, dtype=torch.long)
    target_tokens = torch.tensor([pair["target_token"] for pair in pairs], device=device, dtype=torch.long)
    hypothesis_tokens = torch.tensor([pair["hypothesis_token"] for pair in pairs], device=device, dtype=torch.long)

    baseline_logits = target_logits[target_examples, eval_positions]
    close_token = int(spec.noise_vocab)
    bracket_logits = torch.stack([baseline_logits[:, close_token], baseline_logits[:, close_token + 1]], dim=1)
    baseline_probs = torch.softmax(bracket_logits, dim=1)
    rows: list[dict[str, object]] = []

    for layer in layers:
        target_h = target_traces["h"][layer]
        source_h = source_traces["h"][layer]
        source_values = source_h[source_examples, source_positions]
        target_values = target_h[target_examples, patch_positions]
        perm = torch.as_tensor(rng.permutation(len(pairs)), device=device, dtype=torch.long)
        mean_value = source_values.mean(dim=0, keepdim=True).expand_as(source_values)
        zero_value = torch.zeros_like(source_values)
        bank = axis_bank[layer]
        modes = {
            "target_self": (target_values, hypothesis_tokens),
            "full_source": (source_values, hypothesis_tokens),
            "full_source_shuffle": (source_values[perm], hypothesis_tokens[perm]),
            "mean_forced": (mean_value, hypothesis_tokens),
            "zero": (zero_value, hypothesis_tokens),
        }
        modes.update(axis_patch_values(target_values, source_values, bank, hypothesis_tokens))

        for mode in PATCH_MODES:
            if mode not in modes:
                continue
            patch_values, mode_hypothesis_tokens = modes[mode]
            patched_logits_all = continue_with_selected_patch(
                model,
                target_h,
                patch_layer=layer,
                examples=target_examples,
                patch_positions=patch_positions,
                patch_values=patch_values,
            )
            patched_logits = patched_logits_all[target_examples, eval_positions]
            rows.extend(
                metric_rows(
                    experiment="local_axis_span",
                    spec=spec,
                    layer=layer,
                    mode=mode,
                    batch_index=batch_index,
                    pairs=pairs,
                    baseline_logits=baseline_logits,
                    baseline_probs=baseline_probs,
                    patched_logits=patched_logits,
                    target_tokens=target_tokens,
                    hypothesis_tokens=mode_hypothesis_tokens,
                )
            )
            del patched_logits_all
    return rows


def axis_patch_values(
    target_values: torch.Tensor,
    source_values: torch.Tensor,
    bank: dict[str, torch.Tensor],
    hypothesis_tokens: torch.Tensor,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {
        "output_close_open_scalar": (copy_scalar(target_values, source_values, bank["output_close_open"]), hypothesis_tokens),
        "output_bracket_noise_scalar": (
            copy_scalar(target_values, source_values, bank["output_bracket_noise"]),
            hypothesis_tokens,
        ),
        "output_output2d_span": (copy_span(target_values, source_values, bank["output_2d"]), hypothesis_tokens),
        "height_scalar": (copy_scalar(target_values, source_values, bank["height"]), hypothesis_tokens),
        "left_scalar": (copy_scalar(target_values, source_values, bank["left"]), hypothesis_tokens),
        "right_scalar": (copy_scalar(target_values, source_values, bank["right"]), hypothesis_tokens),
        "left_right_span": (copy_span(target_values, source_values, bank["left_right"]), hypothesis_tokens),
        "height_left_right_span": (copy_span(target_values, source_values, bank["height_left_right"]), hypothesis_tokens),
        "current_bracket_scalar": (copy_scalar(target_values, source_values, bank["current_bracket"]), hypothesis_tokens),
        "forced_next_scalar": (copy_scalar(target_values, source_values, bank["forced_next"]), hypothesis_tokens),
    }


def copy_scalar(target: torch.Tensor, source: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    delta = (source - target) @ direction
    return target + delta[:, None] * direction[None, :]


def copy_span(target: torch.Tensor, source: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    delta = (source - target) @ basis
    return target + delta @ basis.T


def load_axis_bank(spec, layers: list[int], *, device: str) -> dict[int, dict[str, torch.Tensor]]:
    output = load_output_head(spec.run_dir)
    weight = output["weight"]
    close_token = int(spec.noise_vocab)
    open_token = close_token + 1
    output_close_open = normalize(weight[close_token] - weight[open_token])
    output_bracket_noise = normalize(weight[[close_token, open_token]].mean(axis=0) - weight[:close_token].mean(axis=0))
    output_2d = orthonormal_basis([output_close_open, output_bracket_noise])

    labels = prepare_labels(load_labels(spec.run_dir), spec)
    bank = {}
    for layer in layers:
        height = normalize(load_probe_direction(spec.run_dir, layer, "height"))
        left = normalize(load_probe_direction(spec.run_dir, layer, "left"))
        right = normalize(load_probe_direction(spec.run_dir, layer, "right"))
        current_bracket = mean_difference_axis(
            spec.run_dir,
            layer,
            labels,
            positive_mask=labels["is_dyck_position"].to_numpy(dtype=bool) & labels["token"].eq(close_token).to_numpy(dtype=bool),
            negative_mask=labels["is_dyck_position"].to_numpy(dtype=bool) & labels["token"].eq(open_token).to_numpy(dtype=bool),
            seed=setting_seed(spec.setting, 31 + layer),
        )
        forced_next = mean_difference_axis(
            spec.run_dir,
            layer,
            labels,
            positive_mask=forced_target_mask(labels, close_token),
            negative_mask=forced_target_mask(labels, open_token),
            seed=setting_seed(spec.setting, 41 + layer),
        )
        if current_bracket is None:
            current_bracket = output_close_open.copy()
        if forced_next is None:
            forced_next = output_close_open.copy()
        bank[layer] = {
            "output_close_open": to_device(output_close_open, device),
            "output_bracket_noise": to_device(output_bracket_noise, device),
            "output_2d": to_device(output_2d, device),
            "height": to_device(height, device),
            "left": to_device(left, device),
            "right": to_device(right, device),
            "left_right": to_device(orthonormal_basis([left, right]), device),
            "height_left_right": to_device(orthonormal_basis([height, left, right]), device),
            "current_bracket": to_device(current_bracket, device),
            "forced_next": to_device(forced_next, device),
        }
    return bank


def forced_target_mask(labels: pd.DataFrame, token: int) -> np.ndarray:
    return (
        labels["has_target_label"].to_numpy(dtype=bool)
        & labels["target_is_dyck_position"].to_numpy(dtype=bool)
        & labels["forced_state"].ne("free").to_numpy(dtype=bool)
        & labels["target_token"].eq(token).to_numpy(dtype=bool)
    )


def mean_difference_axis(
    run_dir: Path,
    layer: int,
    labels: pd.DataFrame,
    *,
    positive_mask: np.ndarray,
    negative_mask: np.ndarray,
    seed: int,
    max_rows_per_class: int = 60_000,
) -> np.ndarray | None:
    pos_idx = choose_indices(positive_mask, max_rows=max_rows_per_class, seed=seed)
    neg_idx = choose_indices(negative_mask, max_rows=max_rows_per_class, seed=seed + 1)
    if len(pos_idx) < 20 or len(neg_idx) < 20:
        return None
    X_pos = load_hidden_rows(run_dir, layer, pos_idx)
    X_neg = load_hidden_rows(run_dir, layer, neg_idx)
    axis = normalize(X_pos.mean(axis=0) - X_neg.mean(axis=0))
    del X_pos, X_neg
    return axis


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


def to_device(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def make_figures(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    plot_axis_modes(summary, metric="ci_minus_self", path=FIG_DIR / "axis_span_bracket_ci.png")
    plot_axis_modes(
        summary,
        metric="full_vocab_ci_minus_self",
        path=FIG_DIR / "axis_span_full_vocab_ci.png",
        ylabel="best-layer full-vocab CI minus self-patch",
    )
    plot_best_layer_heatmap(summary, path=FIG_DIR / "axis_span_best_layer_heatmap.png")


def plot_axis_modes(
    summary: pd.DataFrame,
    *,
    metric: str,
    path: Path,
    ylabel: str = "best-layer bracket-normalized CI minus self-patch",
) -> None:
    data = best_by_mode(summary)
    modes = [
        "full_source",
        "full_source_shuffle",
        "output_close_open_scalar",
        "output_bracket_noise_scalar",
        "output_output2d_span",
        "height_scalar",
        "left_right_span",
        "height_left_right_span",
        "current_bracket_scalar",
        "forced_next_scalar",
    ]
    settings = list(dict.fromkeys(data["setting"].tolist()))
    fig, axes = plt.subplots(1, len(settings), figsize=(5.2 * len(settings), 4.8), sharey=True)
    if len(settings) == 1:
        axes = [axes]
    for ax, setting in zip(axes, settings):
        sub = data[data["setting"].eq(setting)]
        values = []
        labels = []
        for mode in modes:
            row = sub[sub["mode"].eq(mode)]
            if row.empty:
                continue
            values.append(float(row[metric].iloc[0]))
            labels.append(short_mode(mode))
        ax.bar(np.arange(len(values)), values, color="#4f46e5", alpha=0.82)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(setting)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_best_layer_heatmap(summary: pd.DataFrame, *, path: Path) -> None:
    data = best_by_mode(summary)
    modes = [
        "full_source",
        "output_close_open_scalar",
        "output_bracket_noise_scalar",
        "output_output2d_span",
        "height_scalar",
        "left_scalar",
        "right_scalar",
        "left_right_span",
        "height_left_right_span",
        "current_bracket_scalar",
        "forced_next_scalar",
    ]
    settings = list(dict.fromkeys(data["setting"].tolist()))
    values = np.full((len(settings), len(modes)), np.nan)
    layer_labels = np.full((len(settings), len(modes)), "", dtype=object)
    for i, setting in enumerate(settings):
        for j, mode in enumerate(modes):
            row = data[data["setting"].eq(setting) & data["mode"].eq(mode)]
            if not row.empty:
                values[i, j] = float(row["ci_minus_self"].iloc[0])
                layer_labels[i, j] = f"L{int(row['layer'].iloc[0])}"
    fig, ax = plt.subplots(figsize=(12.5, max(3.5, 0.75 * len(settings))))
    im = ax.imshow(values, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(modes)))
    ax.set_xticklabels([short_mode(mode) for mode in modes], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(settings)))
    ax.set_yticklabels(settings)
    ax.set_title("Best-layer bracket CI by patched direction/span")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.2f}\n{layer_labels[i, j]}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def best_by_mode(summary: pd.DataFrame) -> pd.DataFrame:
    focus = summary[summary["split"].eq("ci_valid")].copy()
    if focus.empty:
        return focus
    return (
        focus.sort_values(["setting", "mode", "ci_minus_self"], ascending=[True, True, False])
        .groupby(["setting", "mode"], as_index=False)
        .first()
        .sort_values(["setting", "mode"])
    )


def short_mode(mode: str) -> str:
    return {
        "full_source": "full",
        "full_source_shuffle": "shuffle",
        "output_close_open_scalar": "out close-open",
        "output_bracket_noise_scalar": "out bracket-noise",
        "output_output2d_span": "out 2D",
        "height_scalar": "height",
        "left_scalar": "left",
        "right_scalar": "right",
        "left_right_span": "left/right",
        "height_left_right_span": "h/l/r",
        "current_bracket_scalar": "current bracket",
        "forced_next_scalar": "forced next",
        "mean_forced": "mean",
        "zero": "zero",
    }.get(mode, mode)


def write_readme(summary: pd.DataFrame) -> None:
    if summary.empty:
        text = "No axis/span patching rows were produced.\n"
    else:
        best = best_by_mode(summary)
        cols = [
            "setting",
            "mode",
            "layer",
            "ci_minus_self",
            "full_vocab_ci_minus_self",
            "patched_hypothesis_acc",
            "patched_full_hypothesis_acc",
        ]
        text = (
            "# Task A Axis/Span Patching\n\n"
            "This is local online activation patching where only selected scalar directions or spans are copied from source to target.\n\n"
            + best[cols].to_string(index=False)
            + "\n"
        )
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
