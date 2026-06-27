from __future__ import annotations

import argparse
import math
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

from hse.models import build_model
from hse.tasks.registry import batch_to_cpu, build_labels, build_sampler, task_name_from_run_config
from hse.utils import load_json

from scripts.task_a_extra_probes import prepare_labels


SUMMARY_PATH = ROOT / "results" / "dyck_counter_task_a_summary.csv"
SPARSE_SUMMARY_PATH = ROOT / "results" / "dyck_counter_sparse_supervision_ablation" / "summary.csv"
OUT_DIR = ROOT / "results" / "dyck_counter_task_a_activation_patch"
FIG_DIR = ROOT / "figures" / "dyck_counter_task_a_activation_patch"
DEFAULT_SETTINGS = ["tiny_extreme_long", "sparse_len2000_b48", "sparse_len2000_b200", "extreme_long"]
DELTAS = [-2.0, -1.0, 0.0, 1.0, 2.0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", nargs="*", default=DEFAULT_SETTINGS)
    parser.add_argument("--num-examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed-offset", type=int, default=40_000)
    parser.add_argument("--from-existing", action="store_true", help="Reuse the raw patch CSV instead of running new forward passes.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = OUT_DIR / "layerwise_activation_patch.csv"
    if args.from_existing:
        df = pd.read_csv(raw_path)
    else:
        specs = load_run_specs(args.settings)
        rows = []
        for spec in specs:
            print(f"patching {spec['setting']} ({int(spec['seq_len'])} tokens, {int(spec['total_length'])} brackets)")
            rows.extend(
                run_setting_patch(
                    spec,
                    num_examples=args.num_examples,
                    batch_size=args.batch_size,
                    device=args.device,
                    seed_offset=args.seed_offset,
                )
            )
        df = pd.DataFrame(rows)
        df.to_csv(raw_path, index=False)
    aggregate_df = aggregate_patch_results(df)
    aggregate_df.to_csv(OUT_DIR / "layerwise_activation_patch_aggregated.csv", index=False)
    slope_df = summarize_slopes(aggregate_df)
    slope_df.to_csv(OUT_DIR / "layerwise_activation_patch_slopes.csv", index=False)
    make_figure(aggregate_df, slope_df)
    print(f"wrote {raw_path}")
    print(f"wrote {OUT_DIR / 'layerwise_activation_patch_aggregated.csv'}")
    print(f"wrote {OUT_DIR / 'layerwise_activation_patch_slopes.csv'}")
    print(f"wrote {FIG_DIR / 'layerwise_activation_patch.png'}")


def load_run_specs(settings: list[str]) -> list[pd.Series]:
    rows = []
    if SUMMARY_PATH.exists():
        primary = pd.read_csv(SUMMARY_PATH)
        for row in primary.to_dict(orient="records"):
            row["source"] = row["setting"]
            rows.append(row)
    if SPARSE_SUMMARY_PATH.exists():
        sparse = pd.read_csv(SPARSE_SUMMARY_PATH)
        for row in sparse.to_dict(orient="records"):
            if row["source"] in {"tiny_extreme_long", "extreme_long"}:
                continue
            rows.append(
                {
                    "setting": row["source"],
                    "source": row["source"],
                    "run_dir": row["run_dir"],
                    "seq_len": row["seq_len"],
                    "total_length": row["bracket_tokens"],
                    "repeat_prob": row["repeat_prob"],
                    "noise_vocab": row["noise_vocab"],
                    "best_layer": row["best_layer"],
                }
            )
    by_setting = {str(row["setting"]): row for row in rows}
    missing = [setting for setting in settings if setting not in by_setting]
    if missing:
        raise FileNotFoundError(f"Missing settings in summary tables: {missing}")
    return [pd.Series(by_setting[setting]) for setting in settings]


@torch.no_grad()
def run_setting_patch(
    spec: pd.Series,
    *,
    num_examples: int,
    batch_size: int,
    device: str,
    seed_offset: int,
) -> list[dict[str, object]]:
    run_dir = ROOT / str(spec["run_dir"])
    config = load_json(run_dir / "config.json")
    task_name = task_name_from_run_config(config)
    task_kwargs = {k: v for k, v in config["task"].items() if k != "device"}
    sampler = build_sampler(task_name, task_kwargs, device="cpu", seed=int(config["seed"]) + seed_offset)
    model = load_model(run_dir, config, sampler.vocab_size, device)
    rows = []
    axis_scales = {layer: hidden_axis_scale(run_dir, layer) for layer in available_layers(run_dir)}
    directions = {
        layer: {
            "height": load_direction(run_dir, layer, "height").to(device),
            "random": random_direction_like(load_direction(run_dir, layer, "height"), seed=setting_seed(str(spec["setting"]), layer)).to(device),
        }
        for layer in available_layers(run_dir)
    }
    seen = 0
    while seen < num_examples:
        current = min(batch_size, num_examples - seen)
        batch = sampler.sample(current)
        tokens = batch.tokens.to(device)
        raw_labels = build_labels(task_name, batch_to_cpu(batch), sampler.config, max_prefix_len=None)
        row_view = make_row_view(spec, run_dir)
        labels = prepare_labels(raw_labels, row_view)
        labels = choose_one_target_per_example(labels, seed=setting_seed(str(spec["setting"]), seen + 17))
        if labels.empty:
            seen += current
            continue
        positions = torch.tensor(labels["position"].to_numpy(dtype=int), device=device)
        examples = torch.tensor(labels["example_id"].to_numpy(dtype=int), device=device)
        target = torch.tensor(labels["target_token"].to_numpy(dtype=int), device=device)
        state = labels["forced_state"].to_numpy()
        close_token = int(spec["noise_vocab"])
        open_token = close_token + 1

        layer_outputs = forward_layer_outputs(model, tokens)
        for layer, layer_h in layer_outputs.items():
            for direction_name, direction in directions[layer].items():
                axis_scale = axis_scales[layer]
                for delta in DELTAS:
                    logits = continue_with_patch(
                        model,
                        layer_h,
                        patch_layer=layer,
                        positions=positions,
                        examples=examples,
                        direction=direction,
                        shift=float(delta) * axis_scale,
                    )
                    selected = logits[examples, positions]
                    pred = selected.argmax(dim=-1)
                    correct = (pred == target).detach().cpu().numpy()
                    margin = (selected[:, close_token] - selected[:, open_token]).detach().cpu().numpy()
                    p_close = torch.sigmoid(selected[:, close_token] - selected[:, open_token]).detach().cpu().numpy()
                    for split_name, mask in split_masks_for_state(state).items():
                        if not mask.any():
                            continue
                        rows.append(
                            {
                                "setting": str(spec["setting"]),
                                "run_dir": str(spec["run_dir"]),
                                "bracket_tokens": int(spec["total_length"]),
                                "seq_len": int(spec["seq_len"]),
                                "layer": int(layer),
                                "direction": direction_name,
                                "delta_height_axis_std": float(delta),
                                "split": split_name,
                                "n": int(mask.sum()),
                                "accuracy": float(correct[mask].mean()),
                                "mean_close_minus_open_margin": float(margin[mask].mean()),
                                "mean_p_close_given_bracket": float(p_close[mask].mean()),
                                "axis_scale": float(axis_scale),
                            }
                        )
        seen += current
    return rows


def load_model(run_dir: Path, config: dict, vocab_size: int, device: str):
    spec = config["model"]
    model_kwargs = {k: v for k, v in spec.items() if k not in {"name", "state_kind"}}
    model = build_model(model_name=config["model_name"], vocab_size=vocab_size, **model_kwargs).to(device)
    checkpoint = torch.load(run_dir / "checkpoints" / "model_final.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def make_row_view(spec: pd.Series, run_dir: Path):
    return type(
        "RowView",
        (),
        {
            "setting": str(spec["setting"]),
            "run_dir_abs": run_dir,
            "noise_vocab": int(spec["noise_vocab"]),
            "total_length": int(spec["total_length"]),
        },
    )()


def choose_one_target_per_example(labels: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    mask = labels["has_target_label"].to_numpy(dtype=bool) & labels["target_is_dyck_position"].to_numpy(dtype=bool)
    candidates = labels.loc[mask].copy()
    if candidates.empty:
        return candidates
    rng = np.random.default_rng(seed)
    chosen = []
    for _, group in candidates.groupby("example_id", sort=False):
        chosen.append(rng.choice(group.index.to_numpy(), size=1)[0])
    return candidates.loc[np.sort(chosen)].reset_index(drop=True)


def split_masks_for_state(state: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "all_dyck_targets": np.ones(len(state), dtype=bool),
        "forced": state != "free",
        "must_open": state == "must_open",
        "must_close": state == "must_close",
        "free": state == "free",
    }


def forward_layer_outputs(model, tokens: torch.Tensor) -> dict[int, torch.Tensor]:
    _, traces = model(tokens, return_traces=True)
    states = traces["h"]
    return {layer: states[layer] for layer in range(int(model.num_layers))}


def continue_with_patch(
    model,
    layer_h: torch.Tensor,
    *,
    patch_layer: int,
    positions: torch.Tensor,
    examples: torch.Tensor,
    direction: torch.Tensor,
    shift: float,
) -> torch.Tensor:
    h = layer_h.clone()
    h[examples, positions] = h[examples, positions] + float(shift) * direction.view(1, -1)
    seq_len = h.shape[1]
    if patch_layer < int(model.num_layers) - 1:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=h.device, dtype=torch.bool), diagonal=1)
        for layer in model.layers[patch_layer + 1 :]:
            h = layer(h, src_mask=mask)
        h = model.final_norm(h)
    return model.output(h)


def hidden_axis_scale(run_dir: Path, layer: int) -> float:
    labels = pd.read_parquet(run_dir / "hidden_states" / "final" / "labels.parquet")
    run_config = load_json(run_dir / "config.json")
    row_view = make_row_view(
        pd.Series(
            {
                "setting": run_dir.parent.name,
                "noise_vocab": run_config["task"]["num_noise_tokens"],
                "total_length": run_config["task"]["total_length"],
            }
        ),
        run_dir,
    )
    labels = prepare_labels(labels, row_view)
    mask = labels["has_target_label"].to_numpy(dtype=bool) & labels["target_is_dyck_position"].to_numpy(dtype=bool)
    idx = np.flatnonzero(mask)
    if len(idx) > 30_000:
        rng = np.random.default_rng(1234 + layer)
        idx = rng.choice(idx, size=30_000, replace=False)
    hidden = torch.load(run_dir / "hidden_states" / "final" / f"layer_{layer}.pt", map_location="cpu")
    direction = load_direction(run_dir, layer, "height").cpu()
    projection = hidden[idx].float() @ direction.float()
    scale = float(projection.std().item())
    return scale if scale > 1e-8 else 1.0


def available_layers(run_dir: Path) -> list[int]:
    return sorted(int(path.stem.removeprefix("layer_")) for path in (run_dir / "hidden_states" / "final").glob("layer_*.pt"))


def load_direction(run_dir: Path, layer: int, target: str) -> torch.Tensor:
    path = run_dir / "probes" / "directions" / f"final_layer_{layer}_{target}.pt"
    direction = torch.load(path, map_location="cpu").float().reshape(-1)
    return direction / (direction.norm() + 1e-12)


def random_direction_like(direction: torch.Tensor, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    random = torch.randn(direction.shape, generator=generator)
    direction = direction.cpu().float()
    random = random - torch.dot(random, direction) * direction
    return random / (random.norm() + 1e-12)


def summarize_slopes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(["setting", "bracket_tokens", "seq_len", "layer", "direction", "split"], sort=False):
        x = group["delta_height_axis_std"].to_numpy(dtype=float)
        for metric in ["accuracy", "mean_p_close_given_bracket", "mean_close_minus_open_margin"]:
            y = group[metric].to_numpy(dtype=float)
            slope = np.polyfit(x, y, deg=1)[0] if len(np.unique(x)) > 1 else np.nan
            rows.append(
                {
                    "setting": keys[0],
                    "bracket_tokens": int(keys[1]),
                    "seq_len": int(keys[2]),
                    "layer": int(keys[3]),
                    "direction": keys[4],
                    "split": keys[5],
                    "metric": metric,
                    "slope_per_axis_std": float(slope),
                }
            )
    return pd.DataFrame(rows)


def aggregate_patch_results(df: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "setting",
        "run_dir",
        "bracket_tokens",
        "seq_len",
        "layer",
        "direction",
        "delta_height_axis_std",
        "split",
    ]
    rows = []
    for group_keys, group in df.groupby(keys, sort=False):
        n = group["n"].to_numpy(dtype=float)
        weight = n / max(float(n.sum()), 1.0)
        row = {key: value for key, value in zip(keys, group_keys)}
        row["n"] = int(n.sum())
        for metric in ["accuracy", "mean_close_minus_open_margin", "mean_p_close_given_bracket", "axis_scale"]:
            row[metric] = float(np.sum(group[metric].to_numpy(dtype=float) * weight))
        rows.append(row)
    return pd.DataFrame(rows)


def make_figure(df: pd.DataFrame, slope_df: pd.DataFrame) -> None:
    focused = df[(df["split"] == "all_dyck_targets") & (df["direction"] == "height")].copy()
    slope_focus = slope_df[
        (slope_df["split"] == "all_dyck_targets")
        & (slope_df["direction"] == "height")
        & (slope_df["metric"] == "mean_p_close_given_bracket")
    ].copy()
    random_focus = slope_df[
        (slope_df["split"] == "all_dyck_targets")
        & (slope_df["direction"] == "random")
        & (slope_df["metric"] == "mean_p_close_given_bracket")
    ].copy()
    order = [setting for setting in DEFAULT_SETTINGS if setting in set(df["setting"])]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for setting in order:
        sub = focused[(focused["setting"] == setting) & (focused["layer"] == focused[focused["setting"] == setting]["layer"].max())]
        if not sub.empty:
            sub = sub.sort_values("delta_height_axis_std")
            axes[0].plot(sub["delta_height_axis_std"], sub["mean_p_close_given_bracket"], marker="o", label=setting)
    axes[0].set_title("Patch final available layer")
    axes[0].set_xlabel("delta along height direction")
    axes[0].set_ylabel("P(close | bracket logits)")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    heat = (
        slope_focus.pivot_table(index="setting", columns="layer", values="slope_per_axis_std", aggfunc="mean")
        .reindex(order)
    )
    im = axes[1].imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-0.1, vmax=0.1)
    axes[1].set_title("Height-patch P(close) slope")
    axes[1].set_xlabel("layer")
    axes[1].set_yticks(np.arange(len(heat.index)))
    axes[1].set_yticklabels(heat.index)
    axes[1].set_xticks(np.arange(len(heat.columns)))
    axes[1].set_xticklabels(heat.columns)
    for i, setting in enumerate(heat.index):
        for j, layer in enumerate(heat.columns):
            value = heat.loc[setting, layer]
            if pd.notna(value):
                axes[1].text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    merged = slope_focus.merge(
        random_focus,
        on=["setting", "bracket_tokens", "seq_len", "layer", "split", "metric"],
        how="left",
        suffixes=("_height", "_random"),
    )
    best = merged.sort_values(["setting", "slope_per_axis_std_height"], ascending=[True, False]).groupby("setting", as_index=False).tail(1)
    best = best.set_index("setting").reindex(order).reset_index()
    x = np.arange(len(best))
    axes[2].bar(x - 0.18, best["slope_per_axis_std_height"], width=0.36, label="height")
    axes[2].bar(x + 0.18, best["slope_per_axis_std_random"], width=0.36, label="random control")
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_title("Best-layer slope vs random")
    axes[2].set_ylabel("slope per axis std")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(best["setting"], rotation=35, ha="right")
    axes[2].grid(True, axis="y", alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "layerwise_activation_patch.png", dpi=180)
    plt.close(fig)


def setting_seed(setting: str, salt: int) -> int:
    return (sum((i + 1) * ord(ch) for i, ch in enumerate(setting)) + 7919 * salt) % (2**32 - 1)


if __name__ == "__main__":
    main()
