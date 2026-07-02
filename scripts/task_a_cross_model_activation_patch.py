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
import torch.nn.functional as F

from hse.models import build_model
from hse.tasks.registry import batch_to_cpu, build_labels, build_sampler, task_name_from_run_config
from hse.utils import load_json
from scripts.task_a_extra_probes import prepare_labels


SPARSE_SUMMARY_PATH = ROOT / "results" / "dyck_counter_sparse_supervision_ablation" / "summary.csv"
OUT_DIR = ROOT / "results" / "dyck_counter_task_a_cross_model_patch"
FIG_DIR = ROOT / "figures" / "dyck_counter_task_a_cross_model_patch"
DEFAULT_PAIRS = ["b100:b20", "b100:b34", "b80:b20", "b80:b34", "b64:b20", "b64:b34", "b20:b100", "b34:b100"]
PATCH_MODES = [
    "recipient_baseline",
    "donor_baseline",
    "full_state_replace",
    "full_state_shuffle",
    "recipient_self_patch",
    "height_scalar_replace",
    "height_scalar_shuffle",
    "retrieval_full_match",
    "retrieval_full_state_random",
    "retrieval_height_match",
    "retrieval_height_state_random",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="*", default=DEFAULT_PAIRS, help="Pairs as donor:recipient, e.g. b100:b20.")
    parser.add_argument("--num-examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed-offset", type=int, default=50_000)
    parser.add_argument("--retrieval-max-bank-rows", type=int, default=50_000)
    parser.add_argument("--retrieval-position-bins", type=int, default=8)
    parser.add_argument("--from-existing", action="store_true", help="Reuse raw CSV and only aggregate/plot.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / "cross_model_patch_raw.csv"

    if args.from_existing:
        raw = pd.read_csv(raw_path)
    else:
        specs = load_specs()
        rows = []
        for pair_index, pair in enumerate(parse_pairs(args.pairs)):
            donor = specs[pair[0]]
            recipient = specs[pair[1]]
            validate_pair(donor, recipient)
            print(f"cross-patching donor={donor.setting} recipient={recipient.setting}")
            rows.extend(
                run_pair_patch(
                    donor,
                    recipient,
                    pair_index=pair_index,
                    num_examples=args.num_examples,
                    batch_size=args.batch_size,
                    device=args.device,
                    seed_offset=args.seed_offset,
                    retrieval_max_bank_rows=args.retrieval_max_bank_rows,
                    retrieval_position_bins=args.retrieval_position_bins,
                )
            )
            gc.collect()
        raw = pd.DataFrame(rows)
        raw.to_csv(raw_path, index=False)

    summary = aggregate_results(raw)
    summary.to_csv(OUT_DIR / "cross_model_patch_summary.csv", index=False)
    make_figure(summary)
    print(f"wrote {raw_path}")
    print(f"wrote {OUT_DIR / 'cross_model_patch_summary.csv'}")
    print(f"wrote {FIG_DIR / 'cross_model_patch_pilot.png'}")


def parse_pairs(raw_pairs: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for raw in raw_pairs:
        if ":" not in raw:
            raise ValueError(f"Pair must be donor:recipient, got {raw!r}")
        donor, recipient = raw.split(":", 1)
        pairs.append((donor.strip(), recipient.strip()))
    return pairs


def load_specs() -> dict[str, SimpleNamespace]:
    sparse = pd.read_csv(SPARSE_SUMMARY_PATH)
    specs = {}
    for row in sparse.to_dict(orient="records"):
        key = f"b{int(row['bracket_tokens'])}"
        specs[key] = SimpleNamespace(
            setting=key,
            source=str(row["source"]),
            run_dir=ROOT / str(row["run_dir"]),
            bracket_tokens=int(row["bracket_tokens"]),
            seq_len=int(row["seq_len"]),
            noise_vocab=int(row["noise_vocab"]),
        )
    return specs


def validate_pair(donor: SimpleNamespace, recipient: SimpleNamespace) -> None:
    if donor.seq_len != recipient.seq_len:
        raise ValueError(f"seq_len mismatch: {donor.setting}={donor.seq_len}, {recipient.setting}={recipient.seq_len}")
    if donor.noise_vocab != recipient.noise_vocab:
        raise ValueError(
            f"noise_vocab mismatch: {donor.setting}={donor.noise_vocab}, {recipient.setting}={recipient.noise_vocab}"
        )
    donor_cfg = load_json(donor.run_dir / "config.json")
    recipient_cfg = load_json(recipient.run_dir / "config.json")
    for key in ["model_name", "model"]:
        if donor_cfg[key] != recipient_cfg[key]:
            raise ValueError(f"Model config mismatch for {donor.setting}->{recipient.setting}: {key}")


@torch.no_grad()
def run_pair_patch(
    donor: SimpleNamespace,
    recipient: SimpleNamespace,
    *,
    pair_index: int,
    num_examples: int,
    batch_size: int,
    device: str,
    seed_offset: int,
    retrieval_max_bank_rows: int,
    retrieval_position_bins: int,
) -> list[dict[str, object]]:
    recipient_cfg = load_json(recipient.run_dir / "config.json")
    donor_cfg = load_json(donor.run_dir / "config.json")
    task_name = task_name_from_run_config(recipient_cfg)
    task_kwargs = {k: v for k, v in recipient_cfg["task"].items() if k != "device"}
    sampler = build_sampler(task_name, task_kwargs, device="cpu", seed=int(recipient_cfg["seed"]) + seed_offset + pair_index)
    vocab_size = int(recipient.noise_vocab) + 2
    recipient_model = load_model(recipient.run_dir, recipient_cfg, vocab_size, device)
    donor_model = load_model(donor.run_dir, donor_cfg, vocab_size, device)

    layers = available_layers(recipient.run_dir)
    recipient_dirs = {layer: load_direction(recipient.run_dir, layer, "height").to(device) for layer in layers}
    donor_dirs = {layer: load_direction(donor.run_dir, layer, "height").to(device) for layer in layers}
    recipient_stats = {layer: projection_stats(recipient, layer, recipient_dirs[layer].cpu()) for layer in layers}
    donor_stats = {layer: projection_stats(donor, layer, donor_dirs[layer].cpu()) for layer in layers}
    donor_banks = {
        layer: build_retrieval_bank(
            donor,
            layer,
            max_rows=retrieval_max_bank_rows,
            position_bins=retrieval_position_bins,
            seed=setting_seed(f"bank:{donor.setting}:L{layer}", pair_index),
        )
        for layer in layers
    }

    rows = []
    seen = 0
    batch_id = 0
    while seen < num_examples:
        current = min(batch_size, num_examples - seen)
        batch = sampler.sample(current)
        tokens = batch.tokens.to(device)
        raw_labels = build_labels(task_name, batch_to_cpu(batch), sampler.config, max_prefix_len=None)
        labels = prepare_labels(raw_labels, make_row_view(recipient))
        labels = choose_one_target_per_example(labels, seed=setting_seed(f"{donor.setting}->{recipient.setting}", batch_id))
        if labels.empty:
            seen += current
            batch_id += 1
            continue

        positions = torch.tensor(labels["position"].to_numpy(dtype=int), device=device)
        examples = torch.tensor(labels["example_id"].to_numpy(dtype=int), device=device)
        target = torch.tensor(labels["target_token"].to_numpy(dtype=int), device=device)
        state = labels["forced_state"].to_numpy()
        close_token = int(recipient.noise_vocab)
        open_token = close_token + 1

        recipient_logits, recipient_traces = recipient_model(tokens, return_traces=True)
        donor_logits, donor_traces = donor_model(tokens, return_traces=True)
        rows.extend(
            metric_rows(
                donor,
                recipient,
                layer=-1,
                mode="recipient_baseline",
                logits=recipient_logits,
                examples=examples,
                positions=positions,
                target=target,
                state=state,
                close_token=close_token,
                open_token=open_token,
            )
        )
        rows.extend(
            metric_rows(
                donor,
                recipient,
                layer=-1,
                mode="donor_baseline",
                logits=donor_logits,
                examples=examples,
                positions=positions,
                target=target,
                state=state,
                close_token=close_token,
                open_token=open_token,
            )
        )

        for layer in layers:
            rec_h = recipient_traces["h"][layer]
            donor_h = donor_traces["h"][layer]
            selected_donor = donor_h[examples, positions]
            selected_recipient = rec_h[examples, positions]
            perm = torch.randperm(len(examples), device=device)
            bank = donor_banks[layer]
            retrieval_rng = np.random.default_rng(setting_seed(f"retrieval:{donor.setting}->{recipient.setting}:L{layer}", batch_id))
            retrieval_match_idx = retrieve_bank_indices(
                labels,
                bank,
                rng=retrieval_rng,
                seq_len=recipient.seq_len,
                position_bins=retrieval_position_bins,
                strategy="match",
            )
            retrieval_state_random_idx = retrieve_bank_indices(
                labels,
                bank,
                rng=retrieval_rng,
                seq_len=recipient.seq_len,
                position_bins=retrieval_position_bins,
                strategy="state_random",
            )
            retrieval_match = bank.hidden[retrieval_match_idx].to(device)
            retrieval_state_random = bank.hidden[retrieval_state_random_idx].to(device)

            patches = {
                "full_state_replace": selected_donor,
                "full_state_shuffle": selected_donor[perm],
                "recipient_self_patch": selected_recipient,
                "height_scalar_replace": height_scalar_patch_values(
                    selected_recipient,
                    selected_donor,
                    recipient_dirs[layer],
                    donor_dirs[layer],
                    recipient_stats[layer],
                    donor_stats[layer],
                ),
                "height_scalar_shuffle": height_scalar_patch_values(
                    selected_recipient,
                    selected_donor[perm],
                    recipient_dirs[layer],
                    donor_dirs[layer],
                    recipient_stats[layer],
                    donor_stats[layer],
                ),
                "retrieval_full_match": retrieval_match,
                "retrieval_full_state_random": retrieval_state_random,
                "retrieval_height_match": height_scalar_patch_values(
                    selected_recipient,
                    retrieval_match,
                    recipient_dirs[layer],
                    donor_dirs[layer],
                    recipient_stats[layer],
                    donor_stats[layer],
                ),
                "retrieval_height_state_random": height_scalar_patch_values(
                    selected_recipient,
                    retrieval_state_random,
                    recipient_dirs[layer],
                    donor_dirs[layer],
                    recipient_stats[layer],
                    donor_stats[layer],
                ),
            }
            for mode, patch_values in patches.items():
                patched_logits = continue_with_selected_patch(
                    recipient_model,
                    rec_h,
                    patch_layer=layer,
                    examples=examples,
                    positions=positions,
                    patch_values=patch_values,
                )
                rows.extend(
                    metric_rows(
                        donor,
                        recipient,
                        layer=layer,
                        mode=mode,
                        logits=patched_logits,
                        examples=examples,
                        positions=positions,
                        target=target,
                        state=state,
                        close_token=close_token,
                        open_token=open_token,
                    )
                )
                del patched_logits

        seen += current
        batch_id += 1
        del recipient_logits, donor_logits, recipient_traces, donor_traces
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return rows


def load_model(run_dir: Path, config: dict, vocab_size: int, device: str):
    spec = config["model"]
    model_kwargs = {k: v for k, v in spec.items() if k not in {"name", "state_kind"}}
    model = build_model(model_name=config["model_name"], vocab_size=vocab_size, **model_kwargs).to(device)
    checkpoint = torch.load(run_dir / "checkpoints" / "model_final.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def make_row_view(spec: SimpleNamespace):
    return type(
        "RowView",
        (),
        {
            "setting": spec.setting,
            "run_dir_abs": spec.run_dir,
            "noise_vocab": int(spec.noise_vocab),
            "total_length": int(spec.bracket_tokens),
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


def height_scalar_patch_values(
    recipient_values: torch.Tensor,
    donor_values: torch.Tensor,
    recipient_direction: torch.Tensor,
    donor_direction: torch.Tensor,
    recipient_stats: tuple[float, float],
    donor_stats: tuple[float, float],
) -> torch.Tensor:
    recipient_mean, recipient_std = recipient_stats
    donor_mean, donor_std = donor_stats
    donor_scalar = donor_values @ donor_direction
    donor_z = (donor_scalar - float(donor_mean)) / max(float(donor_std), 1e-8)
    desired_recipient_scalar = float(recipient_mean) + donor_z * max(float(recipient_std), 1e-8)
    current_recipient_scalar = recipient_values @ recipient_direction
    delta = desired_recipient_scalar - current_recipient_scalar
    return recipient_values + delta.unsqueeze(-1) * recipient_direction.view(1, -1)


def continue_with_selected_patch(
    model,
    layer_h: torch.Tensor,
    *,
    patch_layer: int,
    examples: torch.Tensor,
    positions: torch.Tensor,
    patch_values: torch.Tensor,
) -> torch.Tensor:
    h = layer_h.clone()
    h[examples, positions] = patch_values
    seq_len = h.shape[1]
    if patch_layer < int(model.num_layers) - 1:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=h.device, dtype=torch.bool), diagonal=1)
        for layer in model.layers[patch_layer + 1 :]:
            h = layer(h, src_mask=mask)
        h = model.final_norm(h)
    return model.output(h)


def metric_rows(
    donor: SimpleNamespace,
    recipient: SimpleNamespace,
    *,
    layer: int,
    mode: str,
    logits: torch.Tensor,
    examples: torch.Tensor,
    positions: torch.Tensor,
    target: torch.Tensor,
    state: np.ndarray,
    close_token: int,
    open_token: int,
) -> list[dict[str, object]]:
    selected = logits[examples, positions]
    pred = selected.argmax(dim=-1)
    correct = (pred == target).detach().cpu().numpy()
    logprob = F.log_softmax(selected, dim=-1)
    nll = (-logprob[torch.arange(len(target), device=target.device), target]).detach().cpu().numpy()
    margin = (selected[:, close_token] - selected[:, open_token]).detach().cpu().numpy()
    p_close = torch.sigmoid(selected[:, close_token] - selected[:, open_token]).detach().cpu().numpy()
    rows = []
    for split, mask in split_masks_for_state(state).items():
        if not mask.any():
            continue
        rows.append(
            {
                "donor": donor.setting,
                "recipient": recipient.setting,
                "donor_brackets": int(donor.bracket_tokens),
                "recipient_brackets": int(recipient.bracket_tokens),
                "seq_len": int(recipient.seq_len),
                "layer": int(layer),
                "mode": mode,
                "split": split,
                "n": int(mask.sum()),
                "accuracy": float(correct[mask].mean()),
                "target_nll": float(nll[mask].mean()),
                "mean_close_minus_open_margin": float(margin[mask].mean()),
                "mean_p_close_given_bracket": float(p_close[mask].mean()),
            }
        )
    return rows


def split_masks_for_state(state: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "all_dyck_targets": np.ones(len(state), dtype=bool),
        "forced": state != "free",
        "must_open": state == "must_open",
        "must_close": state == "must_close",
        "free": state == "free",
    }


def build_retrieval_bank(
    spec: SimpleNamespace,
    layer: int,
    *,
    max_rows: int,
    position_bins: int,
    seed: int,
) -> SimpleNamespace:
    labels = pd.read_parquet(spec.run_dir / "hidden_states" / "final" / "labels.parquet")
    labels = prepare_labels(labels, make_row_view(spec))
    mask = labels["has_target_label"].to_numpy(dtype=bool) & labels["target_is_dyck_position"].to_numpy(dtype=bool)
    idx = np.flatnonzero(mask)
    if max_rows > 0 and len(idx) > max_rows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_rows, replace=False))
    meta = labels.iloc[idx].reset_index(drop=True).copy()
    meta["height_key"] = meta["height"].round().astype(int)
    meta["position_bin"] = position_bin(meta["position"].to_numpy(dtype=int), spec.seq_len, position_bins)
    hidden_all = torch.load(spec.run_dir / "hidden_states" / "final" / f"layer_{layer}.pt", map_location="cpu").float()
    hidden = hidden_all[idx].contiguous()
    del hidden_all

    by_state_height_pos: dict[tuple[str, int, int], list[int]] = {}
    by_state_height: dict[tuple[str, int], list[int]] = {}
    by_state: dict[str, list[int]] = {}
    all_indices = list(range(len(meta)))
    for local_idx, row in enumerate(meta.itertuples(index=False)):
        state = str(row.forced_state)
        height = int(row.height_key)
        pos_bin = int(row.position_bin)
        by_state_height_pos.setdefault((state, height, pos_bin), []).append(local_idx)
        by_state_height.setdefault((state, height), []).append(local_idx)
        by_state.setdefault(state, []).append(local_idx)
    return SimpleNamespace(
        hidden=hidden,
        by_state_height_pos={key: np.asarray(value, dtype=np.int64) for key, value in by_state_height_pos.items()},
        by_state_height={key: np.asarray(value, dtype=np.int64) for key, value in by_state_height.items()},
        by_state={key: np.asarray(value, dtype=np.int64) for key, value in by_state.items()},
        all_indices=np.asarray(all_indices, dtype=np.int64),
    )


def retrieve_bank_indices(
    labels: pd.DataFrame,
    bank: SimpleNamespace,
    *,
    rng: np.random.Generator,
    seq_len: int,
    position_bins: int,
    strategy: str,
) -> torch.Tensor:
    pos_bins = position_bin(labels["position"].to_numpy(dtype=int), seq_len, position_bins)
    indices = []
    for row_idx, row in enumerate(labels.itertuples(index=False)):
        state = str(row.forced_state)
        height = int(round(float(row.height)))
        pos_bin = int(pos_bins[row_idx])
        if strategy == "match":
            candidates = bank.by_state_height_pos.get((state, height, pos_bin))
            if candidates is None or len(candidates) == 0:
                candidates = bank.by_state_height.get((state, height))
            if candidates is None or len(candidates) == 0:
                candidates = bank.by_state.get(state)
        elif strategy == "state_random":
            candidates = bank.by_state.get(state)
        else:
            raise ValueError(f"Unknown retrieval strategy={strategy!r}")
        if candidates is None or len(candidates) == 0:
            candidates = bank.all_indices
        indices.append(int(rng.choice(candidates)))
    return torch.as_tensor(indices, dtype=torch.long)


def position_bin(positions: np.ndarray, seq_len: int, bins: int) -> np.ndarray:
    if bins <= 1:
        return np.zeros_like(positions, dtype=int)
    return np.clip((positions.astype(float) / max(float(seq_len), 1.0) * bins).astype(int), 0, bins - 1)


def projection_stats(spec: SimpleNamespace, layer: int, direction: torch.Tensor) -> tuple[float, float]:
    labels = pd.read_parquet(spec.run_dir / "hidden_states" / "final" / "labels.parquet")
    labels = prepare_labels(labels, make_row_view(spec))
    mask = labels["has_target_label"].to_numpy(dtype=bool) & labels["target_is_dyck_position"].to_numpy(dtype=bool)
    idx = np.flatnonzero(mask)
    if len(idx) > 30_000:
        rng = np.random.default_rng(9_000 + int(layer) + int(spec.bracket_tokens))
        idx = np.sort(rng.choice(idx, size=30_000, replace=False))
    hidden = torch.load(spec.run_dir / "hidden_states" / "final" / f"layer_{layer}.pt", map_location="cpu").float()
    projection = hidden[idx] @ direction.float()
    mean = float(projection.mean().item())
    std = float(projection.std().item())
    return mean, std if std > 1e-8 else 1.0


def available_layers(run_dir: Path) -> list[int]:
    return sorted(int(path.stem.removeprefix("layer_")) for path in (run_dir / "hidden_states" / "final").glob("layer_*.pt"))


def load_direction(run_dir: Path, layer: int, target: str) -> torch.Tensor:
    direction = torch.load(run_dir / "probes" / "directions" / f"final_layer_{layer}_{target}.pt", map_location="cpu").float().reshape(-1)
    return direction / (direction.norm() + 1e-12)


def aggregate_results(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["donor", "recipient", "donor_brackets", "recipient_brackets", "seq_len", "layer", "mode", "split"]
    rows = []
    for group_keys, group in raw.groupby(keys, sort=False):
        weights = group["n"].to_numpy(dtype=float)
        weights = weights / max(float(weights.sum()), 1.0)
        row = {key: value for key, value in zip(keys, group_keys)}
        row["n"] = int(group["n"].sum())
        for metric in ["accuracy", "target_nll", "mean_close_minus_open_margin", "mean_p_close_given_bracket"]:
            row[metric] = float(np.sum(group[metric].to_numpy(dtype=float) * weights))
        rows.append(row)
    summary = pd.DataFrame(rows)

    baseline = summary[summary["mode"].eq("recipient_baseline")].copy()
    baseline = baseline.rename(columns={"accuracy": "recipient_baseline_accuracy", "target_nll": "recipient_baseline_nll"})
    baseline = baseline[["donor", "recipient", "split", "recipient_baseline_accuracy", "recipient_baseline_nll"]].drop_duplicates()
    donor = summary[summary["mode"].eq("donor_baseline")].copy()
    donor = donor.rename(columns={"accuracy": "donor_baseline_accuracy", "target_nll": "donor_baseline_nll"})
    donor = donor[["donor", "recipient", "split", "donor_baseline_accuracy", "donor_baseline_nll"]].drop_duplicates()
    summary = summary.merge(baseline, on=["donor", "recipient", "split"], how="left")
    summary = summary.merge(donor, on=["donor", "recipient", "split"], how="left")
    summary["delta_accuracy_vs_recipient"] = summary["accuracy"] - summary["recipient_baseline_accuracy"]
    summary["delta_nll_vs_recipient"] = summary["target_nll"] - summary["recipient_baseline_nll"]
    denom = summary["donor_baseline_accuracy"] - summary["recipient_baseline_accuracy"]
    summary["gap_closed"] = np.where(np.abs(denom) > 1e-8, summary["delta_accuracy_vs_recipient"] / denom, np.nan)
    return summary.sort_values(["donor", "recipient", "layer", "mode", "split"]).reset_index(drop=True)


def make_figure(summary: pd.DataFrame) -> None:
    focused = summary[
        summary["split"].eq("forced")
        & summary["mode"].isin(
            [
                "full_state_replace",
                "full_state_shuffle",
                "retrieval_full_match",
                "retrieval_full_state_random",
                "height_scalar_replace",
                "retrieval_height_match",
            ]
        )
    ].copy()
    if focused.empty:
        return
    focused["pair"] = focused["donor"] + "->" + focused["recipient"]
    pairs = list(dict.fromkeys(focused["pair"].tolist()))
    preferred_modes = [
        "full_state_replace",
        "full_state_shuffle",
        "retrieval_full_match",
        "retrieval_full_state_random",
        "height_scalar_replace",
        "retrieval_height_match",
    ]
    modes = [mode for mode in preferred_modes if mode in set(focused["mode"])]

    ncols = 3
    nrows = int(np.ceil(len(modes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.8 * nrows), constrained_layout=True)
    axes_flat = np.asarray(axes).reshape(-1)
    for ax, mode in zip(axes_flat, modes):
        sub = focused[focused["mode"].eq(mode)]
        heat = sub.pivot_table(index="pair", columns="layer", values="delta_accuracy_vs_recipient", aggfunc="mean").reindex(pairs)
        im = ax.imshow(heat.to_numpy(dtype=float), cmap="coolwarm", vmin=-0.9, vmax=0.9, aspect="auto")
        ax.set_title(f"{mode}: forced acc delta")
        ax.set_xticks(np.arange(len(heat.columns)))
        ax.set_xticklabels([str(c) for c in heat.columns])
        ax.set_yticks(np.arange(len(heat.index)))
        ax.set_yticklabels(heat.index)
        ax.set_xlabel("recipient layer patched")
        for yi in range(heat.shape[0]):
            for xi in range(heat.shape[1]):
                value = heat.iloc[yi, xi]
                if pd.notna(value):
                    ax.text(xi, yi, f"{value:+.2f}", ha="center", va="center", color="black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes_flat[len(modes) :]:
        ax.axis("off")
    fig.savefig(FIG_DIR / "cross_model_patch_pilot.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def setting_seed(name: str, offset: int) -> int:
    value = 2166136261
    for ch in name:
        value ^= ord(ch)
        value *= 16777619
        value &= 0xFFFFFFFF
    return int((value + offset) % (2**32 - 1))


if __name__ == "__main__":
    main()
