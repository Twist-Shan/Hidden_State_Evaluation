from __future__ import annotations

import argparse
import gc
import json
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

from hse.models import build_model
from hse.tasks.registry import batch_to_cpu, build_labels, build_sampler, task_name_from_run_config
from hse.utils import load_json
from scripts.task_a_extra_probes import prepare_labels


SPARSE_SUMMARY_PATH = ROOT / "results" / "dyck_counter_sparse_supervision_ablation" / "summary.csv"
OUT_DIR = ROOT / "results" / "dyck_counter_task_a_countscope_patching"
FIG_DIR = ROOT / "figures" / "dyck_counter_task_a_countscope_patching"
DEFAULT_SETTINGS = ["b20", "b48", "b100"]
PATCH_MODES = ["target_self", "source_state", "source_state_shuffle", "source_same_target", "mean_forced", "zero"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", nargs="*", default=DEFAULT_SETTINGS)
    parser.add_argument("--pairs-per-setting", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=240)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed-offset", type=int, default=70_000)
    parser.add_argument("--from-existing", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / "countscope_online_patching_raw.csv"

    if args.from_existing:
        raw = pd.read_csv(raw_path)
    else:
        specs = load_specs(args.settings)
        rows = []
        for setting_index, spec in enumerate(specs):
            print(f"online patching {spec.setting}: brackets={spec.bracket_tokens}, seq_len={spec.seq_len}")
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
    summary.to_csv(OUT_DIR / "countscope_online_patching_summary.csv", index=False)
    make_figures(summary)
    write_readme(summary)
    print(f"wrote {raw_path}")
    print(f"wrote {OUT_DIR / 'countscope_online_patching_summary.csv'}")
    print(f"wrote figures to {FIG_DIR}")


def load_specs(settings: list[str]) -> list[SimpleNamespace]:
    sparse = pd.read_csv(SPARSE_SUMMARY_PATH)
    specs = {}
    for row in sparse.to_dict(orient="records"):
        key = str(row["setting"])
        specs[key] = SimpleNamespace(
            setting=key,
            source=str(row["source"]),
            run_dir=ROOT / str(row["run_dir"]),
            bracket_tokens=int(row["bracket_tokens"]),
            seq_len=int(row["seq_len"]),
            total_length=int(row["bracket_tokens"]),
            noise_vocab=int(row["noise_vocab"]),
            forced_model_acc=float(row["forced_model_acc"]),
            height_r2=float(row["height_r2"]),
        )
    missing = [setting for setting in settings if setting not in specs]
    if missing:
        raise FileNotFoundError(f"Missing sparse settings: {missing}")
    return [specs[setting] for setting in settings]


@torch.no_grad()
def run_setting(
    spec: SimpleNamespace,
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
    rows: list[dict[str, object]] = []
    collected_local = 0
    collected_future = 0
    batch_index = 0
    rng = np.random.default_rng(setting_seed(spec.setting, 13))

    while batch_index < max_batches and (collected_local < pairs_per_setting or collected_future < pairs_per_setting):
        target_batch = sampler.sample(batch_size)
        source_batch = sampler.sample(batch_size)
        target_tokens = target_batch.tokens.to(device)
        source_tokens = source_batch.tokens.to(device)
        target_labels = label_batch(task_name, target_batch, sampler.config, spec)
        source_labels = label_batch(task_name, source_batch, sampler.config, spec)

        target_logits, target_traces = model(target_tokens, return_traces=True)
        _source_logits, source_traces = model(source_tokens, return_traces=True)

        if collected_local < pairs_per_setting:
            local_pairs = build_local_pairs(
                target_labels,
                source_labels,
                max_pairs=pairs_per_setting - collected_local,
                rng=rng,
            )
            if local_pairs:
                rows.extend(
                    run_pair_set(
                        experiment="local_interchange",
                        spec=spec,
                        model=model,
                        target_logits=target_logits,
                        target_traces=target_traces,
                        source_traces=source_traces,
                        pairs=local_pairs,
                        layers=layers,
                        batch_index=batch_index,
                        rng=rng,
                    )
                )
                collected_local += len(local_pairs)

        if collected_future < pairs_per_setting:
            future_pairs = build_future_pairs(
                target_labels,
                source_labels,
                max_pairs=pairs_per_setting - collected_future,
                total_length=spec.total_length,
                close_token=spec.noise_vocab,
                open_token=spec.noise_vocab + 1,
                rng=rng,
            )
            if future_pairs:
                rows.extend(
                    run_pair_set(
                        experiment="future_continued",
                        spec=spec,
                        model=model,
                        target_logits=target_logits,
                        target_traces=target_traces,
                        source_traces=source_traces,
                        pairs=future_pairs,
                        layers=layers,
                        batch_index=batch_index,
                        rng=rng,
                    )
                )
                collected_future += len(future_pairs)

        batch_index += 1
        del target_logits, target_traces, source_traces
        gc.collect()

    print(f"  collected local={collected_local}, future={collected_future}, batches={batch_index}")
    del model
    return rows


def load_model(run_dir: Path, config: dict, vocab_size: int, device: str):
    spec = config["model"]
    model_kwargs = {key: value for key, value in spec.items() if key not in {"name", "state_kind"}}
    model = build_model(model_name=config["model_name"], vocab_size=vocab_size, **model_kwargs).to(device)
    checkpoint = torch.load(run_dir / "checkpoints" / "model_final.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def label_batch(task_name: str, batch, task_config, spec: SimpleNamespace) -> pd.DataFrame:
    labels = build_labels(task_name, batch_to_cpu(batch), task_config, max_prefix_len=None)
    labels = prepare_labels(labels, make_row_view(spec))
    return labels.sort_values(["example_id", "position"]).reset_index(drop=True)


def make_row_view(spec: SimpleNamespace):
    return type(
        "RowView",
        (),
        {
            "setting": spec.setting,
            "run_dir_abs": spec.run_dir,
            "noise_vocab": int(spec.noise_vocab),
            "total_length": int(spec.total_length),
        },
    )()


def build_local_pairs(
    target_labels: pd.DataFrame,
    source_labels: pd.DataFrame,
    *,
    max_pairs: int,
    rng: np.random.Generator,
) -> list[dict[str, int]]:
    target_candidates = forced_target_rows(target_labels)
    source_candidates = forced_target_rows(source_labels)
    if target_candidates.empty or source_candidates.empty:
        return []

    chosen_targets = one_row_per_example(target_candidates, rng=rng)
    rng.shuffle(chosen_targets)
    pairs = []
    for target_idx in chosen_targets:
        target = target_labels.loc[target_idx]
        opposite = source_candidates[source_candidates["target_token"].ne(target.target_token)]
        same = source_candidates[source_candidates["target_token"].eq(target.target_token)]
        if opposite.empty or same.empty:
            continue
        source_opp = source_labels.loc[int(rng.choice(opposite.index.to_numpy()))]
        source_same = source_labels.loc[int(rng.choice(same.index.to_numpy()))]
        pairs.append(
            {
                "target_example": int(target.example_id),
                "patch_position": int(target.position),
                "eval_position": int(target.position),
                "target_token": int(target.target_token),
                "hypothesis_token": int(source_opp.target_token),
                "source_example": int(source_opp.example_id),
                "source_position": int(source_opp.position),
                "same_source_example": int(source_same.example_id),
                "same_source_position": int(source_same.position),
                "same_source_token": int(source_same.target_token),
                "target_height": float(target.height),
                "source_height": float(source_opp.height),
                "eval_distance": 0,
            }
        )
        if len(pairs) >= max_pairs:
            break
    return pairs


def build_future_pairs(
    target_labels: pd.DataFrame,
    source_labels: pd.DataFrame,
    *,
    max_pairs: int,
    total_length: int,
    close_token: int,
    open_token: int,
    rng: np.random.Generator,
) -> list[dict[str, int]]:
    source_dyck = source_labels[source_labels["is_dyck_position"].to_numpy(dtype=bool)].copy()
    if source_dyck.empty:
        return []
    source_indices = source_dyck.index.to_numpy()
    pairs = []
    example_ids = list(target_labels["example_id"].drop_duplicates().to_numpy(dtype=int))
    rng.shuffle(example_ids)
    for example_id in example_ids:
        group = target_labels[target_labels["example_id"].eq(example_id)].copy()
        patch_candidates = group[group["is_dyck_position"].to_numpy(dtype=bool)]
        if patch_candidates.empty:
            continue
        patch_indices = patch_candidates.index.to_numpy().copy()
        rng.shuffle(patch_indices)
        found = False
        for patch_idx in patch_indices:
            patch_row = target_labels.loc[patch_idx]
            future = group[
                group["position"].gt(patch_row.position)
                & group["has_target_label"]
                & group["target_is_dyck_position"]
                & group["forced_state"].ne("free")
            ]
            if future.empty:
                continue
            source_idx = int(rng.choice(source_indices))
            source_row = source_labels.loc[source_idx]
            possible = []
            for future_idx, eval_row in future.iterrows():
                continued_token = continued_forced_token(
                    source_row=source_row,
                    patch_row=patch_row,
                    eval_row=eval_row,
                    total_length=total_length,
                    close_token=close_token,
                    open_token=open_token,
                )
                if continued_token is not None and int(continued_token) != int(eval_row.target_token):
                    possible.append((future_idx, int(continued_token)))
            if not possible:
                continue
            future_idx, continued_token = possible[int(rng.integers(0, len(possible)))]
            eval_row = target_labels.loc[future_idx]
            same_source = source_row
            same_candidates = source_dyck[np.isclose(source_dyck["height"], patch_row.height)]
            if not same_candidates.empty:
                same_source = source_labels.loc[int(rng.choice(same_candidates.index.to_numpy()))]
            pairs.append(
                {
                    "target_example": int(example_id),
                    "patch_position": int(patch_row.position),
                    "eval_position": int(eval_row.position),
                    "target_token": int(eval_row.target_token),
                    "hypothesis_token": int(continued_token),
                    "source_example": int(source_row.example_id),
                    "source_position": int(source_row.position),
                    "same_source_example": int(same_source.example_id),
                    "same_source_position": int(same_source.position),
                    "same_source_token": int(eval_row.target_token),
                    "target_height": float(eval_row.height),
                    "source_height": float(source_row.height),
                    "eval_distance": int(eval_row.position) - int(patch_row.position),
                }
            )
            found = True
            break
        if found and len(pairs) >= max_pairs:
            break
    return pairs


def forced_target_rows(labels: pd.DataFrame) -> pd.DataFrame:
    return labels[
        labels["has_target_label"].to_numpy(dtype=bool)
        & labels["target_is_dyck_position"].to_numpy(dtype=bool)
        & labels["forced_state"].ne("free")
    ].copy()


def one_row_per_example(labels: pd.DataFrame, *, rng: np.random.Generator) -> np.ndarray:
    chosen = []
    for _, group in labels.groupby("example_id", sort=False):
        chosen.append(int(rng.choice(group.index.to_numpy())))
    return np.asarray(chosen, dtype=np.int64)


def continued_forced_token(
    *,
    source_row,
    patch_row,
    eval_row,
    total_length: int,
    close_token: int,
    open_token: int,
) -> int | None:
    left = int(source_row.left) + int(eval_row.left) - int(patch_row.left)
    right = int(source_row.right) + int(eval_row.right) - int(patch_row.right)
    dyck_seen = int(source_row.dyck_seen) + int(eval_row.dyck_seen) - int(patch_row.dyck_seen)
    max_opens = total_length // 2
    if left < 0 or right < 0 or left > max_opens or right > max_opens or dyck_seen < 0 or dyck_seen >= total_length:
        return None
    height = left - right
    remaining_dyck = total_length - dyck_seen
    remaining_opens = max_opens - left
    if height <= 0:
        return open_token
    if remaining_opens <= 0 or remaining_dyck <= height:
        return close_token
    return None


@torch.no_grad()
def run_pair_set(
    *,
    experiment: str,
    spec: SimpleNamespace,
    model,
    target_logits: torch.Tensor,
    target_traces: dict[str, torch.Tensor],
    source_traces: dict[str, torch.Tensor],
    pairs: list[dict[str, int]],
    layers: list[int],
    batch_index: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    device = target_logits.device
    target_examples = torch.tensor([pair["target_example"] for pair in pairs], device=device, dtype=torch.long)
    patch_positions = torch.tensor([pair["patch_position"] for pair in pairs], device=device, dtype=torch.long)
    eval_positions = torch.tensor([pair["eval_position"] for pair in pairs], device=device, dtype=torch.long)
    source_examples = torch.tensor([pair["source_example"] for pair in pairs], device=device, dtype=torch.long)
    source_positions = torch.tensor([pair["source_position"] for pair in pairs], device=device, dtype=torch.long)
    same_source_examples = torch.tensor([pair["same_source_example"] for pair in pairs], device=device, dtype=torch.long)
    same_source_positions = torch.tensor([pair["same_source_position"] for pair in pairs], device=device, dtype=torch.long)
    target_tokens = torch.tensor([pair["target_token"] for pair in pairs], device=device, dtype=torch.long)
    hypothesis_tokens = torch.tensor([pair["hypothesis_token"] for pair in pairs], device=device, dtype=torch.long)
    same_source_tokens = torch.tensor([pair["same_source_token"] for pair in pairs], device=device, dtype=torch.long)

    baseline_logits = target_logits[target_examples, eval_positions]
    baseline_probs = bracket_probs(baseline_logits, close_token=spec.noise_vocab, open_token=spec.noise_vocab + 1)
    rows: list[dict[str, object]] = []
    for layer in layers:
        target_h = target_traces["h"][layer]
        source_h = source_traces["h"][layer]
        source_values = source_h[source_examples, source_positions]
        same_values = source_h[same_source_examples, same_source_positions]
        target_values = target_h[target_examples, patch_positions]
        perm = torch.as_tensor(rng.permutation(len(pairs)), device=device, dtype=torch.long)
        mean_value = source_values.mean(dim=0, keepdim=True).expand_as(source_values)
        zero_value = torch.zeros_like(source_values)
        modes = {
            "target_self": (target_values, hypothesis_tokens),
            "source_state": (source_values, hypothesis_tokens),
            "source_state_shuffle": (source_values[perm], hypothesis_tokens[perm]),
            "source_same_target": (same_values, same_source_tokens),
            "mean_forced": (mean_value, hypothesis_tokens),
            "zero": (zero_value, hypothesis_tokens),
        }
        for mode, (patch_values, mode_hypothesis_tokens) in modes.items():
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
                    experiment=experiment,
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


def continue_with_selected_patch(
    model,
    layer_h: torch.Tensor,
    *,
    patch_layer: int,
    examples: torch.Tensor,
    patch_positions: torch.Tensor,
    patch_values: torch.Tensor,
) -> torch.Tensor:
    h = layer_h.clone()
    h[examples, patch_positions] = patch_values
    seq_len = h.shape[1]
    if patch_layer < int(model.num_layers) - 1:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=h.device, dtype=torch.bool), diagonal=1)
        for layer in model.layers[patch_layer + 1 :]:
            h = layer(h, src_mask=mask)
        h = model.final_norm(h)
    return model.output(h)


def metric_rows(
    *,
    experiment: str,
    spec: SimpleNamespace,
    layer: int,
    mode: str,
    batch_index: int,
    pairs: list[dict[str, int]],
    baseline_logits: torch.Tensor,
    baseline_probs: torch.Tensor,
    patched_logits: torch.Tensor,
    target_tokens: torch.Tensor,
    hypothesis_tokens: torch.Tensor,
) -> list[dict[str, object]]:
    close_token = int(spec.noise_vocab)
    open_token = close_token + 1
    patched_probs = bracket_probs(patched_logits, close_token=close_token, open_token=open_token)
    target_idx = token_to_bracket_index(target_tokens, close_token=close_token)
    hypothesis_idx = token_to_bracket_index(hypothesis_tokens, close_token=close_token)
    baseline_p_target = baseline_probs[torch.arange(len(target_idx), device=target_idx.device), target_idx]
    baseline_p_hyp = baseline_probs[torch.arange(len(hypothesis_idx), device=hypothesis_idx.device), hypothesis_idx]
    patched_p_target = patched_probs[torch.arange(len(target_idx), device=target_idx.device), target_idx]
    patched_p_hyp = patched_probs[torch.arange(len(hypothesis_idx), device=hypothesis_idx.device), hypothesis_idx]
    baseline_pred = torch.where(baseline_probs[:, 0] >= baseline_probs[:, 1], close_token, open_token)
    patched_pred = torch.where(patched_probs[:, 0] >= patched_probs[:, 1], close_token, open_token)
    baseline_full_probs = torch.softmax(baseline_logits, dim=-1)
    patched_full_probs = torch.softmax(patched_logits, dim=-1)
    row_ids = torch.arange(len(target_tokens), device=target_tokens.device)
    baseline_full_p_target = baseline_full_probs[row_ids, target_tokens]
    baseline_full_p_hyp = baseline_full_probs[row_ids, hypothesis_tokens]
    patched_full_p_target = patched_full_probs[row_ids, target_tokens]
    patched_full_p_hyp = patched_full_probs[row_ids, hypothesis_tokens]
    baseline_full_pred = baseline_logits.argmax(dim=-1)
    patched_full_pred = patched_logits.argmax(dim=-1)
    valid_ci = hypothesis_tokens.ne(target_tokens)
    ci = 0.5 * ((patched_p_hyp - baseline_p_hyp) + (baseline_p_target - patched_p_target))
    full_ci = 0.5 * (
        (patched_full_p_hyp - baseline_full_p_hyp) + (baseline_full_p_target - patched_full_p_target)
    )
    split_masks = {
        "all": torch.ones_like(valid_ci, dtype=torch.bool),
        "target_open": target_tokens.eq(open_token),
        "target_close": target_tokens.eq(close_token),
        "ci_valid": valid_ci,
    }
    rows = []
    eval_distances = np.asarray([pair["eval_distance"] for pair in pairs], dtype=float)
    source_heights = np.asarray([pair["source_height"] for pair in pairs], dtype=float)
    target_heights = np.asarray([pair["target_height"] for pair in pairs], dtype=float)
    for split, mask_t in split_masks.items():
        mask = mask_t.detach().cpu().numpy().astype(bool)
        if not mask.any():
            continue
        rows.append(
            {
                "experiment": experiment,
                "setting": spec.setting,
                "source": spec.source,
                "bracket_tokens": int(spec.bracket_tokens),
                "seq_len": int(spec.seq_len),
                "forced_model_acc": float(spec.forced_model_acc),
                "height_r2": float(spec.height_r2),
                "layer": int(layer),
                "mode": mode,
                "split": split,
                "batch_index": int(batch_index),
                "n": int(mask.sum()),
                "mean_ci": safe_mean(ci, mask),
                "mean_full_vocab_ci": safe_mean(full_ci, mask),
                "mean_delta_p_hypothesis": safe_mean(patched_p_hyp - baseline_p_hyp, mask),
                "mean_delta_p_target": safe_mean(patched_p_target - baseline_p_target, mask),
                "mean_delta_full_p_hypothesis": safe_mean(patched_full_p_hyp - baseline_full_p_hyp, mask),
                "mean_delta_full_p_target": safe_mean(patched_full_p_target - baseline_full_p_target, mask),
                "baseline_target_acc": safe_mean(baseline_pred.eq(target_tokens).float(), mask),
                "patched_target_acc": safe_mean(patched_pred.eq(target_tokens).float(), mask),
                "patched_hypothesis_acc": safe_mean(patched_pred.eq(hypothesis_tokens).float(), mask),
                "baseline_full_target_acc": safe_mean(baseline_full_pred.eq(target_tokens).float(), mask),
                "patched_full_target_acc": safe_mean(patched_full_pred.eq(target_tokens).float(), mask),
                "patched_full_hypothesis_acc": safe_mean(patched_full_pred.eq(hypothesis_tokens).float(), mask),
                "mean_p_target_baseline": safe_mean(baseline_p_target, mask),
                "mean_p_target_patched": safe_mean(patched_p_target, mask),
                "mean_p_hypothesis_baseline": safe_mean(baseline_p_hyp, mask),
                "mean_p_hypothesis_patched": safe_mean(patched_p_hyp, mask),
                "mean_full_p_target_baseline": safe_mean(baseline_full_p_target, mask),
                "mean_full_p_target_patched": safe_mean(patched_full_p_target, mask),
                "mean_full_p_hypothesis_baseline": safe_mean(baseline_full_p_hyp, mask),
                "mean_full_p_hypothesis_patched": safe_mean(patched_full_p_hyp, mask),
                "mean_eval_distance": float(np.mean(eval_distances[mask])),
                "mean_source_height": float(np.mean(source_heights[mask])),
                "mean_target_height": float(np.mean(target_heights[mask])),
            }
        )
    return rows


def bracket_probs(logits: torch.Tensor, *, close_token: int, open_token: int) -> torch.Tensor:
    bracket_logits = torch.stack([logits[:, close_token], logits[:, open_token]], dim=1)
    return torch.softmax(bracket_logits, dim=1)


def token_to_bracket_index(tokens: torch.Tensor, *, close_token: int) -> torch.Tensor:
    return torch.where(tokens.eq(close_token), torch.zeros_like(tokens), torch.ones_like(tokens))


def safe_mean(values: torch.Tensor, mask: np.ndarray) -> float:
    arr = values.detach().cpu().numpy()
    return float(np.mean(arr[mask]))


def available_layers(run_dir: Path) -> list[int]:
    return sorted(int(path.stem.removeprefix("layer_")) for path in (run_dir / "hidden_states" / "final").glob("layer_*.pt"))


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["experiment", "setting", "source", "bracket_tokens", "seq_len", "layer", "mode", "split"]
    rows = []
    if raw.empty:
        return pd.DataFrame()
    for group_keys, group in raw.groupby(keys, sort=False):
        weights = group["n"].to_numpy(dtype=float)
        weights = weights / max(float(weights.sum()), 1.0)
        row = {key: value for key, value in zip(keys, group_keys)}
        row["n"] = int(group["n"].sum())
        row["forced_model_acc"] = float(group["forced_model_acc"].iloc[0])
        row["height_r2"] = float(group["height_r2"].iloc[0])
        for metric in [
            "mean_ci",
            "mean_full_vocab_ci",
            "mean_delta_p_hypothesis",
            "mean_delta_p_target",
            "mean_delta_full_p_hypothesis",
            "mean_delta_full_p_target",
            "baseline_target_acc",
            "patched_target_acc",
            "patched_hypothesis_acc",
            "baseline_full_target_acc",
            "patched_full_target_acc",
            "patched_full_hypothesis_acc",
            "mean_p_target_baseline",
            "mean_p_target_patched",
            "mean_p_hypothesis_baseline",
            "mean_p_hypothesis_patched",
            "mean_full_p_target_baseline",
            "mean_full_p_target_patched",
            "mean_full_p_hypothesis_baseline",
            "mean_full_p_hypothesis_patched",
            "mean_eval_distance",
            "mean_source_height",
            "mean_target_height",
        ]:
            row[metric] = float(np.sum(group[metric].to_numpy(dtype=float) * weights))
        rows.append(row)
    summary = pd.DataFrame(rows)
    self_patch = summary[summary["mode"].eq("target_self")][
        ["experiment", "setting", "layer", "split", "mean_ci", "mean_full_vocab_ci", "patched_target_acc", "patched_full_target_acc"]
    ].rename(
        columns={
            "mean_ci": "self_patch_ci",
            "mean_full_vocab_ci": "self_patch_full_vocab_ci",
            "patched_target_acc": "self_patch_target_acc",
            "patched_full_target_acc": "self_patch_full_target_acc",
        }
    )
    summary = summary.merge(self_patch, on=["experiment", "setting", "layer", "split"], how="left")
    summary["ci_minus_self"] = summary["mean_ci"] - summary["self_patch_ci"]
    summary["full_vocab_ci_minus_self"] = summary["mean_full_vocab_ci"] - summary["self_patch_full_vocab_ci"]
    summary["target_acc_delta_vs_self"] = summary["patched_target_acc"] - summary["self_patch_target_acc"]
    summary["full_target_acc_delta_vs_self"] = summary["patched_full_target_acc"] - summary["self_patch_full_target_acc"]
    return summary.sort_values(["experiment", "setting", "layer", "mode", "split"]).reset_index(drop=True)


def make_figures(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    plot_ci(summary, experiment="local_interchange", path=FIG_DIR / "local_interchange_ci.png")
    plot_ci(summary, experiment="future_continued", path=FIG_DIR / "future_continued_ci.png")
    plot_controls(
        summary,
        path=FIG_DIR / "patching_controls_summary.png",
        metric="ci_minus_self",
        ylabel="best-layer bracket-normalized CI minus self-patch",
    )
    plot_controls(
        summary,
        path=FIG_DIR / "patching_full_vocab_controls_summary.png",
        metric="full_vocab_ci_minus_self",
        ylabel="best-layer full-vocab CI minus self-patch",
    )


def plot_ci(summary: pd.DataFrame, *, experiment: str, path: Path) -> None:
    data = summary[
        summary["experiment"].eq(experiment)
        & summary["split"].eq("ci_valid")
        & summary["mode"].isin(["source_state", "source_state_shuffle", "mean_forced", "zero"])
    ].copy()
    if data.empty:
        return
    settings = list(dict.fromkeys(data["setting"].tolist()))
    fig, axes = plt.subplots(1, len(settings), figsize=(5.2 * len(settings), 4), sharey=True)
    if len(settings) == 1:
        axes = [axes]
    for ax, setting in zip(axes, settings):
        sub = data[data["setting"].eq(setting)]
        for mode in ["source_state", "source_state_shuffle", "mean_forced", "zero"]:
            mode_df = sub[sub["mode"].eq(mode)].sort_values("layer")
            if not mode_df.empty:
                ax.plot(mode_df["layer"], mode_df["ci_minus_self"], marker="o", label=mode)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(setting)
        ax.set_xlabel("patched layer")
        ax.set_ylabel("CI minus self-patch")
        ax.grid(alpha=0.25)
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle(f"{experiment}: source-hypothesis causal influence")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_controls(summary: pd.DataFrame, *, path: Path, metric: str, ylabel: str) -> None:
    data = summary[
        summary["split"].eq("ci_valid")
        & summary["mode"].isin(["source_state", "source_state_shuffle", "source_same_target", "mean_forced", "zero"])
    ].copy()
    if data.empty:
        return
    best = (
        data.sort_values(["experiment", "setting", "mode", "ci_minus_self"], ascending=[True, True, True, False])
        .groupby(["experiment", "setting", "mode"], as_index=False)
        .first()
    )
    best["label"] = best["experiment"] + "\n" + best["setting"]
    labels = list(dict.fromkeys(best["label"].tolist()))
    modes = ["source_state", "source_state_shuffle", "source_same_target", "mean_forced", "zero"]
    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(labels)), 5))
    x = np.arange(len(labels))
    width = 0.15
    for i, mode in enumerate(modes):
        values = []
        for label in labels:
            row = best[best["label"].eq(label) & best["mode"].eq(mode)]
            values.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + (i - 2) * width, values, width=width, label=mode)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_readme(summary: pd.DataFrame) -> None:
    if summary.empty:
        text = "No patching rows were produced.\n"
    else:
        focus = summary[
            summary["split"].eq("ci_valid")
            & summary["mode"].isin(["source_state", "source_state_shuffle", "mean_forced", "zero"])
        ].copy()
        top = focus.sort_values("ci_minus_self", ascending=False).head(12)
        text = (
            "# Task A CountScope-style Online Patching\n\n"
            "Definitions:\n"
            "- `local_interchange`: patch a source forced-state activation into a target forced-state position and decode the same position.\n"
            "- `future_continued`: patch a source bracket activation into an earlier target bracket position, continue later layers, and evaluate a later forced target position whose continued-counting hypothesis differs from the true target.\n"
            "- `mean_ci`: CI = 0.5 * [(P(hyp|patched)-P(hyp|target)) + (P(target|target)-P(target|patched))], using close/open-normalized bracket probabilities.\n"
            "- `ci_minus_self`: subtracts the self-patch control at the same setting/layer/split.\n\n"
            "Top effects:\n\n"
            + top.to_string(index=False)
            + "\n"
        )
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def setting_seed(setting: str, salt: int) -> int:
    return (sum((i + 1) * ord(ch) for i, ch in enumerate(setting)) + 104729 * salt) % (2**32 - 1)


if __name__ == "__main__":
    main()
