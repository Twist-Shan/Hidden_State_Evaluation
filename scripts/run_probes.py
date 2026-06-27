import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import torch

from hse.analysis.compression import run_compression_probes
from hse.analysis.probes import extract_linear_weight, load_probe_data, run_sufficient_statistic_probes
from hse.utils import load_json, save_json


@dataclass
class FeatureSpec:
    path: Path
    labels: Path
    checkpoint: str
    layer: int | None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to extracted hidden states.")
    parser.add_argument("--labels", default=None, help="Path to sufficient-statistics labels.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None, help="Subsample at most this many aligned rows per layer.")
    parser.add_argument("--max-classes", type=int, default=200, help="Skip logistic targets with more than this many classes.")
    args = parser.parse_args()

    features = Path(args.features)
    specs = discover_feature_specs(features, labels_override=Path(args.labels) if args.labels else None)
    if not specs:
        raise FileNotFoundError(f"No feature tensors found under {features}")
    run_config = load_run_config(features)
    regression_targets, classification_targets = probe_targets(run_config)
    out_dir = infer_probe_dir(features)
    directions_dir = out_dir / "directions"
    out_dir.mkdir(parents=True, exist_ok=True)
    directions_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    compression_tables = []
    for spec in specs:
        X, label_df = load_probe_data(spec.path, spec.labels)
        X, label_df = subsample_rows(X, label_df, max_rows=args.max_rows, seed=args.seed)
        probe_results = run_sufficient_statistic_probes(
            X,
            label_df,
            seed=args.seed,
            regression_targets=regression_targets,
            classification_targets=classification_targets,
            max_classes=args.max_classes,
        )
        compression_rows, compression_summary = run_compression_probes(
            X,
            label_df,
            seed=args.seed,
            max_classes=args.max_classes,
        )
        if not compression_rows.empty:
            compression_rows.insert(0, "checkpoint", spec.checkpoint)
            compression_rows.insert(1, "layer", spec.layer if spec.layer is not None else -1)
            compression_tables.append(compression_rows)

        row = {
            "checkpoint": spec.checkpoint,
            "layer": spec.layer if spec.layer is not None else -1,
            **summarize_probe_results(probe_results),
            **compression_summary,
        }
        rows.append(row)
        save_probe_directions(probe_results, directions_dir, checkpoint=spec.checkpoint, layer=row["layer"])

    layerwise_new = pd.DataFrame(rows).sort_values(["checkpoint", "layer"]).reset_index(drop=True)
    layerwise = merge_existing_rows(out_dir / "layerwise_probe.csv", layerwise_new)
    layerwise.to_csv(out_dir / "layerwise_probe.csv", index=False)
    layerwise.to_csv(out_dir / "checkpoint_probe.csv", index=False)
    if compression_tables:
        compression_new = pd.concat(compression_tables, ignore_index=True)
        merge_existing_rows(out_dir / "compression_probe_rows.csv", compression_new).to_csv(
            out_dir / "compression_probe_rows.csv",
            index=False,
        )

    summary = summarize_layerwise_table(layerwise)
    save_json(summary, out_dir / "summary.json")
    print(f"saved probe summaries to {out_dir}")


def discover_feature_specs(features: Path, *, labels_override: Path | None) -> list[FeatureSpec]:
    if features.is_file():
        labels = labels_override or find_labels(features.parent)
        return [FeatureSpec(features, labels, checkpoint=infer_checkpoint_name(features), layer=infer_layer(features))]
    layer_files = sorted(features.glob("layer_*.pt"))
    if layer_files:
        labels = labels_override or find_labels(features)
        checkpoint = checkpoint_name_from_dir(features)
        return [FeatureSpec(path, labels, checkpoint=checkpoint, layer=infer_layer(path)) for path in layer_files]
    specs = []
    for child in sorted(features.iterdir()):
        if child.is_dir():
            specs.extend(discover_feature_specs(child, labels_override=labels_override))
    return specs


def find_labels(directory: Path) -> Path:
    parquet_path = directory / "labels.parquet"
    if parquet_path.exists():
        return parquet_path
    csv_path = directory / "labels.csv"
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f"Missing labels.parquet or labels.csv under {directory}")


def infer_layer(path: Path) -> int | None:
    if path.stem.startswith("layer_"):
        return int(path.stem.removeprefix("layer_"))
    return None


def infer_checkpoint_name(path: Path) -> str:
    if path.parent.parent.name == "hidden_states":
        return path.parent.name
    return "final"


def checkpoint_name_from_dir(directory: Path) -> str:
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        return str(load_json(metadata_path).get("checkpoint", directory.name))
    return directory.name if directory.parent.name == "hidden_states" else "final"


def infer_run_dir(features: Path) -> Path:
    if features.is_file() and features.name == "hidden_states.pt":
        return features.parent
    if features.is_file() and features.parent.parent.name == "hidden_states":
        return features.parents[2]
    if features.is_dir() and features.parent.name == "hidden_states":
        return features.parents[1]
    if features.is_dir() and features.name == "hidden_states":
        return features.parent
    return features.parent


def infer_probe_dir(features: Path) -> Path:
    return infer_run_dir(features) / "probes"


def load_run_config(features: Path) -> dict:
    config_path = infer_run_dir(features) / "config.json"
    return load_json(config_path) if config_path.exists() else {}


def probe_targets(config: dict) -> tuple[list[str] | None, list[str] | None]:
    probe_cfg = config.get("analysis", {}).get("probes", {})
    return probe_cfg.get("ridge_targets"), probe_cfg.get("logistic_targets")


def summarize_probe_results(probe_results: dict) -> dict:
    summary = {}
    for target, result in probe_results.items():
        if target == "geometry":
            summary.update(result)
        elif "r2" in result:
            summary[f"{target}_r2"] = float(result["r2"])
            summary[f"{target}_mae"] = float(result["mae"])
        elif "accuracy" in result:
            summary[f"{target}_accuracy"] = float(result["accuracy"])
    return summary


def subsample_rows(X, labels: pd.DataFrame, *, max_rows: int | None, seed: int):
    if max_rows is None or max_rows <= 0 or len(labels) <= max_rows:
        return X, labels
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(labels), size=max_rows, replace=False))
    return X[idx], labels.iloc[idx].reset_index(drop=True)


def save_probe_directions(probe_results: dict, out_dir: Path, *, checkpoint: str, layer: int) -> None:
    for target, result in probe_results.items():
        if target == "geometry" or "probe" not in result:
            continue
        weight = extract_linear_weight(result["probe"])
        if weight.size == 0:
            continue
        torch.save(torch.as_tensor(weight), out_dir / f"{checkpoint}_layer_{layer}_{target}.pt")


def summarize_layerwise_table(layerwise: pd.DataFrame) -> dict:
    summary = {"num_probe_rows": int(len(layerwise))}
    if "height_r2" in layerwise:
        best_idx = layerwise["height_r2"].astype(float).idxmax()
        best_row = layerwise.loc[best_idx].to_dict()
        summary.update({f"best_{key}": jsonable(value) for key, value in best_row.items()})
    if len(layerwise) == 1:
        summary.update({key: jsonable(value) for key, value in layerwise.iloc[0].to_dict().items()})
    return summary


def merge_existing_rows(path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        existing = pd.read_csv(path)
        if "checkpoint" in existing and "checkpoint" in new_rows:
            existing = existing.loc[~existing["checkpoint"].isin(new_rows["checkpoint"].unique())]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    sort_cols = [col for col in ["checkpoint", "layer", "target"] if col in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols).reset_index(drop=True)
    return combined


def jsonable(value):
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    main()
