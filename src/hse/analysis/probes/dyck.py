from __future__ import annotations

import pandas as pd
import torch

from hse.analysis.geometry.directions import dyck_direction_geometry
from hse.analysis.probes.linear import fit_logistic_probe, fit_ridge_probe
from hse.utils.labels_io import load_labels


def load_probe_data(hidden_path, labels_path):
    hidden = torch.load(hidden_path, map_location="cpu")
    labels = load_labels(labels_path)
    return hidden.numpy(), labels


DEFAULT_REGRESSION_TARGETS = ["left", "right", "height"]
DEFAULT_CLASSIFICATION_TARGETS = ["height_class", "left_right_class", "legal_next_class"]


def run_sufficient_statistic_probes(
    X,
    labels: pd.DataFrame,
    *,
    seed: int = 0,
    regression_targets: list[str] | tuple[str, ...] | None = None,
    classification_targets: list[str] | tuple[str, ...] | None = None,
    max_classes: int | None = None,
) -> dict:
    results = {}
    for target in regression_targets or DEFAULT_REGRESSION_TARGETS:
        if target not in labels:
            continue
        results[target] = fit_ridge_probe(X, labels[target].to_numpy(), seed=seed)
    for target in classification_targets or DEFAULT_CLASSIFICATION_TARGETS:
        if target in labels and labels[target].nunique() > 1:
            if max_classes is not None and labels[target].nunique() > max_classes:
                continue
            results[target] = fit_logistic_probe(X, labels[target].to_numpy(), seed=seed)
    results["geometry"] = dyck_direction_geometry(results) if {"left", "right", "height"}.issubset(results) else {}
    return results
