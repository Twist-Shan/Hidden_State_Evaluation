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


def run_sufficient_statistic_probes(X, labels: pd.DataFrame, *, seed: int = 0) -> dict:
    results = {}
    for target in ["left", "right", "height"]:
        results[target] = fit_ridge_probe(X, labels[target].to_numpy(), seed=seed)
    for target in ["height_class", "left_right_class", "legal_next_class"]:
        if target in labels and labels[target].nunique() > 1:
            results[target] = fit_logistic_probe(X, labels[target].to_numpy(), seed=seed)
    results["geometry"] = dyck_direction_geometry(results)
    return results
