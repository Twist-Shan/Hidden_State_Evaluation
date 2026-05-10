from __future__ import annotations

import numpy as np
import pandas as pd

from hse.analysis.probes.linear import (
    fit_logistic_probe,
    fit_ridge_probe,
    normalized_accuracy,
    normalized_r2,
)


RELEVANT_REGRESSION_TARGETS = ["left", "right", "height"]
RELEVANT_CLASSIFICATION_TARGETS = ["height_class", "left_right_class", "legal_next_class"]
IRRELEVANT_CLASSIFICATION_TARGETS = [
    "last_noise_token_class",
    "distant_noise_token_class",
    "noise_pattern_hash_class",
    "random_marker_class",
]


def run_compression_probes(X, labels: pd.DataFrame, *, seed: int = 0) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    for target in RELEVANT_REGRESSION_TARGETS:
        if target not in labels:
            continue
        out = fit_ridge_probe(X, labels[target].to_numpy(), seed=seed)
        rows.append(_row(target, "relevant", "ridge", out["r2"], normalized_r2(out["r2"])))

    for target in RELEVANT_CLASSIFICATION_TARGETS:
        if target not in labels or labels[target].nunique() <= 1:
            continue
        y = labels[target].to_numpy()
        out = fit_logistic_probe(X, y, seed=seed)
        rows.append(_row(target, "relevant", "logistic", out["accuracy"], normalized_accuracy(out["accuracy"], len(np.unique(y)))))

    for target in IRRELEVANT_CLASSIFICATION_TARGETS:
        if target not in labels or labels[target].nunique() <= 1:
            continue
        y = labels[target].to_numpy()
        out = fit_logistic_probe(X, y, seed=seed)
        rows.append(_row(target, "irrelevant", "logistic", out["accuracy"], normalized_accuracy(out["accuracy"], len(np.unique(y)))))

    df = pd.DataFrame(rows)
    relevant = df.loc[df.kind == "relevant", "normalized_score"].mean()
    irrelevant_decodability = df.loc[df.kind == "irrelevant", "normalized_score"].mean()
    summary = {
        "relevant_retention": float(relevant) if not np.isnan(relevant) else float("nan"),
        "irrelevant_decodability": float(irrelevant_decodability) if not np.isnan(irrelevant_decodability) else float("nan"),
        "irrelevant_forgetting": float(1.0 - irrelevant_decodability) if not np.isnan(irrelevant_decodability) else float("nan"),
    }
    return df, summary


def _row(target: str, kind: str, probe_type: str, raw_score: float, normalized_score: float) -> dict:
    return {
        "target": target,
        "kind": kind,
        "probe_type": probe_type,
        "raw_score": float(raw_score),
        "normalized_score": float(normalized_score),
    }
