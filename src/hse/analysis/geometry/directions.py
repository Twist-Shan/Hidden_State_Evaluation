from __future__ import annotations

import numpy as np

from hse.analysis.probes.linear import extract_linear_weight


def cosine(a, b, eps: float = 1e-12) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def dyck_direction_geometry(probe_results: dict) -> dict[str, float]:
    w_left = extract_linear_weight(probe_results["left"]["probe"])
    w_right = extract_linear_weight(probe_results["right"]["probe"])
    w_height = extract_linear_weight(probe_results["height"]["probe"])
    return {
        "cos_height_left_minus_right": cosine(w_height, w_left - w_right),
        "cos_left_right": cosine(w_left, w_right),
    }
