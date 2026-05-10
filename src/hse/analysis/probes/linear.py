from __future__ import annotations

import numpy as np
import os
import torch

try:  # sklearn/scipy can be unavailable or ABI-mismatched in lightweight envs.
    if os.environ.get("HSE_USE_SKLEARN", "0") != "1":
        raise ImportError("Using torch/numpy probe fallback by default.")
    from sklearn.linear_model import LogisticRegressionCV, RidgeCV
    from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    LogisticRegressionCV = RidgeCV = None
    accuracy_score = mean_absolute_error = r2_score = None
    train_test_split = make_pipeline = StandardScaler = None


RIDGE_ALPHAS = np.logspace(-4, 4, 17)


def fit_ridge_probe(X, y, *, test_size: float = 0.25, seed: int = 0) -> dict:
    if RidgeCV is None:
        return _fit_ridge_probe_fallback(X, y, test_size=test_size, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    probe = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    probe.fit(X_train, y_train)
    pred = probe.predict(X_test)
    return {
        "probe": probe,
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
    }


def fit_logistic_probe(X, y, *, test_size: float = 0.25, seed: int = 0) -> dict:
    if LogisticRegressionCV is None:
        return _fit_logistic_probe_fallback(X, y, test_size=test_size, seed=seed)
    values, counts = np.unique(y, return_counts=True)
    if len(values) < 2 or counts.min() < 2:
        return {"probe": None, "accuracy": float("nan")}
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    _, train_counts = np.unique(y_train, return_counts=True)
    if len(train_counts) < 2 or train_counts.min() < 2:
        return {"probe": None, "accuracy": float("nan")}
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(Cs=8, cv=min(3, int(train_counts.min())), max_iter=2000, n_jobs=-1),
    )
    probe.fit(X_train, y_train)
    pred = probe.predict(X_test)
    return {"probe": probe, "accuracy": float(accuracy_score(y_test, pred))}


def extract_linear_weight(probe) -> np.ndarray:
    if isinstance(probe, dict) and "coef" in probe:
        return np.asarray(probe["coef"])
    if probe is None:
        return np.array([])
    estimator = probe.steps[-1][1]
    coef = estimator.coef_
    if coef.ndim == 2 and coef.shape[0] == 1:
        coef = coef[0]
    return np.asarray(coef)


def normalized_r2(score: float) -> float:
    return float(np.clip(score, 0.0, 1.0))


def normalized_accuracy(acc: float, n_classes: int) -> float:
    if np.isnan(acc):
        return float("nan")
    chance = 1.0 / max(n_classes, 1)
    return float(np.clip((acc - chance) / (1.0 - chance + 1e-12), 0.0, 1.0))


def _split_indices(n: int, test_size: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_size)))
    return idx[n_test:], idx[:n_test]


def _standardize(train: np.ndarray, test: np.ndarray):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (train - mean) / std, (test - mean) / std, mean, std


def _fit_ridge_probe_fallback(X, y, *, test_size: float, seed: int) -> dict:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    train_idx, test_idx = _split_indices(len(X), test_size, seed)
    X_train, X_test, mean, std = _standardize(X[train_idx], X[test_idx])
    y_train = y[train_idx]
    y_test = y[test_idx]
    X_aug = np.concatenate([X_train, np.ones((len(X_train), 1))], axis=1)
    alpha = 1.0
    eye = np.eye(X_aug.shape[1])
    eye[-1, -1] = 0.0
    coef_aug = np.linalg.solve(X_aug.T @ X_aug + alpha * eye, X_aug.T @ y_train)
    pred = np.concatenate([X_test, np.ones((len(X_test), 1))], axis=1) @ coef_aug
    ss_res = float(((y_test - pred) ** 2).sum())
    ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    mae = float(np.abs(y_test - pred).mean())
    coef = coef_aug[:-1] / std.squeeze()
    return {"probe": {"coef": coef}, "r2": float(r2), "mae": mae}


def _fit_logistic_probe_fallback(X, y, *, test_size: float, seed: int) -> dict:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    classes, y_encoded = np.unique(y, return_inverse=True)
    if len(classes) < 2:
        return {"probe": None, "accuracy": float("nan")}
    train_idx, test_idx = _split_indices(len(X), test_size, seed)
    X_train, X_test, mean, std = _standardize(X[train_idx], X[test_idx])
    y_train = torch.tensor(y_encoded[train_idx], dtype=torch.long)
    y_test = y_encoded[test_idx]
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    model = torch.nn.Linear(X_train_t.shape[1], len(classes))
    opt = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=80)
    loss_fn = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(model(X_train_t), y_train)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        pred = model(X_test_t).argmax(dim=-1).numpy()
    acc = float((pred == y_test).mean())
    coef = model.weight.detach().numpy()
    return {"probe": {"coef": coef}, "accuracy": acc}
