from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_labels(labels: pd.DataFrame, path_prefix: str | Path) -> Path:
    """Save labels as parquet when available, otherwise CSV."""
    path_prefix = Path(path_prefix)
    parquet_path = path_prefix.with_suffix(".parquet")
    csv_path = path_prefix.with_suffix(".csv")
    try:
        labels.to_parquet(parquet_path)
        return parquet_path
    except Exception:
        labels.to_csv(csv_path, index=False)
        return csv_path


def load_labels(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            csv_fallback = path.with_suffix(".csv")
            if csv_fallback.exists():
                return pd.read_csv(csv_fallback)
            raise
    return pd.read_csv(path)
