from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_tensor(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)


def load_tensor(path: str | Path, map_location: str | torch.device = "cpu") -> torch.Tensor:
    return torch.load(path, map_location=map_location)
