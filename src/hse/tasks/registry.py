from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Callable

import torch

from hse.tasks.dyck import DyckConfig, DyckSampler
from hse.tasks.dyck.labels import build_prefix_labels as build_dyck_labels
from hse.tasks.dyck_k import DyckKConfig, DyckKSampler
from hse.tasks.dyck_k.labels import build_prefix_labels as build_dyck_k_labels
from hse.tasks.shuffle_dyck import ShuffleDyckConfig, ShuffleDyckSampler
from hse.tasks.shuffle_dyck.labels import build_prefix_labels as build_shuffle_dyck_labels


TASK_REGISTRY: dict[str, dict[str, Any]] = {
    "dyck": {
        "config": DyckConfig,
        "sampler": DyckSampler,
        "labels": build_dyck_labels,
    },
    "shuffle_dyck": {
        "config": ShuffleDyckConfig,
        "sampler": ShuffleDyckSampler,
        "labels": build_shuffle_dyck_labels,
    },
    "dyck_k": {
        "config": DyckKConfig,
        "sampler": DyckKSampler,
        "labels": build_dyck_k_labels,
    },
}


def task_name_from_experiment_config(config: dict) -> str:
    return str(config.get("experiment", {}).get("task", "dyck"))


def task_name_from_run_config(config: dict) -> str:
    return str(config.get("task_name") or config.get("experiment", {}).get("task") or "dyck")


def build_task_config(task_name: str, task_kwargs: dict, *, device: str):
    entry = _entry(task_name)
    kwargs = {**task_kwargs, "device": device}
    return entry["config"](**kwargs)


def build_sampler(task_name: str, task_kwargs: dict, *, device: str, seed: int | None = None):
    task_config = build_task_config(task_name, task_kwargs, device=device)
    return entry_sampler(task_name)(task_config, seed=seed)


def build_labels(task_name: str, batch, task_config, *, max_prefix_len: int | None = None):
    return entry_label_builder(task_name)(batch, config=task_config, max_prefix_len=max_prefix_len)


def batch_to_cpu(batch):
    if not is_dataclass(batch):
        raise TypeError(f"Expected a dataclass batch, got {type(batch)!r}")
    values = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        values[field.name] = value
    return type(batch)(**values)


def entry_sampler(task_name: str) -> Callable:
    return _entry(task_name)["sampler"]


def entry_label_builder(task_name: str) -> Callable:
    return _entry(task_name)["labels"]


def _entry(task_name: str) -> dict[str, Any]:
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task {task_name!r}. Expected one of: {sorted(TASK_REGISTRY)}")
    return TASK_REGISTRY[task_name]
