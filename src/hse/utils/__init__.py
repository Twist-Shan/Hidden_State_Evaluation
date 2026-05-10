"""Shared config, training, extraction, and IO helpers.

This module intentionally avoids eager heavy imports so lightweight CLI entry
points such as ``scripts/run_pipeline.py --help`` can work before optional
runtime dependencies are installed.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "DEFAULT_MODEL_SPECS",
    "evaluate_causal_lm",
    "extract_hidden_states",
    "flatten_states_for_labels",
    "load_json",
    "load_labels",
    "load_tensor",
    "load_yaml",
    "model_specs_from_config",
    "save_json",
    "save_labels",
    "save_tensor",
    "train_causal_lm",
]

_LAZY_EXPORTS = {
    "DEFAULT_MODEL_SPECS": (".config", "DEFAULT_MODEL_SPECS"),
    "load_yaml": (".config", "load_yaml"),
    "model_specs_from_config": (".config", "model_specs_from_config"),
    "load_json": (".io", "load_json"),
    "load_tensor": (".io", "load_tensor"),
    "save_json": (".io", "save_json"),
    "save_tensor": (".io", "save_tensor"),
    "load_labels": (".labels_io", "load_labels"),
    "save_labels": (".labels_io", "save_labels"),
    "extract_hidden_states": (".extraction", "extract_hidden_states"),
    "flatten_states_for_labels": (".extraction", "flatten_states_for_labels"),
    "evaluate_causal_lm": (".training", "evaluate_causal_lm"),
    "train_causal_lm": (".training", "train_causal_lm"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
