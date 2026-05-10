from __future__ import annotations

from pathlib import Path

DEFAULT_MODEL_SPECS = {
    "rnn": {"layers": 3, "emb_dim": 128, "hidden_dim": 256, "state_kind": "h"},
    "lstm": {"layers": 3, "emb_dim": 128, "hidden_dim": 128, "state_kind": "c"},
    "transformer": {"layers": 3, "emb_dim": 128, "hidden_dim": 128, "n_heads": 4, "ffn_dim": 512, "state_kind": "h"},
    "mamba": {
        "layers": 3,
        "emb_dim": 128,
        "hidden_dim": 128,
        "state_dim": 16,
        "expansion_factor": 2,
        "require_official_mamba": True,
        "state_kind": "h",
    },
}


def load_yaml(path: str | Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_specs_from_config(config: dict) -> list[dict]:
    if "models" in config and config["models"]:
        return list(config["models"])
    return [{"name": name, **spec} for name, spec in DEFAULT_MODEL_SPECS.items()]
