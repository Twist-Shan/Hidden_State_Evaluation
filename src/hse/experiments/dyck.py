from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import torch

from hse.analysis.compression import run_compression_probes
from hse.analysis.probes import load_probe_data, run_sufficient_statistic_probes
from hse.models import build_model
from hse.tasks.dyck import DyckConfig, DyckSampler
from hse.utils import evaluate_causal_lm, extract_hidden_states, save_json, train_causal_lm


DEFAULT_DYCK_TASKS = {
    "dyck_no_noise": {
        "dyck_pairs": 24,
        "total_length": 48,
        "seq_len": 48,
        "repeat_prob": 1.0,
        "n_tasks": 512,
        "prefix_probe_max_len": 7,
    },
    "dyck_50_noise": {
        "dyck_pairs": 10,
        "total_length": 20,
        "seq_len": 48,
        "repeat_prob": 0.5,
        "n_tasks": 512,
        "prefix_probe_max_len": 7,
    },
}

DEFAULT_DYCK_MODEL_SPECS = {
    "rnn": {"layers": 3, "emb_dim": 128, "hidden_dim": 256, "state_kind": "h"},
    "lstm": {"layers": 3, "emb_dim": 128, "hidden_dim": 128, "state_kind": "c"},
    "transformer": {
        "layers": 3,
        "emb_dim": 128,
        "hidden_dim": 128,
        "n_heads": 4,
        "ffn_dim": 512,
        "state_kind": "h",
    },
    "mamba": {
        "layers": 3,
        "emb_dim": 128,
        "hidden_dim": 128,
        "state_dim": 16,
        "expansion_factor": 2,
        "require_official_mamba": True,
        "state_kind": "h",
    },
    "mamba_like": {
        "layers": 3,
        "emb_dim": 128,
        "hidden_dim": 128,
        "state_dim": 16,
        "expansion_factor": 2,
        "state_kind": "h",
    },
}


def official_mamba_status() -> dict[str, str | bool]:
    installed = importlib.util.find_spec("mamba_ssm") is not None
    message = (
        "official mamba-ssm is available"
        if installed
        else (
            "official mamba-ssm is not installed. "
            "Use a Linux/WSL2 CUDA environment, then run: "
            "pip install causal-conv1d>=1.4.0 mamba-ssm --no-build-isolation"
        )
    )
    return {
        "installed": installed,
        "package": "mamba_ssm",
        "message": message,
    }


def resolve_dyck_model_specs(
    *,
    require_official_mamba: bool = True,
    fallback_to_mamba_like: bool = False,
) -> dict[str, dict]:
    status = official_mamba_status()
    model_specs = {
        "rnn": dict(DEFAULT_DYCK_MODEL_SPECS["rnn"]),
        "lstm": dict(DEFAULT_DYCK_MODEL_SPECS["lstm"]),
        "transformer": dict(DEFAULT_DYCK_MODEL_SPECS["transformer"]),
    }

    if status["installed"]:
        model_specs["mamba"] = dict(DEFAULT_DYCK_MODEL_SPECS["mamba"])
        return model_specs

    if fallback_to_mamba_like:
        model_specs["mamba_like"] = dict(DEFAULT_DYCK_MODEL_SPECS["mamba_like"])
        return model_specs

    if require_official_mamba:
        raise RuntimeError(str(status["message"]))

    return model_specs


def run_dyck_suite(
    *,
    task_name: str,
    seeds: list[int] | tuple[int, ...],
    training_steps: int,
    batch_size: int,
    extract_examples: int,
    extract_batch_size: int = 512,
    learning_rate: float = 3e-4,
    eval_every: int = 200,
    device: str | None = None,
    results_root: str | Path | None = None,
    require_official_mamba: bool = True,
    fallback_to_mamba_like: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if task_name not in DEFAULT_DYCK_TASKS:
        raise ValueError(f"Unknown Dyck task {task_name!r}. Expected one of: {sorted(DEFAULT_DYCK_TASKS)}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    task_kwargs = dict(DEFAULT_DYCK_TASKS[task_name])
    model_specs = resolve_dyck_model_specs(
        require_official_mamba=require_official_mamba,
        fallback_to_mamba_like=fallback_to_mamba_like,
    )
    results_root = Path(results_root or Path("results") / "notebooks" / "dyck" / task_name)
    results_root.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict] = []
    probe_rows: list[dict] = []

    for model_name, model_spec in model_specs.items():
        for seed in seeds:
            run_dir = results_root / f"{model_name}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            task_config = DyckConfig(**task_kwargs, device=device)
            sampler = DyckSampler(task_config, seed=seed)
            model_kwargs = {k: v for k, v in model_spec.items() if k != "state_kind"}

            torch.manual_seed(seed)
            model = build_model(model_name=model_name, vocab_size=sampler.vocab_size, **model_kwargs).to(device)

            save_json(
                {
                    "setting_name": task_name,
                    "task": {**task_kwargs, "device": device},
                    "model_name": model_name,
                    "model": model_spec,
                    "seed": seed,
                    "training_steps": training_steps,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "device": device,
                },
                run_dir / "config.json",
            )

            train_log = train_causal_lm(
                model=model,
                sampler=sampler,
                steps=training_steps,
                batch_size=batch_size,
                lr=learning_rate,
                run_dir=run_dir,
                eval_every=eval_every,
                device=device,
            )
            eval_metrics = evaluate_causal_lm(model=model, sampler=sampler, batch_size=batch_size, device=device)

            hidden, labels_df = extract_hidden_states(
                model=model,
                sampler=sampler,
                state_kind=model_spec["state_kind"],
                layer=-1,
                num_examples=extract_examples,
                batch_size=extract_batch_size,
                max_prefix_len=task_kwargs["prefix_probe_max_len"],
                device=device,
                run_dir=run_dir,
            )

            X, probe_labels = load_probe_data(run_dir / "hidden_states.pt", _labels_path(run_dir))
            probe_results = run_sufficient_statistic_probes(X, probe_labels, seed=seed)
            compression_table, compression_summary = run_compression_probes(X, probe_labels, seed=seed)

            probes_dir = run_dir / "probes"
            probes_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                "task_name": task_name,
                "model_name": model_name,
                "seed": seed,
                "left_r2": probe_results["left"]["r2"],
                "right_r2": probe_results["right"]["r2"],
                "height_r2": probe_results["height"]["r2"],
                "height_mae": probe_results["height"]["mae"],
                **probe_results["geometry"],
                **compression_summary,
            }
            save_json(summary, probes_dir / "summary.json")
            compression_table.to_csv(probes_dir / "compression_probe_rows.csv", index=False)

            run_rows.append(
                {
                    "task": task_name,
                    "model": model_name,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "loss": eval_metrics["loss"],
                    "accuracy": eval_metrics["accuracy"],
                    "dyck_accuracy": eval_metrics["dyck_accuracy"],
                    "hidden_rows": int(hidden.shape[0]),
                    "hidden_dim": int(hidden.shape[1]),
                    "label_rows": int(labels_df.shape[0]),
                }
            )
            probe_rows.append(summary)

            save_json({"train": train_log, "eval": eval_metrics}, run_dir / "metrics.json")

    runs_df = pd.DataFrame(run_rows)
    probe_df = pd.DataFrame(probe_rows)
    runs_df.to_csv(results_root / "runs.csv", index=False)
    probe_df.to_csv(results_root / "probe_summary.csv", index=False)
    return runs_df, probe_df


def _labels_path(run_dir: Path) -> Path:
    parquet_path = run_dir / "labels.parquet"
    if parquet_path.exists():
        return parquet_path
    csv_path = run_dir / "labels.csv"
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f"Missing labels.parquet or labels.csv under {run_dir}")
