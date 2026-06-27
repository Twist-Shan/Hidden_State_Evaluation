import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hse.utils.config import load_yaml, model_specs_from_config


def _config_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _module_main_cmd(module: str, script_name: str, args: list[str]) -> list[str]:
    launcher = (
        "import sys; "
        f"sys.argv = ['{script_name}'] + sys.argv[1:]; "
        f"import {module} as _script; "
        "_script.main()"
    )
    return [sys.executable, "-c", launcher, *args]


def _official_mamba_available() -> bool:
    return importlib.util.find_spec("mamba_ssm") is not None


def _select_models(config: dict, requested_model: str | None) -> list[dict]:
    specs = model_specs_from_config(config)
    if requested_model is not None:
        specs = [spec for spec in specs if spec["name"] == requested_model]
    if not specs:
        raise ValueError(f"No model specs matched --model={requested_model!r}")
    return specs


def _filter_unavailable_models(specs: list[dict], requested_model: str | None) -> list[dict]:
    available = []
    for spec in specs:
        if spec["name"] == "mamba" and spec.get("require_official_mamba", False) and not _official_mamba_available():
            if requested_model == "mamba":
                raise RuntimeError(
                    "Official Mamba is unavailable in this environment. "
                    "Use --model rnn/lstm/transformer, or switch to a Linux/CUDA environment with mamba-ssm installed."
                )
            print("Skipping model 'mamba' because mamba-ssm is not installed in this environment.")
            continue
        available.append(spec)
    if not available:
        raise RuntimeError("No runnable models remain after filtering unavailable dependencies.")
    return available


def _select_seeds(config: dict, requested_seed: int | None) -> list[int]:
    if requested_seed is not None:
        return [requested_seed]
    return list(config.get("experiment", {}).get("seeds", [0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument("--stage", default="all", choices=["all", "train", "extract", "probe", "geometry"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Run a single seed. Defaults to experiment.seeds from config.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-examples", type=int, default=4096)
    parser.add_argument("--extract-batch-size", type=int, default=512)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to extract/probe. Defaults to analysis.extract_checkpoints or final.")
    parser.add_argument("--extract-layers", default=None, help="Layer list such as -1, 0,1,2, or all.")
    parser.add_argument("--extract-positions", default=None, choices=["all", "prefix", "dyck", "final"])
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--probe-max-rows", type=int, default=None)
    parser.add_argument("--probe-max-classes", type=int, default=200)
    args = parser.parse_args()

    config_path = _config_path(args.config)
    config = load_yaml(config_path)
    exp_name = config["experiment"]["name"]
    seeds = _select_seeds(config, args.seed)
    specs = _filter_unavailable_models(_select_models(config, args.model), args.model)

    for seed in seeds:
        for spec in specs:
            model_name = spec["name"]
            run_dir = ROOT / "results" / exp_name / f"{model_name}_seed{seed}"

            if args.stage in {"all", "train"}:
                train_args = [
                    "--config",
                    str(config_path),
                    "--seed",
                    str(seed),
                    "--model",
                    model_name,
                ]
                if args.device:
                    train_args += ["--device", args.device]
                if args.steps is not None:
                    train_args += ["--steps", str(args.steps)]
                if args.batch_size is not None:
                    train_args += ["--batch-size", str(args.batch_size)]
                _run(_module_main_cmd("scripts.train_model", "train_model.py", train_args))

            if args.stage == "train":
                continue

            if not run_dir.exists():
                raise FileNotFoundError(
                    f"Missing run directory {run_dir}. Run the train stage first or choose an existing results directory."
                )

            if args.stage in {"all", "extract"}:
                for checkpoint in _select_extract_checkpoints(config, args.checkpoint):
                    extract_args = [
                        "--run",
                        str(run_dir),
                        "--checkpoint",
                        checkpoint,
                        "--extract-layers",
                        _extract_layers(config, args.extract_layers),
                        "--extract-positions",
                        _extract_positions(config, args.extract_positions),
                        "--num-examples",
                        str(args.num_examples),
                        "--batch-size",
                        str(args.extract_batch_size),
                    ]
                    if args.device:
                        extract_args += ["--device", args.device]
                    _run(_module_main_cmd("scripts.extract_hidden_states", "extract_hidden_states.py", extract_args))

            if args.stage in {"all", "probe"}:
                for checkpoint in _select_extract_checkpoints(config, args.checkpoint):
                    _run(
                        _probe_cmd(
                            _features_path(run_dir, checkpoint),
                            args.probe_seed,
                            args.probe_max_rows,
                            args.probe_max_classes,
                        )
                    )

            if args.stage in {"all", "geometry"}:
                _run(
                    _module_main_cmd(
                        "scripts.analyze_geometry",
                        "analyze_geometry.py",
                        [
                        "--probe-dir",
                        str(run_dir / "probes"),
                        ],
                    )
                )


def _select_extract_checkpoints(config: dict, requested_checkpoint: str | None) -> list[str]:
    if requested_checkpoint is not None:
        return [str(requested_checkpoint)]
    analysis = config.get("analysis", {})
    checkpoints = analysis.get("extract_checkpoints", "final")
    if checkpoints is None:
        checkpoints = ["final"]
    elif isinstance(checkpoints, (list, tuple)):
        pass
    elif checkpoints in {"all", "training_checkpoint_steps"}:
        checkpoints = [*config.get("training", {}).get("checkpoint_steps", []), "final"]
    elif isinstance(checkpoints, (str, int)):
        checkpoints = [checkpoints]
    seen = set()
    out = []
    for checkpoint in checkpoints:
        checkpoint_str = str(checkpoint)
        if checkpoint_str not in seen:
            out.append(checkpoint_str)
            seen.add(checkpoint_str)
    return out


def _extract_layers(config: dict, requested_layers: str | None) -> str:
    value = requested_layers if requested_layers is not None else config.get("analysis", {}).get("extract_layers", -1)
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def _extract_positions(config: dict, requested_positions: str | None) -> str:
    return str(requested_positions or config.get("analysis", {}).get("extract_positions", "prefix"))


def _features_path(run_dir: Path, checkpoint: str) -> Path:
    checkpoint_name = _checkpoint_name(checkpoint)
    canonical = run_dir / "hidden_states" / checkpoint_name
    if canonical.exists():
        return canonical
    if checkpoint_name == "final":
        return run_dir / "hidden_states.pt"
    return canonical


def _checkpoint_name(checkpoint: str) -> str:
    if checkpoint == "final":
        return "final"
    if checkpoint == "best":
        return "best"
    return f"step_{int(checkpoint)}"


def _probe_cmd(
    features: Path,
    seed: int,
    max_rows: int | None,
    max_classes: int | None,
) -> list[str]:
    args = [
        "--features",
        str(features),
        "--seed",
        str(seed),
    ]
    if max_rows is not None:
        args += ["--max-rows", str(max_rows)]
    if max_classes is not None:
        args += ["--max-classes", str(max_classes)]
    return _module_main_cmd("scripts.run_probes", "run_probes.py", args)


if __name__ == "__main__":
    main()
