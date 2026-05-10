import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hse.utils import load_yaml, model_specs_from_config


def _config_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


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
    parser.add_argument("--probe-seed", type=int, default=0)
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
                train_cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "train_model.py"),
                    "--config",
                    str(config_path),
                    "--seed",
                    str(seed),
                    "--model",
                    model_name,
                ]
                if args.device:
                    train_cmd += ["--device", args.device]
                if args.steps is not None:
                    train_cmd += ["--steps", str(args.steps)]
                if args.batch_size is not None:
                    train_cmd += ["--batch-size", str(args.batch_size)]
                _run(train_cmd)

            if args.stage == "train":
                continue

            if not run_dir.exists():
                raise FileNotFoundError(
                    f"Missing run directory {run_dir}. Run the train stage first or choose an existing results directory."
                )

            if args.stage in {"all", "extract"}:
                extract_cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "extract_hidden_states.py"),
                    "--run",
                    str(run_dir),
                    "--num-examples",
                    str(args.num_examples),
                    "--batch-size",
                    str(args.extract_batch_size),
                ]
                if args.device:
                    extract_cmd += ["--device", args.device]
                _run(extract_cmd)

            if args.stage in {"all", "probe"}:
                _run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "run_probes.py"),
                        "--features",
                        str(run_dir / "hidden_states.pt"),
                        "--seed",
                        str(args.probe_seed),
                    ]
                )

            if args.stage in {"all", "geometry"}:
                _run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "analyze_geometry.py"),
                        "--probe-dir",
                        str(run_dir / "probes"),
                    ]
                )


if __name__ == "__main__":
    main()
