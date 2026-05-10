import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument("--stage", default="all", choices=["all", "train", "extract", "probe", "geometry"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    cmd = [sys.executable, str(ROOT / "scripts" / "train_model.py"), "--config", str(config_path), "--seed", str(args.seed)]
    if args.model:
        cmd += ["--model", args.model]
    if args.device:
        cmd += ["--device", args.device]
    if args.steps:
        cmd += ["--steps", str(args.steps)]
    if args.stage in {"all", "train"}:
        subprocess.run(cmd, check=True)

    if args.stage == "train":
        return
    print("For extract/probe stages, run the per-run scripts with the generated results directory.")


if __name__ == "__main__":
    main()
