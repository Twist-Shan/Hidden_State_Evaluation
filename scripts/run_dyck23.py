import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hse.experiments.dyck23 import run_dyck23_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and probe Dyck-(2,3) CFG next-token models.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", nargs="*", default=None, help="Subset of rnn lstm transformer mamba.")
    parser.add_argument("--steps", type=int, default=15_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--train-examples", type=int, default=10_000)
    parser.add_argument("--test-examples", type=int, default=2_000)
    parser.add_argument("--probe-examples", type=int, default=1_024)
    parser.add_argument("--max-probe-rows", type=int, default=20_000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--results-root", default=str(ROOT / "results" / "dyck23_cfg_next_token"))
    args = parser.parse_args()

    run_dyck23_suite(
        seed=args.seed,
        models=args.models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_examples=args.train_examples,
        test_examples=args.test_examples,
        probe_examples=args.probe_examples,
        steps=args.steps,
        max_probe_rows=args.max_probe_rows,
        device=args.device,
        results_root=args.results_root,
    )


if __name__ == "__main__":
    main()
