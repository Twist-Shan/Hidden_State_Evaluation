import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from hse.models import build_model
from hse.tasks.registry import build_sampler, task_name_from_experiment_config
from hse.utils.config import load_yaml, model_specs_from_config
from hse.utils.io import save_json
from hse.utils.training import evaluate_causal_lm, train_causal_lm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument("--model", default=None, help="Optional single model name to run.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    exp_name = config["experiment"]["name"]
    task_name = task_name_from_experiment_config(config)
    sampler = build_sampler(task_name, config["task"], device="cpu", seed=args.seed)
    training = config.get("training", {})
    steps = args.steps or int(training.get("steps", 10000))
    batch_size = args.batch_size or int(training.get("batch_size", 128))
    lr = float(training.get("learning_rate", 3e-4))
    eval_every = int(training.get("eval_every", 200))
    checkpoint_steps = [int(step) for step in training.get("checkpoint_steps", [])]

    specs = model_specs_from_config(config)
    if args.model is not None:
        specs = [spec for spec in specs if spec["name"] == args.model]
    if not specs:
        raise ValueError(f"No model specs matched --model={args.model!r}")

    for spec in specs:
        model_name = spec["name"]
        model_kwargs = {k: v for k, v in spec.items() if k not in {"name", "state_kind"}}
        torch.manual_seed(args.seed)
        model = build_model(model_name=model_name, vocab_size=sampler.vocab_size, **model_kwargs)
        run_dir = ROOT / "results" / exp_name / f"{model_name}_seed{args.seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            {
                "setting_name": exp_name,
                "task_name": task_name,
                "task": {**config["task"], "device": "cpu"},
                "model_name": model_name,
                "model": spec,
                "seed": args.seed,
                "training_steps": steps,
                "batch_size": batch_size,
                "learning_rate": lr,
                "eval_every": eval_every,
                "checkpoint_steps": checkpoint_steps,
                "analysis": config.get("analysis", {}),
                "device": args.device,
            },
            run_dir / "config.json",
        )
        log = train_causal_lm(
            model=model,
            sampler=sampler,
            steps=steps,
            batch_size=batch_size,
            lr=lr,
            eval_every=eval_every,
            run_dir=run_dir,
            checkpoint_steps=checkpoint_steps,
            device=args.device,
        )
        metrics = evaluate_causal_lm(model=model, sampler=sampler, batch_size=batch_size, device=args.device)
        save_json({"train": log, "eval": metrics}, run_dir / "metrics.json")
        print(f"saved {model_name} seed={args.seed} to {run_dir}")


if __name__ == "__main__":
    main()
