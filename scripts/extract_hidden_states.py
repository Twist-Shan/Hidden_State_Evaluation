import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from hse.models import build_model
from hse.tasks.registry import build_sampler, task_name_from_run_config
from hse.utils import extract_hidden_states, load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Path to a trained run directory.")
    parser.add_argument("--state-kind", default=None, help="Override hidden-state kind, e.g. h or c.")
    parser.add_argument("--checkpoint", default="final", help="Checkpoint to load: final, best, a step number, or a .pt path.")
    parser.add_argument("--extract-layers", default=None, help="Layer list such as -1, 0,1,2, or all. Defaults to final layer.")
    parser.add_argument("--extract-positions", default=None, choices=["all", "prefix", "dyck", "final"])
    parser.add_argument("--num-examples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_dir = Path(args.run)
    config = load_json(run_dir / "config.json")
    task_name = task_name_from_run_config(config)
    task_kwargs = {k: v for k, v in config["task"].items() if k != "device"}
    sampler = build_sampler(task_name, task_kwargs, device="cpu", seed=int(config["seed"]) + 10_000)
    spec = config["model"]
    model_name = config["model_name"]
    model_kwargs = {k: v for k, v in spec.items() if k not in {"name", "state_kind"}}
    model = build_model(model_name=model_name, vocab_size=sampler.vocab_size, **model_kwargs).to(args.device)
    checkpoint_path, checkpoint_name = resolve_checkpoint(run_dir, args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(ckpt["model"])
    state_kind = args.state_kind or spec.get("state_kind", "h")
    extract_layers = args.extract_layers or config.get("analysis", {}).get("extract_layers", -1)
    extract_positions = args.extract_positions or config.get("analysis", {}).get("extract_positions", "prefix")
    max_prefix_len = getattr(sampler.config, "prefix_probe_max_len", None) if extract_positions == "prefix" else None
    extract_hidden_states(
        model=model,
        sampler=sampler,
        task_name=task_name,
        state_kind=state_kind,
        layers=extract_layers,
        num_examples=args.num_examples,
        batch_size=args.batch_size,
        max_prefix_len=max_prefix_len,
        position_mode=extract_positions,
        device=args.device,
        run_dir=run_dir,
        checkpoint_name=checkpoint_name,
    )
    print(f"saved hidden states and labels for checkpoint={checkpoint_name} to {run_dir}")


def resolve_checkpoint(run_dir: Path, checkpoint: str) -> tuple[Path, str]:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.suffix == ".pt":
        return checkpoint_path, checkpoint_path.stem.removeprefix("model_")
    if checkpoint == "final":
        return run_dir / "checkpoints" / "model_final.pt", "final"
    if checkpoint == "best":
        return run_dir / "checkpoints" / "model_best.pt", "best"
    step = int(checkpoint)
    return run_dir / "checkpoints" / f"model_step_{step}.pt", f"step_{step}"


if __name__ == "__main__":
    main()
