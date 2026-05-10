import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from hse.models import build_model
from hse.tasks.dyck import DyckConfig, DyckSampler
from hse.utils import extract_hidden_states, load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Path to a trained run directory.")
    parser.add_argument("--state-kind", default=None, help="Override hidden-state kind, e.g. h or c.")
    parser.add_argument("--num-examples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_dir = Path(args.run)
    config = load_json(run_dir / "config.json")
    task_config = DyckConfig(**{**config["task"], "device": args.device})
    sampler = DyckSampler(task_config, seed=int(config["seed"]) + 10_000)
    spec = config["model"]
    model_name = config["model_name"]
    model_kwargs = {k: v for k, v in spec.items() if k not in {"name", "state_kind"}}
    model = build_model(model_name=model_name, vocab_size=sampler.vocab_size, **model_kwargs).to(args.device)
    ckpt = torch.load(run_dir / "checkpoints" / "model_final.pt", map_location=args.device)
    model.load_state_dict(ckpt["model"])
    state_kind = args.state_kind or spec.get("state_kind", "h")
    extract_hidden_states(
        model=model,
        sampler=sampler,
        state_kind=state_kind,
        layer=-1,
        num_examples=args.num_examples,
        batch_size=args.batch_size,
        max_prefix_len=task_config.prefix_probe_max_len,
        device=args.device,
        run_dir=run_dir,
    )
    print(f"saved hidden_states.pt and labels file to {run_dir}")


if __name__ == "__main__":
    main()
