import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", required=True, help="Directory containing saved probe weights.")
    args = parser.parse_args()
    probe_dir = Path(args.probe_dir)
    summary_path = probe_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Run scripts/run_probes.py first; missing {summary_path}")
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
