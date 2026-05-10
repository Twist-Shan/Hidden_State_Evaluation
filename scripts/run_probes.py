import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hse.analysis.compression import run_compression_probes
from hse.analysis.probes import load_probe_data, run_sufficient_statistic_probes
from hse.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to extracted hidden states.")
    parser.add_argument("--labels", default=None, help="Path to sufficient-statistics labels.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    features = Path(args.features)
    labels = Path(args.labels) if args.labels else features.with_name("labels.parquet")
    if not labels.exists() and labels.with_suffix(".csv").exists():
        labels = labels.with_suffix(".csv")
    X, label_df = load_probe_data(features, labels)
    probe_results = run_sufficient_statistic_probes(X, label_df, seed=args.seed)
    compression_rows, compression_summary = run_compression_probes(X, label_df, seed=args.seed)
    out_dir = features.parent / "probes"
    out_dir.mkdir(exist_ok=True)
    summary = {
        "left_r2": probe_results["left"]["r2"],
        "right_r2": probe_results["right"]["r2"],
        "height_r2": probe_results["height"]["r2"],
        "height_mae": probe_results["height"]["mae"],
        **probe_results["geometry"],
        **compression_summary,
    }
    save_json(summary, out_dir / "summary.json")
    compression_rows.to_csv(out_dir / "compression_probe_rows.csv", index=False)
    print(f"saved probe summaries to {out_dir}")


if __name__ == "__main__":
    main()
