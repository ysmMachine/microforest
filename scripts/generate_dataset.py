from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microforest.data import save_dataset
from microforest.simulation import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a MicroForest synthetic dataset.")
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="data/sample.csv")
    args = parser.parse_args()

    _graph, feature_names, label_names, features, labels = build_dataset(
        n_tasks=args.tasks,
        samples=args.samples,
        horizon=args.horizon,
        seed=args.seed,
    )
    save_dataset(args.out, feature_names, label_names, features, labels)
    print(f"saved {len(features)} rows, {len(feature_names)} features, {len(label_names)} labels to {args.out}")


if __name__ == "__main__":
    main()
