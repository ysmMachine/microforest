from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microforest.data import load_dataset
from microforest.metrics import macro_task_metrics
from microforest.models import MicroForest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a MicroForest from a generated CSV.")
    parser.add_argument("--data", default="data/sample.csv")
    parser.add_argument("--tasks", type=int, required=True)
    parser.add_argument("--teacher-estimators", type=int, default=10)
    parser.add_argument("--teacher-depth", type=int, default=5)
    parser.add_argument("--microtree-depth", type=int, default=5)
    parser.add_argument("--top-percent", type=float, default=25.0)
    parser.add_argument("--model-out", default="artifacts/microforest.pkl")
    args = parser.parse_args()

    _feature_names, _label_names, features, labels = load_dataset(args.data, args.tasks)
    split = max(1, int(len(features) * 0.8))
    train_x, test_x = features[:split], features[split:]
    train_y, test_y = labels[:split], labels[split:]

    model = MicroForest(
        n_tasks=args.tasks,
        teacher_estimators=args.teacher_estimators,
        teacher_max_depth=args.teacher_depth,
        microtree_max_depth=args.microtree_depth,
        top_percent=args.top_percent,
    )

    started = time.perf_counter()
    model.fit(train_x, train_y)
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    predictions = model.predict(test_x)
    predict_seconds = time.perf_counter() - started
    scores = macro_task_metrics(test_y, predictions)

    model.save(args.model_out)
    print(f"train_seconds={train_seconds:.3f}")
    print(f"predict_seconds={predict_seconds:.6f}")
    print(f"macro_precision={scores['precision']:.3f}")
    print(f"macro_recall={scores['recall']:.3f}")
    print(f"macro_f1={scores['f1']:.3f}")
    print(f"teacher_nodes={model.teacher_node_count()}")
    print(f"microtree_nodes={model.microtree_node_count()}")
    print(f"saved_model={args.model_out}")


if __name__ == "__main__":
    main()
