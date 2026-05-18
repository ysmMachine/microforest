from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from microforest.metrics import macro_task_metrics
from microforest.models import MicroForest
from microforest.simulation import build_dataset


class ConstantTaskModel:
    def __init__(self, value: int):
        self.value = int(value)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], self.value, dtype=np.int64)


def train_task_models(factory, train_x: np.ndarray, train_y: np.ndarray) -> list:
    models = []
    for task_idx in range(train_y.shape[1]):
        y_task = train_y[:, task_idx]
        if np.unique(y_task).size == 1:
            models.append(ConstantTaskModel(int(y_task[0])))
            continue
        model = factory(task_idx)
        model.fit(train_x, y_task)
        models.append(model)
    return models


def predict_task_models(models: list, test_x: np.ndarray) -> np.ndarray:
    return np.vstack([model.predict(test_x) for model in models]).T.astype(np.int64)


def pickle_size_bytes(model) -> int:
    return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))


def benchmark_model(name: str, train_func, predict_func, test_y: np.ndarray) -> dict[str, float | str]:
    started = time.perf_counter()
    model = train_func()
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    pred = predict_func(model)
    predict_seconds = time.perf_counter() - started

    scores = macro_task_metrics(test_y, pred)
    return {
        "model": name,
        "macro_precision": scores["precision"],
        "macro_recall": scores["recall"],
        "macro_f1": scores["f1"],
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "serialized_kb": pickle_size_bytes(model) / 1024.0,
    }


def train_compact_microforest(args, train_x: np.ndarray, train_y: np.ndarray) -> MicroForest:
    model = MicroForest(
        n_tasks=args.tasks,
        teacher_estimators=args.rf_estimators,
        teacher_max_depth=args.rf_depth,
        microtree_max_depth=args.rf_depth,
        top_percent=args.micro_top_percent,
        min_samples_leaf=3,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(train_x, train_y)
    model.discard_teachers()
    return model


def write_outputs(rows: list[dict[str, float | str]], out_csv: Path, out_md: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "train_seconds",
        "predict_seconds",
        "serialized_kb",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with out_md.open("w", encoding="utf-8") as handle:
        handle.write("| Model | Macro Precision | Macro Recall | Macro F1 | Train s | Predict s | Pickle KB |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['model']} | "
                f"{row['macro_precision']:.3f} | "
                f"{row['macro_recall']:.3f} | "
                f"{row['macro_f1']:.3f} | "
                f"{row['train_seconds']:.3f} | "
                f"{row['predict_seconds']:.4f} | "
                f"{row['serialized_kb']:.1f} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MicroForest with RF and LightGBM baselines.")
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--prediction-window", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--rf-estimators", type=int, default=40)
    parser.add_argument("--rf-depth", type=int, default=6)
    parser.add_argument("--micro-top-percent", type=float, default=25.0)
    parser.add_argument("--out-csv", default="artifacts/baseline_comparison.csv")
    parser.add_argument("--out-md", default="artifacts/baseline_comparison.md")
    args = parser.parse_args()

    _graph, feature_names, _label_names, features, labels = build_dataset(
        n_tasks=args.tasks,
        samples=args.samples,
        horizon=args.horizon,
        prediction_window=args.prediction_window,
        seed=args.seed,
    )
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    split = int(round(len(x) * (1.0 - args.test_size)))
    train_x, test_x = x[:split], x[split:]
    train_y, test_y = y[:split], y[split:]

    rows: list[dict[str, float | str]] = []
    rows.append(
        benchmark_model(
            "MicroForest",
            lambda: train_compact_microforest(args, train_x, train_y),
            lambda model: model.predict(test_x),
            test_y,
        )
    )

    rows.append(
        benchmark_model(
            "Random Forest",
            lambda: train_task_models(
                lambda task_idx: RandomForestClassifier(
                    n_estimators=args.rf_estimators,
                    max_depth=args.rf_depth,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    random_state=args.seed + task_idx,
                    n_jobs=-1,
                ),
                train_x,
                train_y,
            ),
            lambda models: predict_task_models(models, test_x),
            test_y,
        )
    )

    try:
        from lightgbm import LGBMClassifier

        def lgbm_factory(task_idx: int) -> LGBMClassifier:
            return LGBMClassifier(
                n_estimators=args.rf_estimators,
                max_depth=args.rf_depth,
                learning_rate=0.08,
                num_leaves=31,
                min_child_samples=5,
                objective="binary",
                random_state=args.seed + task_idx,
                n_jobs=-1,
                verbose=-1,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows.append(
                benchmark_model(
                    "LightGBM",
                    lambda: train_task_models(lgbm_factory, train_x, train_y),
                    lambda models: predict_task_models(models, test_x),
                    test_y,
                )
            )
    except ImportError:
        rows.append(
            {
                "model": "LightGBM",
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "train_seconds": 0.0,
                "predict_seconds": 0.0,
                "serialized_kb": 0.0,
            }
        )

    write_outputs(rows, Path(args.out_csv), Path(args.out_md))
    print(f"dataset: rows={len(x)}, features={len(feature_names)}, tasks={args.tasks}")
    for row in rows:
        print(
            f"{row['model']}: f1={row['macro_f1']:.3f}, "
            f"train={row['train_seconds']:.3f}s, predict={row['predict_seconds']:.4f}s, "
            f"pickle={row['serialized_kb']:.1f}KB"
        )
    print(f"saved: {args.out_csv}")
    print(f"saved: {args.out_md}")


if __name__ == "__main__":
    main()
