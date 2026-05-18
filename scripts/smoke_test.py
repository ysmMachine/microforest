from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microforest.data import save_dataset
from microforest.metrics import macro_task_metrics
from microforest.models import MicroForest
from microforest.simulation import build_dataset


def main() -> None:
    tasks = 8
    graph, feature_names, label_names, features, labels = build_dataset(
        n_tasks=tasks,
        samples=120,
        horizon=20,
        seed=11,
    )

    data_path = ROOT / "data" / "smoke.csv"
    model_path = ROOT / "artifacts" / "smoke_microforest.pkl"
    save_dataset(data_path, feature_names, label_names, features, labels)

    split = 90
    train_x, test_x = features[:split], features[split:]
    train_y, test_y = labels[:split], labels[split:]

    model = MicroForest(
        n_tasks=tasks,
        teacher_estimators=6,
        teacher_max_depth=4,
        microtree_max_depth=4,
        top_percent=25.0,
        min_samples_leaf=2,
        random_state=13,
    )

    started = time.perf_counter()
    model.fit(train_x, train_y)
    train_seconds = time.perf_counter() - started

    predictions = model.predict(test_x)
    scores = macro_task_metrics(test_y, predictions)
    model.save(model_path)

    assert len(predictions) == len(test_x)
    assert all(len(row) == tasks for row in predictions)
    assert model.microtree_node_count() > 0

    print("MicroForest smoke test OK")
    print(f"tasks={tasks}, edges={len(graph.edges)}, rows={len(features)}, features={len(feature_names)}")
    print(f"train_seconds={train_seconds:.3f}")
    print(f"macro_precision={scores['precision']:.3f}")
    print(f"macro_recall={scores['recall']:.3f}")
    print(f"macro_f1={scores['f1']:.3f}")
    print(f"teacher_nodes={model.teacher_node_count()}")
    print(f"microtree_nodes={model.microtree_node_count()}")
    print(f"dataset={data_path}")
    print(f"model={model_path}")


if __name__ == "__main__":
    main()
