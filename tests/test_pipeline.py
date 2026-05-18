from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microforest.metrics import macro_task_metrics
from microforest.models import MicroForest
from microforest.simulation import build_dataset


class PipelineTest(unittest.TestCase):
    def test_microforest_pipeline_runs(self) -> None:
        tasks = 5
        _graph, _feature_names, _label_names, features, labels = build_dataset(
            n_tasks=tasks,
            samples=60,
            horizon=10,
            prediction_window=2,
            seed=3,
        )
        model = MicroForest(
            n_tasks=tasks,
            teacher_estimators=4,
            teacher_max_depth=3,
            microtree_max_depth=3,
            min_samples_leaf=2,
            random_state=5,
        )
        model.fit(features[:45], labels[:45])
        predictions = model.predict(features[45:])
        scores = macro_task_metrics(labels[45:], predictions)

        self.assertEqual(len(predictions), 15)
        self.assertTrue(all(len(row) == tasks for row in predictions))
        self.assertGreaterEqual(scores["f1"], 0.0)
        self.assertLessEqual(scores["f1"], 1.0)
        self.assertGreaterEqual(model.teacher_node_count(), model.microtree_node_count())


if __name__ == "__main__":
    unittest.main()
