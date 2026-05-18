"""MicroForest models built around NumPy and scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def _as_array(x: np.ndarray | list[list[float]]) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _as_labels(y: np.ndarray | list[int]) -> np.ndarray:
    return np.asarray(y, dtype=np.int64).reshape(-1)


def _gini(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    p1 = float(np.mean(y))
    return 1.0 - p1 * p1 - (1.0 - p1) * (1.0 - p1)


def information_gain(y: np.ndarray, left_mask: np.ndarray) -> float:
    right_mask = ~left_mask
    left_count = int(left_mask.sum())
    right_count = int(right_mask.sum())
    if left_count == 0 or right_count == 0:
        return 0.0

    left_y = y[left_mask]
    right_y = y[right_mask]
    weighted = (left_count / y.size) * _gini(left_y)
    weighted += (right_count / y.size) * _gini(right_y)
    return _gini(y) - weighted


def majority_class(y: np.ndarray) -> int:
    return int(np.mean(y) >= 0.5)


@dataclass
class MicroNode:
    prediction: int
    feature: int | None = None
    threshold: float | None = None
    gain: float = 0.0
    left: "MicroNode | None" = None
    right: "MicroNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class SplitSelector:
    """Extract top information-gain split rules from a fitted sklearn RF."""

    def __init__(self, top_percent: float = 25.0, round_decimals: int = 8) -> None:
        if not 0.0 < top_percent <= 100.0:
            raise ValueError("top_percent must be in (0, 100].")
        self.top_percent = top_percent
        self.round_decimals = round_decimals

    def select(self, forest: RandomForestClassifier) -> list[tuple[int, float, float]]:
        rules: dict[tuple[int, float], float] = {}
        for estimator in forest.estimators_:
            tree = estimator.tree_
            for node_idx in range(tree.node_count):
                feature = int(tree.feature[node_idx])
                if feature < 0:
                    continue
                threshold = round(float(tree.threshold[node_idx]), self.round_decimals)
                gain = self._node_information_gain(tree, node_idx)
                key = (feature, threshold)
                rules[key] = max(rules.get(key, 0.0), gain)

        ranked = sorted(rules.items(), key=lambda item: item[1], reverse=True)
        keep = max(1, int(np.ceil(len(ranked) * self.top_percent / 100.0)))
        return [(feature, threshold, gain) for (feature, threshold), gain in ranked[:keep]]

    @staticmethod
    def _node_information_gain(tree, node_idx: int) -> float:
        left = int(tree.children_left[node_idx])
        right = int(tree.children_right[node_idx])
        if left < 0 or right < 0:
            return 0.0

        parent_weight = float(tree.weighted_n_node_samples[node_idx])
        left_weight = float(tree.weighted_n_node_samples[left])
        right_weight = float(tree.weighted_n_node_samples[right])
        if parent_weight <= 0:
            return 0.0

        parent_impurity = float(tree.impurity[node_idx])
        child_impurity = (left_weight / parent_weight) * float(tree.impurity[left])
        child_impurity += (right_weight / parent_weight) * float(tree.impurity[right])
        return parent_impurity - child_impurity


class MicroTree:
    """Decision tree constrained to split rules selected from teacher forests."""

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_leaf: int = 3,
        min_gain: float = 1e-9,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.root_: MicroNode | None = None
        self.split_pool_: list[tuple[int, float, float]] = []

    def fit(
        self,
        x: np.ndarray | list[list[float]],
        y: np.ndarray | list[int],
        split_pool: list[tuple[int, float, float]],
    ) -> "MicroTree":
        x_array = _as_array(x)
        y_array = _as_labels(y)
        self.split_pool_ = list(split_pool)
        self.root_ = self._build(x_array, y_array, self.split_pool_, depth=0)
        return self

    def predict(self, x: np.ndarray | list[list[float]]) -> np.ndarray:
        x_array = _as_array(x)
        return np.asarray([self.predict_one(row) for row in x_array], dtype=np.int64)

    def predict_one(self, row: np.ndarray) -> int:
        if self.root_ is None:
            raise ValueError("MicroTree is not fitted.")
        node = self.root_
        while not node.is_leaf:
            assert node.feature is not None and node.threshold is not None
            node = node.left if row[node.feature] <= node.threshold else node.right
            assert node is not None
        return node.prediction

    def node_count(self) -> int:
        def count(node: MicroNode | None) -> int:
            if node is None:
                return 0
            return 1 + count(node.left) + count(node.right)

        return count(self.root_)

    def _build(
        self,
        x: np.ndarray,
        y: np.ndarray,
        split_pool: list[tuple[int, float, float]],
        depth: int,
    ) -> MicroNode:
        prediction = majority_class(y)
        if (
            depth >= self.max_depth
            or y.size < self.min_samples_leaf * 2
            or np.unique(y).size == 1
            or not split_pool
        ):
            return MicroNode(prediction=prediction)

        best: tuple[int, float, float, np.ndarray] | None = None
        for feature, threshold, _teacher_gain in split_pool:
            left_mask = x[:, feature] <= threshold
            left_count = int(left_mask.sum())
            right_count = y.size - left_count
            if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                continue
            gain = information_gain(y, left_mask)
            if best is None or gain > best[2]:
                best = (feature, threshold, gain, left_mask)

        if best is None or best[2] < self.min_gain:
            return MicroNode(prediction=prediction)

        feature, threshold, gain, left_mask = best
        remaining_pool = [
            rule for rule in split_pool if not (rule[0] == feature and rule[1] == threshold)
        ]
        return MicroNode(
            prediction=prediction,
            feature=feature,
            threshold=threshold,
            gain=gain,
            left=self._build(x[left_mask], y[left_mask], remaining_pool, depth + 1),
            right=self._build(x[~left_mask], y[~left_mask], remaining_pool, depth + 1),
        )


class MicroForest:
    """Multi-task MicroForest with one teacher RF and MicroTree per task."""

    def __init__(
        self,
        n_tasks: int,
        teacher_estimators: int = 40,
        teacher_max_depth: int = 5,
        microtree_max_depth: int = 5,
        top_percent: float = 25.0,
        min_samples_leaf: int = 3,
        random_state: int = 0,
        n_jobs: int | None = None,
    ) -> None:
        self.n_tasks = n_tasks
        self.teacher_estimators = teacher_estimators
        self.teacher_max_depth = teacher_max_depth
        self.microtree_max_depth = microtree_max_depth
        self.top_percent = top_percent
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.teachers_: list[RandomForestClassifier] = []
        self.microtrees_: list[MicroTree] = []
        self.split_pools_: list[list[tuple[int, float, float]]] = []

    def fit(
        self,
        x: np.ndarray | list[list[float]],
        y: np.ndarray | list[list[int]],
    ) -> "MicroForest":
        x_array = _as_array(x)
        y_array = np.asarray(y, dtype=np.int64)
        if y_array.ndim != 2 or y_array.shape[1] != self.n_tasks:
            raise ValueError(f"Expected y shape (n_samples, {self.n_tasks}).")

        selector = SplitSelector(top_percent=self.top_percent)
        self.teachers_ = []
        self.microtrees_ = []
        self.split_pools_ = []

        for task_idx in range(self.n_tasks):
            teacher = RandomForestClassifier(
                n_estimators=self.teacher_estimators,
                max_depth=self.teacher_max_depth,
                min_samples_leaf=self.min_samples_leaf,
                class_weight="balanced_subsample",
                random_state=self.random_state + task_idx,
                n_jobs=self.n_jobs,
            )
            y_task = y_array[:, task_idx]
            teacher.fit(x_array, y_task)
            split_pool = selector.select(teacher)

            microtree = MicroTree(
                max_depth=self.microtree_max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
            microtree.fit(x_array, y_task, split_pool)

            self.teachers_.append(teacher)
            self.split_pools_.append(split_pool)
            self.microtrees_.append(microtree)
        return self

    def predict(self, x: np.ndarray | list[list[float]]) -> np.ndarray:
        x_array = _as_array(x)
        predictions = [tree.predict(x_array) for tree in self.microtrees_]
        return np.vstack(predictions).T

    def teacher_node_count(self) -> int:
        return int(
            sum(
                estimator.tree_.node_count
                for teacher in self.teachers_
                for estimator in teacher.estimators_
            )
        )

    def microtree_node_count(self) -> int:
        return int(sum(tree.node_count() for tree in self.microtrees_))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "MicroForest":
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError("Pickle did not contain a MicroForest model.")
        return model

