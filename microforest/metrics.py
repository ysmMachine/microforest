"""Metric helpers for multi-task binary classification."""

from __future__ import annotations

import numpy as np


def binary_counts(y_true, y_pred) -> tuple[int, int, int, int]:
    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    tp = int(np.logical_and(truth == 1, pred == 1).sum())
    fp = int(np.logical_and(truth == 0, pred == 1).sum())
    tn = int(np.logical_and(truth == 0, pred == 0).sum())
    fn = int(np.logical_and(truth == 1, pred == 0).sum())
    return tp, fp, tn, fn


def precision_recall_f1(y_true, y_pred) -> dict[str, float]:
    tp, fp, _tn, fn = binary_counts(y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def macro_task_metrics(y_true, y_pred) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if truth.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if truth.ndim == 1:
        truth = truth.reshape(-1, 1)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)

    n_tasks = truth.shape[1]
    totals = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for task_idx in range(n_tasks):
        scores = precision_recall_f1(truth[:, task_idx], pred[:, task_idx])
        for key in totals:
            totals[key] += scores[key]

    return {key: value / n_tasks for key, value in totals.items()}
