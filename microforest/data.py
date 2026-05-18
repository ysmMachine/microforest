"""CSV helpers for generated MicroForest datasets."""

from __future__ import annotations

import csv
from pathlib import Path


def save_dataset(
    path: str | Path,
    feature_names: list[str],
    label_names: list[str],
    features: list[list[float]],
    labels: list[list[int]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(feature_names + label_names)
        for x_row, y_row in zip(features, labels):
            writer.writerow([*x_row, *y_row])


def load_dataset(
    path: str | Path, n_tasks: int
) -> tuple[list[str], list[str], list[list[float]], list[list[int]]]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        feature_names = header[:-n_tasks]
        label_names = header[-n_tasks:]
        features: list[list[float]] = []
        labels: list[list[int]] = []
        for row in reader:
            features.append([float(value) for value in row[:-n_tasks]])
            labels.append([int(float(value)) for value in row[-n_tasks:]])
    return feature_names, label_names, features, labels

