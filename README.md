# MicroForest

Reconstructed reference implementation for **MicroForest: Lightweight Bottleneck
Prediction for Manufacturing Processes on Edge Devices**.

This repository includes:

- a DAG-based manufacturing process simulator,
- synthetic dataset generation for multi-task bottleneck prediction,
- a small pure-Python random forest teacher,
- `SplitSelector` for extracting high-information-gain split rules,
- `MicroTree` and `MicroForest` implementations,
- a smoke test that verifies the full pipeline without long training.

The implementation is designed as a runnable research artifact, not as an exact
reproduction of every experiment table in the paper.

## Setup

```bash
python -m venv .venv
.\.venv\bin\python.exe -m pip install -r requirements.txt
```

On Windows Python distributions that create `Scripts` instead of `bin`, use:

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Quick Start

```bash
.\.venv\bin\python.exe scripts/smoke_test.py
```

Expected output includes generated dataset dimensions, a trained teacher forest,
MicroForest metrics, model size estimates, and a saved model path.

## Generate a Dataset

```bash
python scripts/generate_dataset.py --tasks 20 --samples 500 --out data/sample.csv
```

The generated CSV contains feature columns followed by label columns
`y_task_0 ... y_task_N`.

## Train and Check MicroForest

```bash
python scripts/train_microforest.py --data data/sample.csv --tasks 20 --model-out artifacts/microforest.pkl
```

This trains scikit-learn teacher random forests and builds compact MicroTrees
from the selected split pool. The defaults are intentionally small so the script
finishes quickly on a normal laptop.

## Project Layout

```text
microforest/
  data.py          CSV loading and saving helpers
  metrics.py       binary and macro metrics
  models.py        sklearn RF teachers, SplitSelector, MicroTree, MicroForest
  simulation.py    DAG manufacturing simulator and dataset builder
scripts/
  generate_dataset.py
  smoke_test.py
  train_microforest.py
tests/
  test_pipeline.py
```

## Notes

The paper evaluates task counts up to 150 and compares against RF, LightGBM,
DCT, and SSF on edge hardware. This repository focuses on reconstructing the
core MicroForest mechanism and providing a clean, working baseline that can be
extended for the full experimental setup.
