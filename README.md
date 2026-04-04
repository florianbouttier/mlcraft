# mlcraft

`mlcraft` is a modular, numpy-first machine learning library built to stay reusable across projects while keeping a small dependency surface and a coherent object model.

## What the repo already contains

- column-oriented data handling with `dict[str, np.ndarray]`
- schema inference with reusable `DataSchema` and `ColumnSchema`
- shared task configuration through `TaskSpec`
- pure-numpy metrics and evaluation
- unified model wrappers for XGBoost, LightGBM, and CatBoost
- Optuna tuning with a validation overfitting penalty
- HTML reporting for evaluation, tuning, and SHAP
- parquet-based end-to-end tests on real feature columns

## Project layout

```text
src/mlcraft/
  core/
  data/
  split/
  metrics/
  evaluation/
  models/
  tuning/
  shap/
  reporting/
  utils/

examples/
tests/
```

## Install

The package metadata targets Python `>=3.10`.

Minimal install:

```bash
python -m pip install -e .
```

Install with reporting extras:

```bash
python -m pip install -e ".[reporting,dev]"
```

If you want real backend wrappers instead of the dummy end-to-end demo models, also install one or more optional backends:

```bash
python -m pip install xgboost lightgbm catboost
```

## Run the test suite

```bash
python -m pytest -q
```

Run only the parquet end-to-end coverage:

```bash
python -m pytest -q tests/integration/test_parquet_end_to_end.py
```

Those tests use the real fixture [btcusdt_regression_h4.parquet](/Users/florianbouttier/mlcraft/tests/btcusdt_regression_h4.parquet), including real timestamp, numeric, and categorical columns.

Covered scenarios:

- regression with chronological train/test split
- classification with chronological train/test split
- regression with KFold tuning plus final chronological holdout
- classification with KFold tuning plus final chronological holdout
- HTML and JSON artifact generation for each path

## Generate real demo outputs from the parquet fixture

This repo now includes a reproducible demo script built on the same real fixture and the same reporting stack used by the tests:

[generate_btcusdt_reports.py](/Users/florianbouttier/mlcraft/examples/generate_btcusdt_reports.py)

Run it from the repository root:

```bash
PYTHONPATH=src python3 examples/generate_btcusdt_reports.py
```

It generates outputs under:

```text
artifacts/btcusdt_demo/
  summary.json
  regression_time/
  classification_time/
  regression_kfold/
  classification_kfold/
```

Typical files produced:

- `evaluation.html`
- `evaluation.json`
- `report.html`
- `tuning.json`
- `full_report.html`
- `schema.json`

`summary.json` lists every generated file path so you can jump directly to the outputs.

## Quick start

```python
import numpy as np

from mlcraft.core.prediction import PredictionBundle
from mlcraft.core.task import TaskSpec
from mlcraft.data.inference import infer_schema
from mlcraft.evaluation.evaluator import Evaluator

X = {
    "f1": np.array([1.0, 2.0, 3.0, 4.0]),
    "f2": np.array([0, 1, 0, 1]),
}
y = np.array([1.2, 2.1, 3.2, 3.8])

schema = infer_schema(X)
task = TaskSpec(task_type="regression")
bundle = PredictionBundle(
    name="baseline",
    y_pred=np.array([1.0, 2.0, 3.0, 4.0]),
    task_spec=task,
)

result = Evaluator().evaluate(y, bundle)
print(schema.to_dict())
print(result.to_dict())
```

## End-to-end examples on real market data

### 1. Chronological holdout evaluation

```python
import pyarrow.parquet as pq

from mlcraft.core.task import TaskSpec
from mlcraft.split.train_test import train_test_split_time

table = pq.read_table("tests/btcusdt_regression_h4.parquet")
X = {
    name: table[name].to_numpy(zero_copy_only=False)
    for name in table.column_names
    if name not in {"target_regression", "future_return"}
}
y = table["target_regression"].to_numpy(zero_copy_only=False)

task = TaskSpec(task_type="regression")
X_train, X_test, y_train, y_test = train_test_split_time(
    X,
    y,
    time_column="timestamp",
    test_size=0.2,
)
```

### 2. KFold tuning with final holdout evaluation

```python
from mlcraft.core.task import TaskSpec
from mlcraft.tuning.optuna_search import OptunaSearch

search = OptunaSearch(
    task_spec=TaskSpec(task_type="classification"),
    model_type="xgboost",
    n_trials=20,
    cv=5,
    alpha=0.05,
    random_state=13,
)

result = search.run(
    X_train,
    y_train,
    X_test=X_test,
    y_test=y_test,
)
```

## Important notes

- The library is designed around `numpy` containers first.
- `pandas` and `polars` are intentionally not part of the main data API.
- SHAP is optional.
- XGBoost, LightGBM, and CatBoost are optional.
- The report layer is separate from metric and tuning computation.
- The parquet end-to-end demo uses small deterministic dummy models when optional GBM backends are not required, so the artifact pipeline can be exercised without installing every backend.

## Useful files

- [AGENT.md](/Users/florianbouttier/mlcraft/AGENT.md)
- [pyproject.toml](/Users/florianbouttier/mlcraft/pyproject.toml)
- [test_parquet_end_to_end.py](/Users/florianbouttier/mlcraft/tests/integration/test_parquet_end_to_end.py)
- [generate_btcusdt_reports.py](/Users/florianbouttier/mlcraft/examples/generate_btcusdt_reports.py)
