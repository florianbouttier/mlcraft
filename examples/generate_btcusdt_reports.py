"""Generate end-to-end HTML and JSON artifacts from the BTCUSDT parquet fixture."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlcraft.core.task import TaskSpec, TaskType
from mlcraft.data.inference import infer_schema
from mlcraft.evaluation.evaluator import Evaluator
from mlcraft.evaluation.renderer import EvaluationReportRenderer
from mlcraft.models.base import BaseGBMModel
from mlcraft.reporting.full_report import FullReportBuilder
from mlcraft.split.train_test import train_test_split_time
from mlcraft.tuning.optuna_search import OptunaSearch


class DummyParquetRegressorModel(BaseGBMModel):
    """Fit a tiny deterministic regressor for report generation."""

    backend_name = "dummy"

    @staticmethod
    def _select_feature(X: np.ndarray) -> np.ndarray:
        start = 1 if X.shape[1] > 1 else 0
        stop = min(start + 6, X.shape[1])
        features = np.nan_to_num(np.asarray(X[:, start:stop], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if features.shape[1] == 0:
            return np.zeros((X.shape[0],), dtype=float)
        return np.asarray(features[:, 0], dtype=float)

    def _fit_backend(self, X, y, *, sample_weight=None, exposure=None, eval_set=None, metadata=None):
        feature = self._select_feature(X)
        centered_feature = feature - float(np.mean(feature))
        variance = float(np.mean(centered_feature**2))
        if variance <= 1e-12:
            slope = 0.0
        else:
            slope = float(np.mean(centered_feature * (np.asarray(y, dtype=float) - float(np.mean(y)))) / variance)
        intercept = float(np.mean(y) - slope * np.mean(feature))
        return {"intercept": intercept, "slope": slope}

    def _predict_backend(self, X, *, metadata=None):
        feature = self._select_feature(X)
        return self.model_["intercept"] + self.model_["slope"] * feature


class DummyParquetClassifierModel(BaseGBMModel):
    """Fit a tiny deterministic classifier for report generation."""

    backend_name = "dummy"

    @staticmethod
    def _select_feature(X: np.ndarray) -> np.ndarray:
        start = 1 if X.shape[1] > 1 else 0
        stop = min(start + 6, X.shape[1])
        features = np.nan_to_num(np.asarray(X[:, start:stop], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if features.shape[1] == 0:
            return np.zeros((X.shape[0],), dtype=float)
        return np.asarray(features[:, 0], dtype=float)

    def _fit_backend(self, X, y, *, sample_weight=None, exposure=None, eval_set=None, metadata=None):
        feature = self._select_feature(X)
        centered_feature = feature - float(np.mean(feature))
        target = np.asarray(y, dtype=float)
        centered_target = target - float(np.mean(target))
        variance = float(np.mean(centered_feature**2))
        if variance <= 1e-12:
            slope = 0.0
        else:
            slope = float(np.mean(centered_feature * centered_target) / variance)
        baseline = float(np.clip(np.mean(target), 1e-6, 1.0 - 1e-6))
        intercept = float(np.log(baseline / (1.0 - baseline)) - slope * np.mean(feature))
        return {"intercept": intercept, "slope": slope}

    def _predict_backend(self, X, *, metadata=None):
        feature = self._select_feature(X)
        logits = self.model_["intercept"] + self.model_["slope"] * feature
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -25.0, 25.0)))


def load_btcusdt_fixture() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Load the repository parquet fixture as column-oriented numpy arrays."""

    table = pq.read_table(ROOT / "tests" / "btcusdt_regression_h4.parquet")
    X = {
        name: table[name].to_numpy(zero_copy_only=False)
        for name in table.column_names
        if name not in {"target_regression", "future_return"}
    }
    y_regression = table["target_regression"].to_numpy(zero_copy_only=False).astype(float)
    y_classification = (y_regression > 0.0).astype(int)
    return X, y_regression, y_classification


def make_dummy_model(task_spec: TaskSpec) -> BaseGBMModel:
    """Create a deterministic dummy model matching the requested task."""

    if task_spec.task_type == TaskType.CLASSIFICATION:
        return DummyParquetClassifierModel(task_spec=task_spec)
    return DummyParquetRegressorModel(task_spec=task_spec)


def write_json(path: Path, payload: dict) -> None:
    """Write a JSON payload with UTF-8 encoding."""

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_time_split_artifacts(
    X: dict[str, np.ndarray],
    y: np.ndarray,
    *,
    task_spec: TaskSpec,
    output_dir: Path,
) -> dict[str, str]:
    """Generate evaluation artifacts for one chronological holdout split."""

    output_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split_time(X, y, time_column="timestamp", test_size=0.2)
    model = make_dummy_model(task_spec)
    model.fit(X_train, y_train)
    bundle = model.predict_bundle(X_test, name=f"{task_spec.task_type.value}_holdout")
    evaluation = Evaluator().evaluate(y_test, bundle, task_spec=task_spec)

    evaluation_html = output_dir / "evaluation.html"
    evaluation_json = output_dir / "evaluation.json"
    full_report_html = output_dir / "full_report.html"
    schema_json = output_dir / "schema.json"

    EvaluationReportRenderer().render(evaluation, output_path=evaluation_html)
    write_json(evaluation_json, evaluation.to_dict(include_arrays=True))
    FullReportBuilder().build(evaluation=evaluation, output_path=full_report_html)
    write_json(schema_json, infer_schema(X).to_dict())
    return {
        "evaluation_html": str(evaluation_html),
        "evaluation_json": str(evaluation_json),
        "full_report_html": str(full_report_html),
        "schema_json": str(schema_json),
    }


def build_kfold_artifacts(
    X: dict[str, np.ndarray],
    y: np.ndarray,
    *,
    task_spec: TaskSpec,
    output_dir: Path,
) -> dict[str, str]:
    """Generate tuning artifacts with KFold CV and a final chronological holdout."""

    output_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split_time(X, y, time_column="timestamp", test_size=0.2)

    def fake_create(model_type, *, task_spec, model_params=None, fit_params=None, random_state=None, logger=None):
        model = make_dummy_model(task_spec)
        model.model_params = dict(model_params or {})
        model.fit_params = dict(fit_params or {})
        model.random_state = random_state
        model.logger = logger
        return model

    with patch("mlcraft.tuning.optuna_search.ModelFactory.create", fake_create):
        result, artifacts = OptunaSearch(
            task_spec=task_spec,
            model_type="xgboost",
            n_trials=2,
            cv=3,
            alpha=0.05,
            random_state=13,
            search_space={
                "max_depth": {"type": "int", "low": 2, "high": 4},
                "learning_rate": {"type": "float", "low": 0.03, "high": 0.15},
            },
        ).run_with_artifacts(
            X_train,
            y_train,
            X_test=X_test,
            y_test=y_test,
            output_dir=output_dir,
            title=f"{task_spec.task_type.value.title()} parquet tuning",
        )
    return {
        "tuning_html": str(artifacts.report_path),
        "tuning_json": str(artifacts.result_path),
        "full_report_html": str(artifacts.full_report_path) if artifacts.full_report_path is not None else "",
        "metric_name": str(result.metric_name),
    }


def main() -> None:
    """Generate all demo outputs under `artifacts/btcusdt_demo`."""

    X, y_regression, y_classification = load_btcusdt_fixture()
    output_root = ROOT / "artifacts" / "btcusdt_demo"
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": {
            "path": str(ROOT / "tests" / "btcusdt_regression_h4.parquet"),
            "row_count": int(len(y_regression)),
            "feature_count": int(len(X)),
            "feature_names": list(X.keys()),
        },
        "artifacts": {
            "regression_time": build_time_split_artifacts(
                X,
                y_regression,
                task_spec=TaskSpec(task_type="regression"),
                output_dir=output_root / "regression_time",
            ),
            "classification_time": build_time_split_artifacts(
                X,
                y_classification,
                task_spec=TaskSpec(task_type="classification"),
                output_dir=output_root / "classification_time",
            ),
            "regression_kfold": build_kfold_artifacts(
                X,
                y_regression,
                task_spec=TaskSpec(task_type="regression"),
                output_dir=output_root / "regression_kfold",
            ),
            "classification_kfold": build_kfold_artifacts(
                X,
                y_classification,
                task_spec=TaskSpec(task_type="classification"),
                output_dir=output_root / "classification_kfold",
            ),
        },
    }

    summary_path = output_root / "summary.json"
    write_json(summary_path, summary)
    print(f"Generated demo artifacts in: {output_root}")
    print(f"Summary: {summary_path}")
    for section_name, payload in summary["artifacts"].items():
        print(f"- {section_name}:")
        for key, value in payload.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
