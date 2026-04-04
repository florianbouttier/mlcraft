import json
from pathlib import Path

import numpy as np
import pytest

from mlcraft.core.task import TaskSpec, TaskType
from mlcraft.evaluation.evaluator import Evaluator
from mlcraft.evaluation.renderer import EvaluationReportRenderer
from mlcraft.models.base import BaseGBMModel
from mlcraft.reporting.full_report import FullReportBuilder
from mlcraft.split.train_test import train_test_split_time
from mlcraft.tuning.optuna_search import OptunaSearch


class DummyParquetRegressorModel(BaseGBMModel):
    backend_name = "dummy"

    @staticmethod
    def _select_features(X):
        start = 1 if X.shape[1] > 1 else 0
        stop = min(start + 6, X.shape[1])
        features = np.nan_to_num(np.asarray(X[:, start:stop], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if features.shape[1] == 0:
            return np.zeros((X.shape[0],), dtype=float)
        return np.asarray(features[:, 0], dtype=float)

    def _fit_backend(self, X, y, *, sample_weight=None, exposure=None, eval_set=None, metadata=None):
        feature = self._select_features(X)
        centered_feature = feature - float(np.mean(feature))
        variance = float(np.mean(centered_feature**2))
        if variance <= 1e-12:
            slope = 0.0
        else:
            slope = float(np.mean(centered_feature * (np.asarray(y, dtype=float) - float(np.mean(y)))) / variance)
        intercept = float(np.mean(y) - slope * np.mean(feature))
        return {"intercept": intercept, "slope": slope}

    def _predict_backend(self, X, *, metadata=None):
        feature = self._select_features(X)
        return self.model_["intercept"] + self.model_["slope"] * feature


class DummyParquetClassifierModel(BaseGBMModel):
    backend_name = "dummy"

    @staticmethod
    def _select_features(X):
        start = 1 if X.shape[1] > 1 else 0
        stop = min(start + 6, X.shape[1])
        features = np.nan_to_num(np.asarray(X[:, start:stop], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if features.shape[1] == 0:
            return np.zeros((X.shape[0],), dtype=float)
        return np.asarray(features[:, 0], dtype=float)

    def _fit_backend(self, X, y, *, sample_weight=None, exposure=None, eval_set=None, metadata=None):
        feature = self._select_features(X)
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
        feature = self._select_features(X)
        logits = self.model_["intercept"] + self.model_["slope"] * feature
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -25.0, 25.0)))


def _make_dummy_model(task_spec):
    if task_spec.task_type == TaskType.CLASSIFICATION:
        return DummyParquetClassifierModel(task_spec=task_spec)
    return DummyParquetRegressorModel(task_spec=task_spec)


def _write_evaluation_artifacts(result, output_dir: Path, prefix: str) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{prefix}_evaluation.html"
    json_path = output_dir / f"{prefix}_evaluation.json"
    full_path = output_dir / f"{prefix}_full_report.html"

    EvaluationReportRenderer().render(result, output_path=html_path)
    json_path.write_text(json.dumps(result.to_dict(include_arrays=True), indent=2), encoding="utf-8")
    FullReportBuilder().build(evaluation=result, output_path=full_path)
    return html_path, json_path, full_path


def _assert_html_and_json_artifacts(html_path: Path, json_path: Path, full_path: Path, *, html_marker: str, json_marker: str) -> None:
    assert html_path.exists()
    assert json_path.exists()
    assert full_path.exists()
    assert html_marker in html_path.read_text(encoding="utf-8")
    assert json_marker in json_path.read_text(encoding="utf-8")
    assert "mlcraft Full Report" in full_path.read_text(encoding="utf-8")


def _run_kfold_tuning_with_artifacts(monkeypatch, X, y, *, task_spec: TaskSpec, output_dir: Path):
    def fake_create(model_type, *, task_spec, model_params=None, fit_params=None, random_state=None, logger=None):
        model = _make_dummy_model(task_spec)
        model.model_params = dict(model_params or {})
        model.fit_params = dict(fit_params or {})
        model.random_state = random_state
        model.logger = logger
        return model

    monkeypatch.setattr("mlcraft.tuning.optuna_search.ModelFactory.create", fake_create)
    X_train, X_test, y_train, y_test = train_test_split_time(X, y, time_column="timestamp", test_size=0.2)
    search = OptunaSearch(
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
    )
    return search.run_with_artifacts(
        X_train,
        y_train,
        X_test=X_test,
        y_test=y_test,
        output_dir=output_dir,
        title=f"{task_spec.task_type.value.title()} parquet tuning",
    )


@pytest.mark.parametrize(
    ("task_spec", "target_key", "prefix", "expected_metric"),
    [
        (TaskSpec(task_type="regression"), "regression", "btc_regression_time", "rmse"),
        (TaskSpec(task_type="classification"), "classification", "btc_classification_time", "roc_auc"),
    ],
)
def test_parquet_time_split_end_to_end_writes_evaluation_artifacts(
    btcusdt_parquet_data,
    tmp_path,
    task_spec,
    target_key,
    prefix,
    expected_metric,
):
    pytest.importorskip("jinja2")
    pytest.importorskip("matplotlib")
    X, y_regression, y_classification = btcusdt_parquet_data
    y = y_regression if target_key == "regression" else y_classification

    X_train, X_test, y_train, y_test = train_test_split_time(X, y, time_column="timestamp", test_size=0.2)
    model = _make_dummy_model(task_spec)
    model.fit(X_train, y_train)
    bundle = model.predict_bundle(X_test, name=f"{target_key}_holdout")
    evaluation = Evaluator().evaluate(y_test, bundle, task_spec=task_spec)

    html_path, json_path, full_path = _write_evaluation_artifacts(evaluation, tmp_path / prefix, prefix)

    metrics = evaluation.metrics_by_prediction()[f"{target_key}_holdout"]
    assert expected_metric in metrics
    _assert_html_and_json_artifacts(
        html_path,
        json_path,
        full_path,
        html_marker="Graphical leaderboard",
        json_marker=f'"metric_name": "{expected_metric}"',
    )


@pytest.mark.parametrize(
    ("task_spec", "target_key", "prefix", "expected_metric"),
    [
        (TaskSpec(task_type="regression"), "regression", "btc_regression_kfold", "rmse"),
        (TaskSpec(task_type="classification"), "classification", "btc_classification_kfold", "roc_auc"),
    ],
)
def test_parquet_kfold_end_to_end_writes_tuning_artifacts(
    monkeypatch,
    btcusdt_parquet_data,
    tmp_path,
    task_spec,
    target_key,
    prefix,
    expected_metric,
):
    pytest.importorskip("optuna")
    pytest.importorskip("jinja2")
    pytest.importorskip("matplotlib")
    pytest.importorskip("plotly")
    X, y_regression, y_classification = btcusdt_parquet_data
    y = y_regression if target_key == "regression" else y_classification

    result, artifacts = _run_kfold_tuning_with_artifacts(
        monkeypatch,
        X,
        y,
        task_spec=task_spec,
        output_dir=tmp_path / prefix,
    )

    assert result.best_params
    assert result.test_evaluation is not None
    assert result.test_metrics is not None
    assert expected_metric in result.test_metrics
    assert artifacts.report_path.exists()
    assert artifacts.result_path.exists()
    assert artifacts.full_report_path is not None
    assert artifacts.full_report_path.exists()
    report_html = artifacts.report_path.read_text(encoding="utf-8")
    result_json = artifacts.result_path.read_text(encoding="utf-8")
    full_report_html = artifacts.full_report_path.read_text(encoding="utf-8")
    assert "Optuna Plotly" in report_html
    assert "Holdout behavior" in report_html
    assert f'"metric_name": "{expected_metric}"' in result_json
    assert '"test_metrics"' in result_json
    assert "Tuning" in full_report_html
    assert "Evaluation" in full_report_html
