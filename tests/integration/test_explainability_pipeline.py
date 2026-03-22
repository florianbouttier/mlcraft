import numpy as np
import pytest

from mlcraft.core.results import EvaluationResult, FoldSummary, MetricRow, ShapResult, TrialSummary, TuningResult
from mlcraft.core.task import TaskSpec
from mlcraft.reporting.full_report import FullReportBuilder


def test_full_report_builder_combines_sections():
    pytest.importorskip("matplotlib")
    pytest.importorskip("jinja2")
    evaluation = EvaluationResult(
        task_spec=TaskSpec(task_type="regression"),
        metric_rows=[MetricRow("m1", "rmse", 0.2, -0.2, False)],
    )
    fold = FoldSummary(0, {"rmse": 0.1}, {"rmse": 0.2}, -0.1, -0.2, -0.2)
    trial = TrialSummary(0, {"depth": 4}, {"rmse": 0.1}, {"rmse": 0.2}, -0.1, -0.2, -0.2, [fold])
    tuning = TuningResult(
        task_spec=TaskSpec(task_type="regression"),
        best_params={"depth": 4},
        best_score=-0.2,
        best_trial=trial,
        history=[trial],
        train_metrics={"rmse": 0.1},
        val_metrics={"rmse": 0.2},
        penalized_score=-0.2,
        fold_summaries=[fold],
        metric_name="rmse",
    )
    shap = ShapResult(
        feature_names=["a", "b"],
        shap_values=np.array([[0.1, -0.1], [0.2, -0.2]]),
        feature_values=np.array([[1.0, 0.0], [2.0, 1.0]]),
        importance=np.array([0.15, 0.15]),
    )
    html = FullReportBuilder().build(evaluation=evaluation, tuning=tuning, shap=shap)
    assert "Full Report" in html
    assert "Evaluation" in html
    assert "SHAP" in html
    assert html.find("<h2>Tuning</h2>") < html.find("<h2>Evaluation</h2>")
