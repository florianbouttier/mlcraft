import pytest

import numpy as np

from mlcraft.core.results import CurveData, EvaluationResult, FoldSummary, MetricRow, ShapResult, TrialSummary, TuningResult
from mlcraft.core.task import TaskSpec


def test_tuning_renderer_generates_html():
    pytest.importorskip("jinja2")
    from mlcraft.tuning.renderer import TuningReportRenderer

    train_eval = EvaluationResult(
        task_spec=TaskSpec(task_type="regression"),
        metric_rows=[MetricRow("train", "rmse", 0.1, -0.1, False)],
        curves={"train": [CurveData("residuals", np.array([0.0, 1.0]), np.array([2.0, 1.0]), "Residual", "Count")]},
    )
    val_eval = EvaluationResult(
        task_spec=TaskSpec(task_type="regression"),
        metric_rows=[MetricRow("validation", "rmse", 0.2, -0.2, False)],
        curves={"validation": [CurveData("residuals", np.array([0.0, 1.0]), np.array([1.0, 2.0]), "Residual", "Count")]},
    )
    fold = FoldSummary(0, {"rmse": 0.1}, {"rmse": 0.2}, -0.1, -0.2, -0.2, train_eval, val_eval)
    trial = TrialSummary(0, {"depth": 4}, {"rmse": 0.1}, {"rmse": 0.2}, -0.1, -0.2, -0.2, [fold])
    result = TuningResult(
        task_spec=TaskSpec(task_type="regression"),
        best_params={"depth": 4},
        best_score=-0.2,
        best_trial=trial,
        history=[trial],
        train_metrics={"rmse": 0.1},
        val_metrics={"rmse": 0.2},
        penalized_score=-0.2,
        fold_summaries=[fold],
        alpha=0.0,
        metric_name="rmse",
        test_metrics={"rmse": 0.3},
        test_score=-0.3,
        test_evaluation=EvaluationResult(
            task_spec=TaskSpec(task_type="regression"),
            metric_rows=[MetricRow("final_test", "rmse", 0.3, -0.3, False)],
            curves={"final_test": [CurveData("residuals", np.array([0.0, 1.0]), np.array([2.0, 1.0]), "Residual", "Count")]},
        ),
    )
    html = TuningReportRenderer().render(result)
    assert "KPI Matrix" in html
    assert "Metric Explorer" in html
    assert "Train, validation, and holdout summary for every tracked metric" in html
    assert "Holdout behavior" in html
    assert "Backend Comparison" in html
    assert "How the search moved across trials" in html
    assert "Fold Curves" in html
    assert "cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js" in html
    assert html.find("KPI Matrix") < html.find("Tuning Overview")
    assert "data-toggle-group='tuning-metrics'" in html

    context = TuningReportRenderer().build_context(result)
    assert isinstance(context, dict)
    assert context["metric_name"] == "rmse"
    assert context["split_points"][0]["label"] == "train"
    assert context["metric_catalog"][0]["metric_name"] == "rmse"


def test_tuning_renderer_renders_search_dynamics_without_plotly():
    pytest.importorskip("jinja2")
    from mlcraft.tuning.renderer import TuningReportRenderer

    fold = FoldSummary(0, {"rmse": 0.1}, {"rmse": 0.2}, -0.1, -0.2, -0.2)
    trial_0 = TrialSummary(0, {"learning_rate": 0.05}, {"rmse": 0.1}, {"rmse": 0.21}, -0.1, -0.21, -0.22, [fold])
    trial_1 = TrialSummary(1, {"learning_rate": 0.08}, {"rmse": 0.1}, {"rmse": 0.2}, -0.1, -0.2, -0.2, [fold])
    result = TuningResult(
        task_spec=TaskSpec(task_type="regression"),
        best_params={"learning_rate": 0.08},
        best_score=-0.2,
        best_trial=trial_1,
        history=[trial_0, trial_1],
        train_metrics={"rmse": 0.1},
        val_metrics={"rmse": 0.2},
        penalized_score=-0.2,
        fold_summaries=[fold],
        alpha=0.0,
        metric_name="rmse",
    )

    html = TuningReportRenderer().render(result)
    assert "Search Dynamics" in html
    assert "Trial History" in html
    assert "Top Trials" in html
    assert "window.mlcraftPendingCharts" in html


def test_tuning_renderer_metric_catalog_includes_multiple_metrics():
    from mlcraft.tuning.renderer import TuningReportRenderer

    fold = FoldSummary(
        0,
        {"rmse": 0.1, "mae": 0.08},
        {"rmse": 0.2, "mae": 0.14},
        -0.1,
        -0.2,
        -0.2,
    )
    trial = TrialSummary(0, {"depth": 4}, {"rmse": 0.1, "mae": 0.08}, {"rmse": 0.2, "mae": 0.14}, -0.1, -0.2, -0.2, [fold])
    result = TuningResult(
        task_spec=TaskSpec(task_type="regression"),
        best_params={"depth": 4},
        best_score=-0.2,
        best_trial=trial,
        history=[trial],
        train_metrics={"rmse": 0.1, "mae": 0.08},
        val_metrics={"rmse": 0.2, "mae": 0.14},
        penalized_score=-0.2,
        fold_summaries=[fold],
        alpha=0.0,
        metric_name="rmse",
    )

    context = TuningReportRenderer().build_context(result)
    metric_names = [item["metric_name"] for item in context["metric_catalog"]]

    assert metric_names == ["rmse", "mae"]
    assert context["metric_catalog"][1]["fold_rows"][0]["train_value"] == pytest.approx(0.08)


def test_tuning_artifact_writer_writes_report_and_json(tmp_path):
    pytest.importorskip("jinja2")
    from mlcraft.tuning.artifacts import write_tuning_artifacts

    fold = FoldSummary(0, {"rmse": 0.1}, {"rmse": 0.2}, -0.1, -0.2, -0.2)
    trial = TrialSummary(0, {"depth": 4}, {"rmse": 0.1}, {"rmse": 0.2}, -0.1, -0.2, -0.2, [fold])
    result = TuningResult(
        task_spec=TaskSpec(task_type="regression"),
        best_params={"depth": 4},
        best_score=-0.2,
        best_trial=trial,
        history=[trial],
        train_metrics={"rmse": 0.1},
        val_metrics={"rmse": 0.2},
        penalized_score=-0.2,
        fold_summaries=[fold],
        alpha=0.0,
        metric_name="rmse",
    )

    artifacts = write_tuning_artifacts(result, output_dir=tmp_path)
    payload = artifacts.result_path.read_text(encoding="utf-8")

    assert artifacts.report_path.exists()
    assert artifacts.result_path.exists()
    assert "mlcraft Tuning Report" in artifacts.report_path.read_text(encoding="utf-8")
    assert '"best_params"' in payload
    assert '"fold_summaries"' in payload


def test_tuning_artifact_writer_writes_full_and_shap_reports(tmp_path):
    pytest.importorskip("jinja2")
    from mlcraft.tuning.artifacts import write_tuning_artifacts

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
        alpha=0.0,
        metric_name="rmse",
    )
    evaluation = EvaluationResult(
        task_spec=TaskSpec(task_type="regression"),
        metric_rows=[MetricRow("final_test", "rmse", 0.3, -0.3, False)],
        curves={"final_test": [CurveData("residuals", np.array([0.0, 1.0]), np.array([2.0, 1.0]), "Residual", "Count")]},
    )
    shap = ShapResult(
        feature_names=["a", "b"],
        shap_values=np.array([[0.1, -0.1], [0.2, -0.2]]),
        feature_values=np.array([[1.0, 0.0], [2.0, 1.0]]),
        importance=np.array([0.15, 0.15]),
    )

    artifacts = write_tuning_artifacts(
        tuning,
        output_dir=tmp_path,
        evaluation=evaluation,
        shap=shap,
    )

    assert artifacts.full_report_path is not None
    assert artifacts.shap_report_path is not None
    assert artifacts.shap_result_path is not None
    assert artifacts.full_report_path.exists()
    assert artifacts.shap_report_path.exists()
    assert artifacts.shap_result_path.exists()
    assert "SHAP" in artifacts.full_report_path.read_text(encoding="utf-8")
