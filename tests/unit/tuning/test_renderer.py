import pytest

import numpy as np
import optuna

from mlcraft.core.results import CurveData, EvaluationResult, FoldSummary, MetricRow, ShapResult, TrialSummary, TuningResult
from mlcraft.core.task import TaskSpec


def test_tuning_renderer_generates_html():
    pytest.importorskip("matplotlib")
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
    assert "Generalization Overview" in html
    assert "Holdout behavior" in html
    assert "Backend Comparison" in html
    assert "Optimized Metric by Fold" in html
    assert "Generalization Gap by Fold" in html
    assert "rmse by Fold" in html
    assert "Fold Curves" in html
    assert html.find("Backend Comparison") < html.find("Tuning Overview")
    assert "Optuna Plotly" not in html

    context = TuningReportRenderer().build_context(result)
    assert isinstance(context, dict)
    assert context["metric_name"] == "rmse"
    assert context["split_points"][0]["label"] == "train"


def test_tuning_renderer_renders_official_optuna_visuals():
    pytest.importorskip("matplotlib")
    pytest.importorskip("jinja2")
    pytest.importorskip("plotly")
    from mlcraft.tuning.renderer import TuningReportRenderer

    study = optuna.create_study(direction="maximize")
    distribution = optuna.distributions.FloatDistribution(0.01, 0.2)
    study.add_trial(
        optuna.trial.create_trial(
            params={"learning_rate": 0.05},
            distributions={"learning_rate": distribution},
            value=0.81,
        )
    )
    study.add_trial(
        optuna.trial.create_trial(
            params={"learning_rate": 0.08},
            distributions={"learning_rate": distribution},
            value=0.87,
        )
    )

    fold = FoldSummary(0, {"rmse": 0.1}, {"rmse": 0.2}, -0.1, -0.2, -0.2)
    trial = TrialSummary(0, {"learning_rate": 0.08}, {"rmse": 0.1}, {"rmse": 0.2}, -0.1, -0.2, -0.2, [fold])
    result = TuningResult(
        task_spec=TaskSpec(task_type="regression"),
        best_params={"learning_rate": 0.08},
        best_score=0.87,
        best_trial=trial,
        history=[trial],
        train_metrics={"rmse": 0.1},
        val_metrics={"rmse": 0.2},
        penalized_score=0.87,
        fold_summaries=[fold],
        alpha=0.0,
        metric_name="rmse",
        study=study,
    )

    html = TuningReportRenderer().render(result)
    assert "Optimization History" in html
    assert "Optuna Plotly" in html


def test_tuning_curve_plot_keeps_same_color_per_fold_and_dash_per_split():
    pytest.importorskip("plotly")
    from mlcraft.tuning.renderer import TuningReportRenderer

    curve_group = {
        "curve_name": "roc",
        "title": "Roc by fold",
        "x_label": "False positive rate",
        "y_label": "True positive rate",
        "series": [
            {
                "series_name": "Fold 0 Train",
                "prediction_name": "train",
                "fold_index": 0,
                "split": "train",
                "x": [0.0, 1.0],
                "y": [0.0, 1.0],
                "metadata": {},
            },
            {
                "series_name": "Fold 0 Validation",
                "prediction_name": "validation",
                "fold_index": 0,
                "split": "validation",
                "x": [0.0, 1.0],
                "y": [0.0, 0.8],
                "metadata": {},
            },
        ],
    }

    figure = TuningReportRenderer()._plot_curve_group_figure(curve_group)

    assert figure.data[0].line.color == figure.data[1].line.color
    assert figure.data[0].line.dash == "solid"
    assert figure.data[1].line.dash == "dash"


def test_tuning_artifact_writer_writes_report_and_json(tmp_path):
    pytest.importorskip("matplotlib")
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
    pytest.importorskip("matplotlib")
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
