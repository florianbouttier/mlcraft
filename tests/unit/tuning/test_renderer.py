import pytest

import numpy as np
import optuna

from mlcraft.core.results import CurveData, EvaluationResult, FoldSummary, MetricRow, TrialSummary, TuningResult
from mlcraft.core.task import TaskSpec


def test_tuning_renderer_generates_html():
    pytest.importorskip("matplotlib")
    pytest.importorskip("jinja2")
    from mlcraft.tuning.renderer import TuningReportRenderer

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
    assert "Official study plots" in html
    assert "Holdout behavior" in html


def test_tuning_renderer_renders_official_optuna_visuals():
    pytest.importorskip("matplotlib")
    pytest.importorskip("jinja2")
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
