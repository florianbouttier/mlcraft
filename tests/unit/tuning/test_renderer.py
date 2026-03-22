import pytest

from mlcraft.core.results import FoldSummary, TrialSummary, TuningResult
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
    )
    html = TuningReportRenderer().render(result)
    assert "Best Trial" in html
    assert "Optimization History" in html

