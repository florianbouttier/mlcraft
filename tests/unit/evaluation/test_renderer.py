import pytest

from mlcraft.core.prediction import PredictionBundle
from mlcraft.core.task import TaskSpec
from mlcraft.evaluation.evaluator import Evaluator


def test_evaluation_renderer_generates_html():
    pytest.importorskip("matplotlib")
    pytest.importorskip("jinja2")
    from mlcraft.evaluation.renderer import EvaluationReportRenderer

    result = Evaluator().evaluate(
        [0, 1, 0, 1],
        PredictionBundle(name="m1", y_pred=[0, 1, 0, 1], y_score=[0.1, 0.8, 0.2, 0.9], task_spec=TaskSpec(task_type="classification")),
    )
    html = EvaluationReportRenderer().render(result)
    assert "mlcraft Evaluation Report" in html
    assert "Graphical leaderboard" in html
    assert "Curves on shared axes" in html
    assert "Plot Zoom" in html
    assert 'data-plot-scale="0.85"' in html


def test_evaluation_renderer_builds_dictionary_context():
    from mlcraft.evaluation.renderer import EvaluationReportRenderer

    result = Evaluator().evaluate(
        [0, 1, 0, 1],
        PredictionBundle(name="m1", y_pred=[0, 1, 0, 1], y_score=[0.1, 0.8, 0.2, 0.9], task_spec=TaskSpec(task_type="classification")),
    )
    context = EvaluationReportRenderer().build_context(result)
    assert isinstance(context, dict)
    assert context["primary_metric_name"] == "roc_auc"
    assert "curve_groups" in context
