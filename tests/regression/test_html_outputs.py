import numpy as np
import pytest

from mlcraft.core.prediction import PredictionBundle
from mlcraft.core.task import TaskSpec
from mlcraft.evaluation.evaluator import Evaluator


def test_evaluation_html_contains_expected_sections():
    pytest.importorskip("matplotlib")
    pytest.importorskip("jinja2")
    from mlcraft.evaluation.renderer import EvaluationReportRenderer

    result = Evaluator().evaluate(
        np.array([0, 1, 0, 1]),
        PredictionBundle(name="baseline", y_pred=np.array([0, 1, 0, 1]), y_score=np.array([0.1, 0.9, 0.2, 0.8]), task_spec=TaskSpec(task_type="classification")),
    )
    html = EvaluationReportRenderer().render(result)
    assert "Metrics" in html
    assert "Curves" in html
