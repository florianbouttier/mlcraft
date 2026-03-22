import numpy as np

from mlcraft.core.prediction import PredictionBundle
from mlcraft.core.task import TaskSpec
from mlcraft.evaluation.evaluator import Evaluator


def test_evaluator_handles_single_prediction_classification():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.8, 0.2, 0.9])
    bundle = PredictionBundle(
        name="model_a",
        y_pred=(y_score >= 0.5).astype(int),
        y_score=y_score,
        task_spec=TaskSpec(task_type="classification"),
    )
    result = Evaluator().evaluate(y_true, bundle)
    assert "model_a" in result.metrics_by_prediction()
    assert len(result.curves["model_a"]) == 3


def test_evaluator_compares_multiple_predictions(regression_data):
    _, y_true = regression_data
    bundles = [
        PredictionBundle(name="m1", y_pred=y_true),
        PredictionBundle(name="m2", y_pred=y_true + 0.5),
    ]
    result = Evaluator().evaluate(y_true, bundles, task_spec=TaskSpec(task_type="regression"))
    assert set(result.metrics_by_prediction().keys()) == {"m1", "m2"}


def test_evaluator_handles_poisson(poisson_data):
    _, y_true, exposure = poisson_data
    bundle = PredictionBundle(name="pois", y_pred=np.array([0.5, 0.8, 0.9, 1.2, 1.4, 2.1]), task_spec=TaskSpec(task_type="poisson"))
    result = Evaluator().evaluate(y_true, bundle, exposure=exposure)
    assert "poisson_deviance" in result.metrics_by_prediction()["pois"]

