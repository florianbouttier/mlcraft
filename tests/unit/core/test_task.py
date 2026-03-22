from mlcraft.core.prediction import PredictionBundle, resolve_task_spec
from mlcraft.core.task import PredictionType, TaskSpec


def test_task_spec_defaults():
    regression = TaskSpec(task_type="regression")
    classification = TaskSpec(task_type="classification")
    poisson = TaskSpec(task_type="poisson")
    assert regression.eval_metric == "rmse"
    assert classification.prediction_type == PredictionType.PROBABILITY
    assert poisson.higher_is_better is False


def test_resolve_task_spec_from_bundle():
    task = TaskSpec(task_type="classification")
    bundle = PredictionBundle(name="pred", y_pred=[0, 1], y_score=[0.2, 0.8], task_spec=task)
    assert resolve_task_spec(None, bundle) is task

