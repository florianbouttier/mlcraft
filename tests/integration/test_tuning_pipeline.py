import pytest

from mlcraft.core.task import TaskSpec
from mlcraft.tuning.optuna_search import OptunaSearch


def _available_backend():
    for name in ("xgboost", "lightgbm", "catboost"):
        try:
            __import__(name)
            return name
        except ModuleNotFoundError:
            continue
    pytest.skip("No backend installed for tuning integration.")


def test_tuning_pipeline_builds_result(classification_data):
    pytest.importorskip("optuna")
    backend = _available_backend()
    X, y = classification_data
    result = OptunaSearch(
        task_spec=TaskSpec(task_type="classification"),
        model_type=backend,
        n_trials=2,
        cv=2,
        fit_params={"num_boost_round": 5},
        random_state=11,
    ).run(X, y)
    assert result.best_params
    assert result.history

