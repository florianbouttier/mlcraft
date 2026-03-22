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
    pytest.skip("No gradient boosting backend installed.")


def test_optuna_search_runs_small_cv(classification_data):
    pytest.importorskip("optuna")
    backend = _available_backend()
    X, y = classification_data
    search = OptunaSearch(
        task_spec=TaskSpec(task_type="classification"),
        model_type=backend,
        n_trials=2,
        cv=2,
        alpha=0.1,
        random_state=5,
        fit_params={"num_boost_round": 5},
    )
    result = search.run(X, y)
    assert result.best_trial.trial_number in {0, 1}
    assert result.alpha == 0.1
    assert result.metric_name == "roc_auc"

