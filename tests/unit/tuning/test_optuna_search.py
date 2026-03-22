import pytest
import numpy as np

from mlcraft.core.prediction import PredictionBundle
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


def test_optuna_search_evaluates_optional_holdout(monkeypatch, regression_data):
    pytest.importorskip("optuna")
    X, y = regression_data
    X_test = {
        "num_a": np.array([7.0, 8.0]),
        "num_b": np.array([0, 1]),
    }
    y_test = np.array([6.8, 8.2])
    create_calls = []

    class DummyModel:
        def __init__(self, *, task_spec, model_params=None, fit_params=None, random_state=None, logger=None):
            self.task_spec = task_spec
            self.model_params = dict(model_params or {})
            self.fit_params = dict(fit_params or {})
            self.random_state = random_state
            self.logger = logger
            self.fit_history = []
            self.mean_ = 0.0

        def fit(self, X, y, *, sample_weight=None, exposure=None, eval_set=None):
            self.mean_ = float(np.mean(y))
            self.fit_history.append(
                {
                    "row_count": len(y),
                    "sample_weight": sample_weight,
                    "exposure": exposure,
                    "eval_set": eval_set,
                }
            )
            return self

        def predict_bundle(self, X, *, name="prediction", exposure=None):
            row_count = len(next(iter(X.values())))
            y_pred = np.full(row_count, self.mean_, dtype=float)
            return PredictionBundle(name=name, y_pred=y_pred, task_spec=self.task_spec)

    def fake_create(model_type, *, task_spec, model_params=None, fit_params=None, random_state=None, logger=None):
        model = DummyModel(
            task_spec=task_spec,
            model_params=model_params,
            fit_params=fit_params,
            random_state=random_state,
            logger=logger,
        )
        create_calls.append(
            {
                "model_type": model_type,
                "model_params": dict(model_params or {}),
                "fit_params": dict(fit_params or {}),
                "model": model,
            }
        )
        return model

    monkeypatch.setattr("mlcraft.tuning.optuna_search.ModelFactory.create", fake_create)
    search = OptunaSearch(
        task_spec=TaskSpec(task_type="regression"),
        model_type="xgboost",
        n_trials=1,
        cv=2,
        alpha=0.0,
        random_state=7,
        fit_params={"early_stopping_rounds": 10},
    )

    result = search.run(X, y, X_test=X_test, y_test=y_test)

    assert len(create_calls) == 3
    assert result.test_evaluation is not None
    assert result.test_metrics is not None
    assert "rmse" in result.test_metrics
    assert result.test_score == pytest.approx(-result.test_metrics["rmse"])
    assert "early_stopping_rounds" not in create_calls[-1]["fit_params"]
    assert create_calls[-1]["model"].fit_history[0]["eval_set"] is None


def test_optuna_search_requires_complete_holdout_pair(regression_data):
    pytest.importorskip("optuna")
    X, y = regression_data
    search = OptunaSearch(
        task_spec=TaskSpec(task_type="regression"),
        model_type="xgboost",
        n_trials=1,
        cv=2,
    )

    with pytest.raises(ValueError, match="X_test and y_test must be provided together."):
        search.run(X, y, X_test=X)
