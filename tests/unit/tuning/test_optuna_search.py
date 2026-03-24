import pytest
import numpy as np

from mlcraft.core.prediction import PredictionBundle
from mlcraft.core.results import ShapResult
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
    assert result.fold_summaries[0].train_evaluation is not None
    assert result.fold_summaries[0].val_evaluation is not None
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


def test_optuna_search_supports_multi_backend_selection(monkeypatch, regression_data):
    X, y = regression_data
    X_test = {
        "num_a": np.array([7.0, 8.0]),
        "num_b": np.array([0, 1]),
    }
    y_test = np.array([6.8, 8.2])
    create_calls = []
    optimize_calls = []

    class DummyModel:
        def __init__(self, *, backend_name, task_spec, model_params=None, fit_params=None, random_state=None, logger=None):
            self.backend_name = backend_name
            self.task_spec = task_spec
            self.model_params = dict(model_params or {})
            self.fit_params = dict(fit_params or {})
            self.random_state = random_state
            self.logger = logger
            self.mean_ = 0.0

        def fit(self, X, y, *, sample_weight=None, exposure=None, eval_set=None):
            self.mean_ = float(np.mean(y))
            return self

        def predict_bundle(self, X, *, name="prediction", exposure=None):
            row_count = len(next(iter(X.values())))
            bias = 0.0 if self.backend_name == "catboost" else 1.0
            y_pred = np.full(row_count, self.mean_ + bias, dtype=float)
            return PredictionBundle(name=name, y_pred=y_pred, task_spec=self.task_spec)

    def fake_create(model_type, *, task_spec, model_params=None, fit_params=None, random_state=None, logger=None):
        backend_name = str(model_type)
        create_calls.append({"model_type": backend_name, "model_params": dict(model_params or {})})
        return DummyModel(
            backend_name=backend_name,
            task_spec=task_spec,
            model_params=model_params,
            fit_params=fit_params,
            random_state=random_state,
            logger=logger,
        )

    class FakeTrial:
        def __init__(self, number):
            self.number = number
            self.params = {}

        def suggest_float(self, name, low, high, log=False, step=None):
            value = low
            self.params[name] = value
            return value

        def suggest_int(self, name, low, high, step=1, log=False):
            value = low
            self.params[name] = value
            return value

        def set_user_attr(self, name, value):
            setattr(self, name, value)

    class FakeStudy:
        def __init__(self):
            self.best_trial = None
            self.best_value = None
            self.best_params = None

        def optimize(self, objective, n_trials):
            optimize_calls.append(int(n_trials))
            trials = [FakeTrial(index) for index in range(n_trials)]
            outcomes = []
            for trial in trials:
                value = objective(trial)
                outcomes.append((value, trial))
            best_value, best_trial = max(outcomes, key=lambda item: item[0])
            self.best_value = best_value
            self.best_trial = best_trial
            self.best_params = dict(best_trial.params)

    class FakeOptunaModule:
        class samplers:
            class TPESampler:
                def __init__(self, seed=None):
                    self.seed = seed

        @staticmethod
        def create_study(direction="maximize", sampler=None):
            return FakeStudy()

    monkeypatch.setattr("mlcraft.tuning.optuna_search.optional_import", lambda name, extra_name=None: FakeOptunaModule)
    monkeypatch.setattr("mlcraft.tuning.optuna_search.ModelFactory.create", fake_create)

    search = OptunaSearch(
        task_spec=TaskSpec(task_type="regression"),
        model_type=["xgboost", "cat"],
        n_trials=2,
        cv=2,
        alpha=0.0,
        random_state=7,
        fit_params={"early_stopping_rounds": 10},
    )

    result = search.run(X, y, X_test=X_test, y_test=y_test)

    assert result.metadata["model_type"] == "catboost"
    assert result.metadata["model_types"] == ["xgboost", "catboost"]
    assert result.metadata["selected_model_type"] == "catboost"
    assert set(result.metadata["backend_comparison"]) == {"xgboost", "catboost"}
    assert result.test_metrics is not None
    assert create_calls[-1]["model_type"] == "catboost"
    assert optimize_calls == [2, 2]


def test_optuna_search_routes_fit_target_params_to_fit_params(monkeypatch, regression_data):
    X, y = regression_data
    create_calls = []

    class DummyModel:
        def __init__(self, *, task_spec, model_params=None, fit_params=None, random_state=None, logger=None):
            self.task_spec = task_spec
            self.model_params = dict(model_params or {})
            self.fit_params = dict(fit_params or {})
            self.mean_ = 0.0

        def fit(self, X, y, *, sample_weight=None, exposure=None, eval_set=None):
            self.mean_ = float(np.mean(y))
            return self

        def predict_bundle(self, X, *, name="prediction", exposure=None):
            row_count = len(next(iter(X.values())))
            y_pred = np.full(row_count, self.mean_, dtype=float)
            return PredictionBundle(name=name, y_pred=y_pred, task_spec=self.task_spec)

    def fake_create(model_type, *, task_spec, model_params=None, fit_params=None, random_state=None, logger=None):
        create_calls.append(
            {
                "model_type": model_type,
                "model_params": dict(model_params or {}),
                "fit_params": dict(fit_params or {}),
            }
        )
        return DummyModel(
            task_spec=task_spec,
            model_params=model_params,
            fit_params=fit_params,
            random_state=random_state,
            logger=logger,
        )

    class FakeTrial:
        def __init__(self, number):
            self.number = number
            self.params = {}

        def suggest_float(self, name, low, high, log=False, step=None):
            value = low
            self.params[name] = value
            return value

        def suggest_int(self, name, low, high, step=1, log=False):
            value = high if name == "num_boost_round" else low
            self.params[name] = value
            return value

        def set_user_attr(self, name, value):
            setattr(self, name, value)

    class FakeStudy:
        def __init__(self):
            self.best_trial = None
            self.best_value = None
            self.best_params = None

        def optimize(self, objective, n_trials):
            trial = FakeTrial(0)
            self.best_value = objective(trial)
            self.best_trial = trial
            self.best_params = dict(trial.params)

    class FakeOptunaModule:
        class samplers:
            class TPESampler:
                def __init__(self, seed=None):
                    self.seed = seed

        @staticmethod
        def create_study(direction="maximize", sampler=None):
            return FakeStudy()

    monkeypatch.setattr("mlcraft.tuning.optuna_search.optional_import", lambda name, extra_name=None: FakeOptunaModule)
    monkeypatch.setattr("mlcraft.tuning.optuna_search.ModelFactory.create", fake_create)

    search = OptunaSearch(
        task_spec=TaskSpec(task_type="regression"),
        model_type="xgboost",
        n_trials=1,
        cv=2,
        search_space={
            "num_boost_round": {"type": "int", "low": 150, "high": 350, "target": "fit"},
            "max_depth": {"type": "int", "low": 3, "high": 3},
        },
    )

    result = search.run(X, y)

    assert create_calls
    assert create_calls[0]["fit_params"]["num_boost_round"] == 350
    assert "num_boost_round" not in create_calls[0]["model_params"]
    assert result.best_params["num_boost_round"] == 350


def test_optuna_search_run_with_artifacts_writes_default_outputs(monkeypatch, regression_data, tmp_path):
    X, y = regression_data

    class DummyModel:
        def __init__(self, *, task_spec, model_params=None, fit_params=None, random_state=None, logger=None):
            self.task_spec = task_spec
            self.mean_ = 0.0

        def fit(self, X, y, *, sample_weight=None, exposure=None, eval_set=None):
            self.mean_ = float(np.mean(y))
            return self

        def predict_bundle(self, X, *, name="prediction", exposure=None):
            row_count = len(next(iter(X.values())))
            y_pred = np.full(row_count, self.mean_, dtype=float)
            return PredictionBundle(name=name, y_pred=y_pred, task_spec=self.task_spec)

    class FakeTrial:
        def __init__(self, number):
            self.number = number
            self.params = {}

        def suggest_float(self, name, low, high, log=False, step=None):
            value = low
            self.params[name] = value
            return value

        def suggest_int(self, name, low, high, step=1, log=False):
            value = low
            self.params[name] = value
            return value

        def set_user_attr(self, name, value):
            setattr(self, name, value)

    class FakeStudy:
        def __init__(self):
            self.best_trial = None
            self.best_value = None
            self.best_params = None

        def optimize(self, objective, n_trials):
            trial = FakeTrial(0)
            value = objective(trial)
            self.best_trial = trial
            self.best_value = value
            self.best_params = dict(trial.params)

    class FakeOptunaModule:
        class samplers:
            class TPESampler:
                def __init__(self, seed=None):
                    self.seed = seed

        @staticmethod
        def create_study(direction="maximize", sampler=None):
            return FakeStudy()

    monkeypatch.setattr("mlcraft.tuning.optuna_search.optional_import", lambda name, extra_name=None: FakeOptunaModule)
    monkeypatch.setattr(
        "mlcraft.tuning.optuna_search.ModelFactory.create",
        lambda *args, **kwargs: DummyModel(
            task_spec=kwargs["task_spec"],
            model_params=kwargs.get("model_params"),
            fit_params=kwargs.get("fit_params"),
            random_state=kwargs.get("random_state"),
            logger=kwargs.get("logger"),
        ),
    )

    search = OptunaSearch(
        task_spec=TaskSpec(task_type="regression"),
        model_type="xgboost",
        n_trials=1,
        cv=2,
        fit_params={"num_boost_round": 5},
    )

    result, artifacts = search.run_with_artifacts(X, y, output_dir=tmp_path)

    assert result.best_params
    assert artifacts.report_path.exists()
    assert artifacts.result_path.exists()


def test_optuna_search_run_with_artifacts_can_write_shap_and_full_report(monkeypatch, regression_data, tmp_path):
    X, y = regression_data
    X_test = {
        "num_a": np.array([7.0, 8.0]),
        "num_b": np.array([0, 1]),
    }
    y_test = np.array([6.8, 8.2])

    class DummyModel:
        def __init__(self, *, task_spec, model_params=None, fit_params=None, random_state=None, logger=None):
            self.task_spec = task_spec
            self.mean_ = 0.0
            self.model_ = object()

        def fit(self, X, y, *, sample_weight=None, exposure=None, eval_set=None):
            self.mean_ = float(np.mean(y))
            return self

        def predict_bundle(self, X, *, name="prediction", exposure=None):
            row_count = len(next(iter(X.values())))
            y_pred = np.full(row_count, self.mean_, dtype=float)
            return PredictionBundle(name=name, y_pred=y_pred, task_spec=self.task_spec)

    class FakeTrial:
        def __init__(self, number):
            self.number = number
            self.params = {}

        def suggest_float(self, name, low, high, log=False, step=None):
            value = low
            self.params[name] = value
            return value

        def suggest_int(self, name, low, high, step=1, log=False):
            value = low
            self.params[name] = value
            return value

        def set_user_attr(self, name, value):
            setattr(self, name, value)

    class FakeStudy:
        def __init__(self):
            self.best_trial = None
            self.best_value = None
            self.best_params = None

        def optimize(self, objective, n_trials):
            trial = FakeTrial(0)
            value = objective(trial)
            self.best_trial = trial
            self.best_value = value
            self.best_params = dict(trial.params)

    class FakeOptunaModule:
        class samplers:
            class TPESampler:
                def __init__(self, seed=None):
                    self.seed = seed

        @staticmethod
        def create_study(direction="maximize", sampler=None):
            return FakeStudy()

    monkeypatch.setattr("mlcraft.tuning.optuna_search.optional_import", lambda name, extra_name=None: FakeOptunaModule)
    monkeypatch.setattr(
        "mlcraft.tuning.optuna_search.ModelFactory.create",
        lambda *args, **kwargs: DummyModel(
            task_spec=kwargs["task_spec"],
            model_params=kwargs.get("model_params"),
            fit_params=kwargs.get("fit_params"),
            random_state=kwargs.get("random_state"),
            logger=kwargs.get("logger"),
        ),
    )
    monkeypatch.setattr(
        "mlcraft.tuning.optuna_search.ShapAnalyzer.compute",
        lambda self, model, X, sample_weight=None, max_samples=None, interaction_values=False: ShapResult(
            feature_names=["num_a", "num_b"],
            shap_values=np.array([[0.1, -0.1], [0.2, -0.2]]),
            feature_values=np.array([[7.0, 0.0], [8.0, 1.0]]),
            importance=np.array([0.15, 0.15]),
        ),
    )

    search = OptunaSearch(
        task_spec=TaskSpec(task_type="regression"),
        model_type="xgboost",
        n_trials=1,
        cv=2,
        fit_params={"num_boost_round": 5},
    )

    result, artifacts = search.run_with_artifacts(
        X,
        y,
        X_test=X_test,
        y_test=y_test,
        output_dir=tmp_path,
        compute_shap=True,
    )

    assert result.best_params
    assert artifacts.report_path.exists()
    assert artifacts.result_path.exists()
    assert artifacts.full_report_path is not None and artifacts.full_report_path.exists()
    assert artifacts.shap_report_path is not None and artifacts.shap_report_path.exists()
    assert artifacts.shap_result_path is not None and artifacts.shap_result_path.exists()
