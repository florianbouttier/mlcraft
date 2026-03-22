import numpy as np

from mlcraft.core.task import TaskSpec
from mlcraft.evaluation.evaluator import Evaluator
from mlcraft.models.base import BaseGBMModel
from mlcraft.split.train_test import train_test_split_random


class DummyRegressorModel(BaseGBMModel):
    backend_name = "dummy"

    def _fit_backend(self, X, y, *, sample_weight=None, exposure=None, eval_set=None, metadata=None):
        return {"mean": float(np.mean(y))}

    def _predict_backend(self, X, *, metadata=None):
        return np.full(X.shape[0], self.model_["mean"], dtype=float)


def test_basic_pipeline_runs_end_to_end(regression_data):
    X, y = regression_data
    X_train, X_test, y_train, y_test = train_test_split_random(X, y, test_size=2, random_state=3)
    model = DummyRegressorModel(task_spec=TaskSpec(task_type="regression"))
    model.fit(X_train, y_train)
    bundle = model.predict_bundle(X_test, name="dummy")
    result = Evaluator().evaluate(y_test, bundle)
    assert "dummy" in result.metrics_by_prediction()

