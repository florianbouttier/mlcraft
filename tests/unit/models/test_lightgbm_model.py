import pytest

from mlcraft.core.task import TaskSpec
from mlcraft.models.factory import ModelFactory


def test_lightgbm_fit_predict_smoke(regression_data):
    pytest.importorskip("lightgbm")
    X, y = regression_data
    model = ModelFactory.create(
        "lightgbm",
        task_spec=TaskSpec(task_type="regression"),
        fit_params={"num_boost_round": 5},
        random_state=7,
    )
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape[0] == y.shape[0]

