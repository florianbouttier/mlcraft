import pytest

from mlcraft.core.task import TaskSpec
from mlcraft.models.factory import ModelFactory


def test_xgboost_fit_predict_smoke(classification_data):
    pytest.importorskip("xgboost")
    X, y = classification_data
    model = ModelFactory.create(
        "xgboost",
        task_spec=TaskSpec(task_type="classification"),
        fit_params={"num_boost_round": 5},
        random_state=7,
    )
    model.fit(X, y)
    pred = model.predict(X)
    proba = model.predict_proba(X)
    assert pred.shape[0] == y.shape[0]
    assert proba.shape == (y.shape[0], 2)

