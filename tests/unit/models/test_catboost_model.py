import pytest

from mlcraft.core.task import TaskSpec
from mlcraft.models.factory import ModelFactory


def test_catboost_fit_predict_smoke(classification_data):
    pytest.importorskip("catboost")
    X, y = classification_data
    model = ModelFactory.create(
        "catboost",
        task_spec=TaskSpec(task_type="classification"),
        model_params={"iterations": 5},
        random_state=7,
    )
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape[0] == y.shape[0]

