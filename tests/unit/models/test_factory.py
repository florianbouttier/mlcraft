from mlcraft.core.task import TaskSpec
from mlcraft.models.factory import ModelFactory
from mlcraft.models.xgboost_model import XGBoostModel


def test_model_factory_creates_expected_wrapper():
    model = ModelFactory.create("xgboost", task_spec=TaskSpec(task_type="regression"), model_params={"max_depth": 3})
    assert isinstance(model, XGBoostModel)
    params = model.get_params()
    assert params["model_params"]["max_depth"] == 3


def test_model_set_params_updates_model_params():
    model = ModelFactory.create("xgboost", task_spec=TaskSpec(task_type="classification"))
    model.set_params(max_depth=4, fit_params={"num_boost_round": 10})
    assert model.model_params["max_depth"] == 4
    assert model.fit_params["num_boost_round"] == 10

