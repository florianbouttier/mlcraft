import pytest

from mlcraft.core.task import TaskSpec
from mlcraft.tuning.search_space import default_search_space, split_suggested_params


def test_default_search_spaces_expose_tree_count_and_broader_backend_params():
    classification_task = TaskSpec(task_type="classification")

    xgboost_space = default_search_space("xgboost", classification_task)
    assert xgboost_space["num_boost_round"]["target"] == "fit"
    assert "gamma" in xgboost_space
    assert "lambda" in xgboost_space
    assert "alpha" in xgboost_space

    lightgbm_space = default_search_space("lightgbm", classification_task)
    assert lightgbm_space["num_boost_round"]["target"] == "fit"
    assert "lambda_l1" in lightgbm_space
    assert "lambda_l2" in lightgbm_space
    assert "min_gain_to_split" in lightgbm_space

    catboost_space = default_search_space("catboost", classification_task)
    assert "iterations" in catboost_space
    assert "random_strength" in catboost_space
    assert "bagging_temperature" in catboost_space
    assert "rsm" in catboost_space


def test_split_suggested_params_separates_fit_and_model_targets():
    params = {
        "num_boost_round": 350,
        "max_depth": 6,
        "learning_rate": 0.05,
    }
    search_space = {
        "num_boost_round": {"type": "int", "low": 100, "high": 500, "target": "fit"},
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
    }

    model_params, fit_params = split_suggested_params(params, search_space)

    assert model_params == {"max_depth": 6, "learning_rate": 0.05}
    assert fit_params == {"num_boost_round": 350}
