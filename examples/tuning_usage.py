from mlcraft.core.task import TaskSpec
from mlcraft.tuning.optuna_search import OptunaSearch

# This example assumes an optional backend and Optuna are installed.
search = OptunaSearch(
    task_spec=TaskSpec(task_type="classification"),
    model_type="xgboost",
    n_trials=20,
    cv=5,
    alpha=0.1,
    model_params={"max_depth": 4},
    fit_params={"num_boost_round": 100, "early_stopping_rounds": 20},
)

print(search)

