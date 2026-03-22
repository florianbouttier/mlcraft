"""Search space helpers for Optuna-based tuning."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from mlcraft.core.task import TaskSpec, TaskType

SearchSpaceSpec = dict[str, dict[str, Any]]


def default_search_space(model_type: str, task_spec: TaskSpec) -> SearchSpaceSpec:
    """Return the default Optuna search space for one backend and task.

    Args:
        model_type: Canonical backend name such as `xgboost`.
        task_spec: Shared task specification.

    Returns:
        SearchSpaceSpec: Declarative search space compatible with
        `suggest_params`.
    """

    common = {
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
    }
    if model_type == "xgboost":
        common.update(
            {
                "max_depth": {"type": "int", "low": 3, "high": 10},
                "subsample": {"type": "float", "low": 0.6, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
                "min_child_weight": {"type": "float", "low": 1.0, "high": 10.0},
            }
        )
    elif model_type == "lightgbm":
        common.update(
            {
                "num_leaves": {"type": "int", "low": 15, "high": 127},
                "feature_fraction": {"type": "float", "low": 0.6, "high": 1.0},
                "bagging_fraction": {"type": "float", "low": 0.6, "high": 1.0},
            }
        )
    elif model_type == "catboost":
        common.update(
            {
                "depth": {"type": "int", "low": 4, "high": 10},
                "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 10.0},
                "border_count": {"type": "int", "low": 32, "high": 255},
            }
        )
    if task_spec.task_type == TaskType.POISSON:
        common.setdefault("learning_rate", {"type": "float", "low": 0.01, "high": 0.2, "log": True})
    return common


def merge_search_spaces(base: SearchSpaceSpec, override: SearchSpaceSpec | None = None) -> SearchSpaceSpec:
    """Merge a default search space with user overrides.

    Args:
        base: Default search space.
        override: Optional user-provided override space.

    Returns:
        SearchSpaceSpec: Merged search space.
    """

    result = deepcopy(base)
    for key, value in (override or {}).items():
        result[key] = dict(value)
    return result


def suggest_params(trial, search_space: SearchSpaceSpec) -> dict[str, Any]:
    """Suggest trial parameters from a declarative search space.

    Args:
        trial: Optuna trial object.
        search_space: Declarative search space specification.

    Returns:
        dict[str, Any]: Concrete parameter values for the current trial.
    """

    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        kind = spec["type"]
        if kind == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=bool(spec.get("log", False)), step=spec.get("step"))
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], step=int(spec.get("step", 1)), log=bool(spec.get("log", False)))
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unsupported search space type: {kind}")
    return params
