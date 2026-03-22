"""Backend objective and metric mapping helpers."""

from __future__ import annotations

from mlcraft.core.task import TaskSpec, TaskType
from mlcraft.metrics.registry import MetricRegistry, default_metric_registry


def resolve_backend_objective(backend: str, task_spec: TaskSpec) -> str:
    """Map a task specification to the backend-native objective.

    Args:
        backend: Backend name such as `xgboost`.
        task_spec: Shared task specification.

    Returns:
        str: Backend-native objective or loss name.
    """

    mapping = {
        "xgboost": {
            TaskType.REGRESSION: "reg:squarederror",
            TaskType.CLASSIFICATION: "binary:logistic",
            TaskType.POISSON: "count:poisson",
        },
        "lightgbm": {
            TaskType.REGRESSION: "regression",
            TaskType.CLASSIFICATION: "binary",
            TaskType.POISSON: "poisson",
        },
        "catboost": {
            TaskType.REGRESSION: "RMSE",
            TaskType.CLASSIFICATION: "Logloss",
            TaskType.POISSON: "Poisson",
        },
    }
    return mapping[backend][task_spec.task_type]


def resolve_backend_metric(
    backend: str,
    task_spec: TaskSpec,
    *,
    metric_registry: MetricRegistry | None = None,
) -> str | None:
    """Resolve the backend-native alias for the canonical evaluation metric.

    Args:
        backend: Backend name such as `xgboost`.
        task_spec: Shared task specification.
        metric_registry: Optional metric registry override.

    Returns:
        str | None: Backend-native metric alias when available.
    """

    registry = metric_registry or default_metric_registry
    return registry.backend_name(task_spec.eval_metric, backend)


def backend_seed_key(backend: str) -> str:
    """Return the random seed parameter name expected by a backend.

    Args:
        backend: Backend name such as `xgboost`.

    Returns:
        str: Backend-native parameter key for random seed control.
    """

    return {
        "xgboost": "seed",
        "lightgbm": "seed",
        "catboost": "random_seed",
    }[backend]
