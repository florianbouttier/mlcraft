"""Backend objective and metric mapping helpers."""

from __future__ import annotations

from mlcraft.core.task import TaskSpec, TaskType
from mlcraft.metrics.registry import MetricRegistry, default_metric_registry


def resolve_backend_objective(backend: str, task_spec: TaskSpec) -> str:
    """Map a TaskSpec to the backend-native objective/loss name."""

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
    """Resolve the native metric name for the TaskSpec evaluation metric."""

    registry = metric_registry or default_metric_registry
    return registry.backend_name(task_spec.eval_metric, backend)


def backend_seed_key(backend: str) -> str:
    """Return the seed parameter key for a backend."""

    return {
        "xgboost": "seed",
        "lightgbm": "seed",
        "catboost": "random_seed",
    }[backend]

