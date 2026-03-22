"""Task specification objects and defaults."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class TaskType(str, Enum):
    """Enumerate the task families supported by the library."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    POISSON = "poisson"


class PredictionType(str, Enum):
    """Enumerate the default semantics of model outputs."""

    VALUE = "value"
    PROBABILITY = "probability"
    MEAN_COUNT = "mean_count"


_TASK_DEFAULTS: dict[TaskType, dict[str, Any]] = {
    TaskType.REGRESSION: {
        "loss_name": "squared_error",
        "eval_metric": "rmse",
        "higher_is_better": False,
        "prediction_type": PredictionType.VALUE,
    },
    TaskType.CLASSIFICATION: {
        "loss_name": "logloss",
        "eval_metric": "roc_auc",
        "higher_is_better": True,
        "prediction_type": PredictionType.PROBABILITY,
    },
    TaskType.POISSON: {
        "loss_name": "poisson_deviance",
        "eval_metric": "poisson_deviance",
        "higher_is_better": False,
        "prediction_type": PredictionType.MEAN_COUNT,
    },
}


def resolve_task_spec_defaults(task_type: TaskType | str) -> dict[str, Any]:
    """Return the strong defaults attached to a task family.

    Args:
        task_type: Task family to normalize.

    Returns:
        dict[str, Any]: Default loss, metric, optimization direction, and
        prediction semantics for the task.

    Example:
        >>> resolve_task_spec_defaults("classification")["eval_metric"]
        'roc_auc'
    """

    normalized = task_type if isinstance(task_type, TaskType) else TaskType(str(task_type))
    return dict(_TASK_DEFAULTS[normalized])


@dataclass
class TaskSpec:
    """Describe the shared prediction contract for a modeling task.

    `TaskSpec` centralizes choices that should stay consistent across models,
    evaluators, prediction bundles, and tuning runs. When optional fields are
    omitted, the class injects strong defaults derived from `task_type`.

    Args:
        task_type: Modeling family to configure.
        loss_name: Backend-facing loss or objective name override.
        eval_metric: Canonical metric name used across the package.
        higher_is_better: Optimization direction for the canonical metric.
        prediction_type: Expected semantics of model outputs.

    Example:
        >>> task = TaskSpec(task_type="classification")
        >>> task.eval_metric
        'roc_auc'

    Notes:
        Defaults are resolved in `__post_init__`, so downstream objects can
        reuse a fully specified task contract without repeating configuration.
    """

    task_type: TaskType | str
    loss_name: str | None = None
    eval_metric: str | None = None
    higher_is_better: bool | None = None
    prediction_type: PredictionType | str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task_type, TaskType):
            self.task_type = TaskType(str(self.task_type))
        defaults = resolve_task_spec_defaults(self.task_type)
        self.loss_name = self.loss_name or defaults["loss_name"]
        self.eval_metric = self.eval_metric or defaults["eval_metric"]
        if self.higher_is_better is None:
            self.higher_is_better = bool(defaults["higher_is_better"])
        if self.prediction_type is None:
            self.prediction_type = defaults["prediction_type"]
        elif not isinstance(self.prediction_type, PredictionType):
            self.prediction_type = PredictionType(str(self.prediction_type))

    def is_classification(self) -> bool:
        """Return whether the task is binary classification.

        Returns:
            bool: `True` for binary classification tasks.
        """

        return self.task_type == TaskType.CLASSIFICATION

    def is_regression(self) -> bool:
        """Return whether the task is regression.

        Returns:
            bool: `True` for regression tasks.
        """

        return self.task_type == TaskType.REGRESSION

    def is_poisson(self) -> bool:
        """Return whether the task is Poisson count modeling.

        Returns:
            bool: `True` for Poisson tasks.
        """

        return self.task_type == TaskType.POISSON

    def to_dict(self) -> dict[str, Any]:
        """Serialize the task specification into a plain dictionary.

        Returns:
            dict[str, Any]: Serialized task specification.
        """

        return {
            "task_type": self.task_type.value,
            "loss_name": self.loss_name,
            "eval_metric": self.eval_metric,
            "higher_is_better": self.higher_is_better,
            "prediction_type": self.prediction_type.value if isinstance(self.prediction_type, PredictionType) else self.prediction_type,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskSpec":
        """Build a `TaskSpec` from serialized data.

        Args:
            payload: Serialized payload produced by `to_dict`.

        Returns:
            TaskSpec: Reconstructed task specification.
        """

        return cls(**payload)
