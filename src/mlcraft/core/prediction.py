"""Prediction bundle helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from mlcraft.core.task import TaskSpec
from mlcraft.utils.serialization import to_serializable


@runtime_checkable
class TaskSpecCarrier(Protocol):
    """Define the minimal protocol for objects carrying a task specification."""

    task_spec: TaskSpec | None


def resolve_task_spec(
    task_spec: TaskSpec | None = None,
    *carriers: TaskSpecCarrier | None,
) -> TaskSpec | None:
    """Resolve a task specification from explicit input or context objects.

    Args:
        task_spec: Explicit task specification provided by the caller.
        *carriers: Candidate objects exposing a `task_spec` attribute.

    Returns:
        TaskSpec | None: First available task specification, or `None` when no
        context can provide one.

    Example:
        >>> task = TaskSpec(task_type="regression")
        >>> bundle = PredictionBundle(name="baseline", y_pred=[1.0], task_spec=task)
        >>> resolve_task_spec(None, bundle) is task
        True
    """

    if task_spec is not None:
        return task_spec
    for carrier in carriers:
        if carrier is None:
            continue
        candidate = getattr(carrier, "task_spec", None)
        if candidate is not None:
            return candidate
    return None


@dataclass
class PredictionBundle:
    """Store one set of predictions together with its modeling context.

    The bundle keeps raw predictions, optional scores, and optional metadata in
    a form that can move between evaluation, tuning, and reporting without
    duplicating task configuration.

    Args:
        name: Human-readable label used in tables and reports.
        y_pred: Main prediction array of shape `(n_samples,)`.
        y_score: Optional score or probability array of shape `(n_samples,)`.
        task_spec: Optional shared task specification.
        metadata: Additional metadata to keep alongside the predictions.

    Example:
        >>> task = TaskSpec(task_type="classification")
        >>> bundle = PredictionBundle(name="gbm", y_pred=[0, 1], y_score=[0.2, 0.8], task_spec=task)
        >>> bundle.y_score.shape
        (2,)
    """

    name: str
    y_pred: np.ndarray
    y_score: np.ndarray | None = None
    task_spec: TaskSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.y_pred = np.asarray(self.y_pred)
        if self.y_score is not None:
            self.y_score = np.asarray(self.y_score)

    def with_task_spec(self, task_spec: TaskSpec | None) -> "PredictionBundle":
        """Return a copy enriched with a resolved task specification.

        Args:
            task_spec: Task specification to inject when the bundle does not
                already carry one.

        Returns:
            PredictionBundle: New bundle with the resolved task context.
        """

        resolved = task_spec or self.task_spec
        return PredictionBundle(
            name=self.name,
            y_pred=self.y_pred,
            y_score=self.y_score,
            task_spec=resolved,
            metadata=dict(self.metadata),
        )

    def to_dict(self, *, include_arrays: bool = True) -> dict[str, Any]:
        """Serialize the prediction bundle into a JSON-friendly dictionary.

        Args:
            include_arrays: Whether to inline array values instead of compact
                array metadata.

        Returns:
            dict[str, Any]: Serialized prediction payload.
        """

        return {
            "name": self.name,
            "y_pred": to_serializable(self.y_pred, include_arrays=include_arrays),
            "y_score": to_serializable(self.y_score, include_arrays=include_arrays),
            "task_spec": self.task_spec.to_dict() if self.task_spec else None,
            "metadata": to_serializable(self.metadata, include_arrays=include_arrays),
        }
