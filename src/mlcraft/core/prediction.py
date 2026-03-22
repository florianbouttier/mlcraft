"""Prediction bundle helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from mlcraft.core.task import TaskSpec
from mlcraft.utils.serialization import to_serializable


@runtime_checkable
class TaskSpecCarrier(Protocol):
    task_spec: TaskSpec | None


def resolve_task_spec(
    task_spec: TaskSpec | None = None,
    *carriers: TaskSpecCarrier | None,
) -> TaskSpec | None:
    """Resolve a task spec from an explicit value or available carriers."""

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
        resolved = task_spec or self.task_spec
        return PredictionBundle(
            name=self.name,
            y_pred=self.y_pred,
            y_score=self.y_score,
            task_spec=resolved,
            metadata=dict(self.metadata),
        )

    def to_dict(self, *, include_arrays: bool = True) -> dict[str, Any]:
        return {
            "name": self.name,
            "y_pred": to_serializable(self.y_pred, include_arrays=include_arrays),
            "y_score": to_serializable(self.y_score, include_arrays=include_arrays),
            "task_spec": self.task_spec.to_dict() if self.task_spec else None,
            "metadata": to_serializable(self.metadata, include_arrays=include_arrays),
        }
