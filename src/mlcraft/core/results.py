"""Structured result containers used across the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mlcraft.core.task import TaskSpec
from mlcraft.utils.serialization import to_serializable


@dataclass
class MetricRow:
    prediction_name: str
    metric_name: str
    value: float
    score: float
    higher_is_better: bool


@dataclass
class CurveData:
    name: str
    x: np.ndarray
    y: np.ndarray
    x_label: str
    y_label: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    task_spec: TaskSpec
    metric_rows: list[MetricRow]
    curves: dict[str, list[CurveData]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def metrics_by_prediction(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for row in self.metric_rows:
            result.setdefault(row.prediction_name, {})[row.metric_name] = row.value
        return result

    def to_dict(self, *, include_arrays: bool = True) -> dict[str, Any]:
        return {
            "task_spec": self.task_spec.to_dict(),
            "metric_rows": to_serializable(self.metric_rows, include_arrays=include_arrays),
            "curves": to_serializable(self.curves, include_arrays=include_arrays),
            "metadata": to_serializable(self.metadata, include_arrays=include_arrays),
        }


@dataclass
class FoldSummary:
    fold_index: int
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    train_score: float
    val_score: float
    penalized_score: float


@dataclass
class TrialSummary:
    trial_number: int
    params: dict[str, Any]
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    train_score: float
    val_score: float
    penalized_score: float
    folds: list[FoldSummary] = field(default_factory=list)


@dataclass
class TuningResult:
    task_spec: TaskSpec
    best_params: dict[str, Any]
    best_score: float
    best_trial: TrialSummary
    history: list[TrialSummary]
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    penalized_score: float
    fold_summaries: list[FoldSummary]
    alpha: float = 0.0
    metric_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    study: Any = field(default=None, repr=False, compare=False)

    def to_dict(self, *, include_arrays: bool = True) -> dict[str, Any]:
        payload = {
            "task_spec": self.task_spec.to_dict(),
            "best_params": to_serializable(self.best_params, include_arrays=include_arrays),
            "best_score": self.best_score,
            "best_trial": to_serializable(self.best_trial, include_arrays=include_arrays),
            "history": to_serializable(self.history, include_arrays=include_arrays),
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "penalized_score": self.penalized_score,
            "fold_summaries": to_serializable(self.fold_summaries, include_arrays=include_arrays),
            "alpha": self.alpha,
            "metric_name": self.metric_name,
            "metadata": to_serializable(self.metadata, include_arrays=include_arrays),
        }
        return payload


@dataclass
class ShapResult:
    feature_names: list[str]
    shap_values: np.ndarray
    feature_values: np.ndarray | None = None
    base_values: np.ndarray | float | None = None
    interaction_values: np.ndarray | None = None
    importance: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "shap_values": to_serializable(self.shap_values, include_arrays=include_arrays),
            "feature_values": to_serializable(self.feature_values, include_arrays=include_arrays),
            "base_values": to_serializable(self.base_values, include_arrays=include_arrays),
            "interaction_values": to_serializable(self.interaction_values, include_arrays=include_arrays),
            "importance": to_serializable(self.importance, include_arrays=include_arrays),
            "metadata": to_serializable(self.metadata, include_arrays=include_arrays),
        }
