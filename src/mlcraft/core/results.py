"""Structured result containers used across the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mlcraft.core.task import TaskSpec
from mlcraft.utils.serialization import to_serializable


@dataclass
class MetricRow:
    """Store one metric value for one prediction bundle."""

    prediction_name: str
    metric_name: str
    value: float
    score: float
    higher_is_better: bool


@dataclass
class CurveData:
    """Store one curve ready for plotting or HTML rendering."""

    name: str
    x: np.ndarray
    y: np.ndarray
    x_label: str
    y_label: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Store evaluation metrics and plots for one evaluation run.

    Args:
        task_spec: Shared task specification used for the evaluation.
        metric_rows: Flat metric table across all evaluated predictions.
        curves: Plot-ready curves keyed by prediction bundle name.
        metadata: Additional evaluation metadata.
    """

    task_spec: TaskSpec
    metric_rows: list[MetricRow]
    curves: dict[str, list[CurveData]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def metrics_by_prediction(self) -> dict[str, dict[str, float]]:
        """Group metric values by prediction name.

        Returns:
            dict[str, dict[str, float]]: Nested mapping
            `{prediction_name: {metric_name: value}}`.
        """

        result: dict[str, dict[str, float]] = {}
        for row in self.metric_rows:
            result.setdefault(row.prediction_name, {})[row.metric_name] = row.value
        return result

    def to_dict(self, *, include_arrays: bool = True) -> dict[str, Any]:
        """Serialize the evaluation result into a JSON-friendly dictionary.

        Args:
            include_arrays: Whether to inline curve arrays instead of compact
                metadata.

        Returns:
            dict[str, Any]: Serialized evaluation payload.
        """

        return {
            "task_spec": self.task_spec.to_dict(),
            "metric_rows": to_serializable(self.metric_rows, include_arrays=include_arrays),
            "curves": to_serializable(self.curves, include_arrays=include_arrays),
            "metadata": to_serializable(self.metadata, include_arrays=include_arrays),
        }


@dataclass
class FoldSummary:
    """Store aggregated train and validation metrics for one CV fold."""

    fold_index: int
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    train_score: float
    val_score: float
    penalized_score: float


@dataclass
class TrialSummary:
    """Store the aggregated outcome of one tuning trial."""

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
    """Store the outcome of an Optuna search run.

    Args:
        task_spec: Shared task specification optimized during tuning.
        best_params: Best hyperparameters found by the study.
        best_score: Best penalized score returned to Optuna.
        best_trial: Structured summary for the winning trial.
        history: Structured summaries for all executed trials.
        train_metrics: Aggregated train metrics for the winning trial.
        val_metrics: Aggregated validation metrics for the winning trial.
        penalized_score: Validation score penalized by overfitting.
        fold_summaries: Fold-level aggregates for the winning trial.
        alpha: Penalty factor used in the objective.
        metric_name: Canonical metric optimized by the search.
        test_metrics: Optional holdout metrics computed with the final model.
        test_score: Optional normalized score of the optimized metric on the
            final holdout set.
        test_evaluation: Optional structured evaluation of the final holdout
            set.
        metadata: Additional metadata about the search run.
        study: Optional in-memory Optuna study.
    """

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
    test_metrics: dict[str, float] | None = None
    test_score: float | None = None
    test_evaluation: EvaluationResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    study: Any = field(default=None, repr=False, compare=False)

    def to_dict(self, *, include_arrays: bool = True) -> dict[str, Any]:
        """Serialize the tuning result into a JSON-friendly dictionary.

        Args:
            include_arrays: Whether to inline arrays instead of compact array
                metadata.

        Returns:
            dict[str, Any]: Serialized tuning payload.
        """

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
            "test_metrics": to_serializable(self.test_metrics, include_arrays=include_arrays),
            "test_score": self.test_score,
            "test_evaluation": to_serializable(self.test_evaluation, include_arrays=include_arrays),
            "metadata": to_serializable(self.metadata, include_arrays=include_arrays),
        }
        return payload


@dataclass
class ShapResult:
    """Store SHAP values and related artifacts for one explainability run.

    Args:
        feature_names: Feature names aligned with the SHAP matrices.
        shap_values: SHAP matrix of shape `(n_samples, n_features)`.
        feature_values: Optional feature matrix aligned with `shap_values`.
        base_values: Optional SHAP base value or vector of base values.
        interaction_values: Optional interaction tensor.
        importance: Optional mean absolute SHAP importance vector.
        metadata: Additional explainability metadata.
    """

    feature_names: list[str]
    shap_values: np.ndarray
    feature_values: np.ndarray | None = None
    base_values: np.ndarray | float | None = None
    interaction_values: np.ndarray | None = None
    importance: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        """Serialize the SHAP result into a JSON-friendly dictionary.

        Args:
            include_arrays: Whether to inline raw arrays instead of compact
                array metadata. The default stays compact because SHAP outputs
                can be large.

        Returns:
            dict[str, Any]: Serialized explainability payload.
        """

        return {
            "feature_names": list(self.feature_names),
            "shap_values": to_serializable(self.shap_values, include_arrays=include_arrays),
            "feature_values": to_serializable(self.feature_values, include_arrays=include_arrays),
            "base_values": to_serializable(self.base_values, include_arrays=include_arrays),
            "interaction_values": to_serializable(self.interaction_values, include_arrays=include_arrays),
            "importance": to_serializable(self.importance, include_arrays=include_arrays),
            "metadata": to_serializable(self.metadata, include_arrays=include_arrays),
        }
