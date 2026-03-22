"""Central evaluator comparing one or more prediction bundles."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mlcraft.core.prediction import PredictionBundle, resolve_task_spec
from mlcraft.core.results import EvaluationResult, MetricRow
from mlcraft.evaluation.curves import calibration_curve_data, poisson_calibration_curve, pr_curve_data, residual_distribution_data, roc_curve_data
from mlcraft.metrics.registry import MetricRegistry, default_metric_registry
from mlcraft.utils.logging import get_logger, inject_logger


class Evaluator:
    """Evaluate one or multiple predictions against a single ground truth."""

    def __init__(self, metric_registry: MetricRegistry | None = None, logger=None) -> None:
        self.metric_registry = metric_registry or default_metric_registry
        self.logger = inject_logger(logger, "evaluation")

    def evaluate(
        self,
        y_true,
        predictions,
        *,
        task_spec=None,
        sample_weight=None,
        exposure=None,
        metric_names=None,
        metric_options=None,
    ) -> EvaluationResult:
        y_true_array = np.asarray(y_true)
        bundles = [predictions] if isinstance(predictions, PredictionBundle) else list(predictions)
        if not bundles:
            raise ValueError("At least one prediction bundle is required.")
        resolved_task = resolve_task_spec(task_spec, *bundles)
        if resolved_task is None:
            raise ValueError("A task_spec must be provided directly or via a PredictionBundle.")
        metric_names = list(metric_names or self._default_metric_names(resolved_task.task_type.value))
        metric_options = metric_options or {}

        metric_rows: list[MetricRow] = []
        curves: dict[str, list] = {}
        for bundle in bundles:
            enriched = bundle.with_task_spec(resolved_task)
            y_pred = np.asarray(enriched.y_pred)
            y_score = None if enriched.y_score is None else np.asarray(enriched.y_score)
            for metric_name in metric_names:
                value, score = self.metric_registry.evaluate(
                    metric_name,
                    y_true_array,
                    y_pred=y_pred,
                    y_score=y_score,
                    sample_weight=sample_weight,
                    exposure=exposure,
                    **metric_options.get(metric_name, {}),
                )
                definition = self.metric_registry.get(metric_name)
                metric_rows.append(
                    MetricRow(
                        prediction_name=enriched.name,
                        metric_name=metric_name,
                        value=value,
                        score=score,
                        higher_is_better=definition.higher_is_better,
                    )
                )
            curves[enriched.name] = self._build_curves(
                resolved_task.task_type.value,
                y_true_array,
                y_pred,
                y_score,
                sample_weight=sample_weight,
                exposure=exposure,
            )
        return EvaluationResult(
            task_spec=resolved_task,
            metric_rows=metric_rows,
            curves=curves,
            metadata={"metric_names": metric_names},
        )

    def _default_metric_names(self, task_type: str) -> list[str]:
        if task_type == "classification":
            return ["roc_auc", "pr_auc", "logloss", "accuracy", "precision", "recall", "f1", "brier_score", "gini"]
        if task_type == "poisson":
            return ["poisson_deviance", "mae", "rmse", "observed_mean", "predicted_mean"]
        return ["mae", "mse", "rmse", "r2", "medae"]

    def _build_curves(self, task_type: str, y_true, y_pred, y_score, *, sample_weight=None, exposure=None):
        curves = []
        if task_type == "classification":
            score_values = y_score if y_score is not None else y_pred
            curves.append(roc_curve_data(y_true, score_values, sample_weight=sample_weight))
            curves.append(pr_curve_data(y_true, score_values, sample_weight=sample_weight))
            curves.append(calibration_curve_data(y_true, score_values, sample_weight=sample_weight))
        elif task_type == "poisson":
            curves.append(poisson_calibration_curve(y_true, y_pred, exposure=exposure))
            curves.append(residual_distribution_data(y_true, np.asarray(y_pred) * (1.0 if exposure is None else np.asarray(exposure))))
        else:
            curves.append(residual_distribution_data(y_true, y_pred))
        return curves

