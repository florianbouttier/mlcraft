"""View-model builders for HTML reporting."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from mlcraft.core.results import CurveData, EvaluationResult, MetricRow, ShapResult, TuningResult


def build_curve_groups(curves_by_prediction: dict[str, list[CurveData]]) -> list[dict[str, Any]]:
    """Convert curve payloads into JSON-friendly grouped dictionaries.

    Args:
        curves_by_prediction: Curves keyed by prediction bundle name.

    Returns:
        list[dict[str, Any]]: Curve groups ready for rendering.
    """

    grouped: dict[str, list[tuple[str, CurveData]]] = defaultdict(list)
    for prediction_name, curves in curves_by_prediction.items():
        for curve in curves:
            grouped[curve.name].append((prediction_name, curve))

    curve_groups: list[dict[str, Any]] = []
    for curve_name, entries in grouped.items():
        reference_curve = entries[0][1]
        curve_groups.append(
            {
                "curve_name": curve_name,
                "title": curve_name.replace("_", " ").title(),
                "x_label": reference_curve.x_label,
                "y_label": reference_curve.y_label,
                "series": [
                    {
                        "prediction_name": prediction_name,
                        "x": np.asarray(curve.x, dtype=float).tolist(),
                        "y": np.asarray(curve.y, dtype=float).tolist(),
                        "metadata": dict(curve.metadata),
                    }
                    for prediction_name, curve in entries
                ],
            }
        )
    return curve_groups


def build_fold_curve_groups(fold_summaries) -> list[dict[str, Any]]:
    """Convert fold-level train/validation evaluations into grouped curves."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fold in fold_summaries:
        for split_name, evaluation in (("train", fold.train_evaluation), ("validation", fold.val_evaluation)):
            if evaluation is None:
                continue
            for prediction_name, curves in evaluation.curves.items():
                for curve in curves:
                    grouped[curve.name].append(
                        {
                            "series_name": f"Fold {int(fold.fold_index)} {split_name.title()}",
                            "prediction_name": prediction_name,
                            "fold_index": int(fold.fold_index),
                            "split": split_name,
                            "x": np.asarray(curve.x, dtype=float).tolist(),
                            "y": np.asarray(curve.y, dtype=float).tolist(),
                            "metadata": dict(curve.metadata),
                        }
                    )

    curve_groups: list[dict[str, Any]] = []
    for curve_name, series in grouped.items():
        reference = series[0]
        curve_groups.append(
            {
                "curve_name": curve_name,
                "title": f"{curve_name.replace('_', ' ').title()} by fold",
                "x_label": next(
                    (
                        curve.x_label
                        for fold in fold_summaries
                        for evaluation in (fold.train_evaluation, fold.val_evaluation)
                        if evaluation is not None
                        for curves in evaluation.curves.values()
                        for curve in curves
                        if curve.name == curve_name
                    ),
                    "",
                ),
                "y_label": next(
                    (
                        curve.y_label
                        for fold in fold_summaries
                        for evaluation in (fold.train_evaluation, fold.val_evaluation)
                        if evaluation is not None
                        for curves in evaluation.curves.values()
                        for curve in curves
                        if curve.name == curve_name
                    ),
                    "",
                ),
                "series": series,
            }
        )
    return curve_groups


def build_fold_metric_rows(fold_summaries) -> list[dict[str, Any]]:
    """Flatten fold-level train/validation metric evaluations."""

    rows: list[dict[str, Any]] = []
    for fold in fold_summaries:
        for split_name, evaluation, metrics in (
            ("train", fold.train_evaluation, fold.train_metrics),
            ("validation", fold.val_evaluation, fold.val_metrics),
        ):
            if evaluation is not None:
                for metric_row in evaluation.metric_rows:
                    rows.append(
                        {
                            "fold_index": int(fold.fold_index),
                            "split": split_name,
                            "metric_name": metric_row.metric_name,
                            "value": float(metric_row.value),
                            "score": float(metric_row.score),
                            "higher_is_better": bool(metric_row.higher_is_better),
                        }
                    )
                continue
            for metric_name, metric_value in metrics.items():
                rows.append(
                    {
                        "fold_index": int(fold.fold_index),
                        "split": split_name,
                        "metric_name": str(metric_name),
                        "value": float(metric_value),
                        "score": float(metric_value),
                        "higher_is_better": True,
                    }
                )
    return rows


def build_evaluation_context(result: EvaluationResult, *, title: str | None = "mlcraft Evaluation Report") -> dict[str, Any]:
    """Build the render context for an evaluation report.

    Args:
        result: Evaluation result to expose as a render context.
        title: Optional report title.

    Returns:
        dict[str, Any]: JSON-friendly evaluation view model.
    """

    metric_rows = [
        {
            "prediction_name": row.prediction_name,
            "metric_name": row.metric_name,
            "value": float(row.value),
            "score": float(row.score),
            "higher_is_better": bool(row.higher_is_better),
        }
        for row in result.metric_rows
    ]
    metrics_by_prediction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        metrics_by_prediction[row["prediction_name"]].append(row)

    primary_metric = result.task_spec.eval_metric
    available_metrics = [row["metric_name"] for row in metric_rows]
    if primary_metric not in available_metrics and available_metrics:
        primary_metric = available_metrics[0]

    primary_rows = sorted(
        (row for row in metric_rows if row["metric_name"] == primary_metric),
        key=lambda row: row["score"],
        reverse=True,
    )
    leader = primary_rows[0] if primary_rows else None
    lead_margin = primary_rows[0]["score"] - primary_rows[1]["score"] if len(primary_rows) > 1 else 0.0

    return {
        "title": title,
        "task_type": result.task_spec.task_type.value,
        "primary_metric_name": primary_metric,
        "metric_rows": metric_rows,
        "metrics_by_prediction": dict(metrics_by_prediction),
        "summary": {
            "model_count": len(metrics_by_prediction),
            "leader": leader,
            "lead_margin": float(lead_margin),
        },
        "curve_groups": build_curve_groups(result.curves),
    }


def build_tuning_context(result: TuningResult, *, title: str | None = "mlcraft Tuning Report") -> dict[str, Any]:
    """Build the render context for a tuning report.

    Args:
        result: Tuning result to expose as a render context.
        title: Optional report title.

    Returns:
        dict[str, Any]: Tuning view model consumed by the HTML renderer.
    """

    metric_name = result.metric_name or next(iter(result.train_metrics.keys()), "metric")
    split_points = [
        {"label": "train", "value": float(result.train_metrics.get(metric_name, next(iter(result.train_metrics.values()), np.nan)))},
        {"label": "validation", "value": float(result.val_metrics.get(metric_name, next(iter(result.val_metrics.values()), np.nan)))},
    ]
    if result.test_metrics:
        split_points.append(
            {
                "label": "final_test",
                "value": float(result.test_metrics.get(metric_name, next(iter(result.test_metrics.values()), np.nan))),
            }
        )

    fold_points = [
        {
            "fold_index": int(fold.fold_index),
            "train_score": float(fold.train_score),
            "val_score": float(fold.val_score),
            "penalized_score": float(fold.penalized_score),
            "gap": float(fold.train_score - fold.val_score),
            "train_metric_value": float(fold.train_metrics.get(metric_name, next(iter(fold.train_metrics.values()), np.nan))),
            "val_metric_value": float(fold.val_metrics.get(metric_name, next(iter(fold.val_metrics.values()), np.nan))),
        }
        for fold in result.fold_summaries
    ]
    best_fold = max(fold_points, key=lambda fold: fold["val_score"]) if fold_points else None
    worst_fold = min(fold_points, key=lambda fold: fold["val_score"]) if fold_points else None
    fold_metric_rows = build_fold_metric_rows(result.fold_summaries)
    fold_curve_groups = build_fold_curve_groups(result.fold_summaries)
    selected_model_type = result.metadata.get("selected_model_type", result.metadata.get("model_type"))
    backend_results = result.metadata.get("backend_results")
    if not backend_results:
        backend_name = selected_model_type or "model"
        backend_results = {backend_name: result.to_dict(include_arrays=True)}
    backend_summary_rows = []
    for backend_name, payload in backend_results.items():
        payload_train = payload.get("train_metrics") or {}
        payload_val = payload.get("val_metrics") or {}
        payload_test = payload.get("test_metrics") or {}
        backend_summary_rows.append(
            {
                "backend_name": str(backend_name),
                "best_score": float(payload.get("best_score", np.nan)),
                "train_metric_value": float(payload_train.get(metric_name, next(iter(payload_train.values()), np.nan))),
                "val_metric_value": float(payload_val.get(metric_name, next(iter(payload_val.values()), np.nan))),
                "test_metric_value": None if not payload_test else float(payload_test.get(metric_name, next(iter(payload_test.values()), np.nan))),
                "is_selected": str(backend_name) == str(selected_model_type),
            }
        )
    backend_summary_rows.sort(key=lambda row: row["best_score"], reverse=True)

    return {
        "title": title,
        "task_type": result.task_spec.task_type.value,
        "metric_name": metric_name,
        "alpha": float(result.alpha),
        "best_score": float(result.best_score),
        "train_score": float(result.best_trial.train_score),
        "val_score": float(result.best_trial.val_score),
        "generalization_gap": float(abs(result.best_trial.train_score - result.best_trial.val_score)),
        "best_params": {str(key): value for key, value in result.best_params.items()},
        "split_points": split_points,
        "fold_points": fold_points,
        "fold_metric_rows": fold_metric_rows,
        "fold_curve_groups": fold_curve_groups,
        "best_fold": best_fold,
        "worst_fold": worst_fold,
        "test_metric_value": None if not result.test_metrics else float(result.test_metrics.get(metric_name, next(iter(result.test_metrics.values()), np.nan))),
        "test_score": None if result.test_score is None else float(result.test_score),
        "holdout_curve_groups": [] if result.test_evaluation is None else build_curve_groups(result.test_evaluation.curves),
        "selected_model_type": selected_model_type,
        "backend_comparison": result.metadata.get("backend_comparison", {}),
        "backend_summary_rows": backend_summary_rows,
        "optuna_plots": ["optimization_history", "parallel_coordinate", "slice"],
        "study": result.study,
    }


def build_shap_context(
    result: ShapResult,
    *,
    title: str | None = "mlcraft SHAP Report",
    top_n: int = 10,
) -> dict[str, Any]:
    """Build the render context for a SHAP report.

    Args:
        result: SHAP result to expose as a render context.
        title: Optional report title.
        top_n: Number of top features to keep in summary plots.

    Returns:
        dict[str, Any]: SHAP view model consumed by the HTML renderer.
    """

    shap_values = np.asarray(result.shap_values)
    feature_values = None if result.feature_values is None else np.array(result.feature_values, copy=True)
    base_values = None if result.base_values is None else np.asarray(result.base_values)
    interaction_values = None if result.interaction_values is None else np.asarray(result.interaction_values)
    importance = result.importance if result.importance is not None else np.mean(np.abs(shap_values), axis=0)
    full_order = np.argsort(importance)[::-1]
    order = full_order[:top_n]
    top_feature_names = np.asarray(result.feature_names)[order].tolist()
    interactions = None
    if interaction_values is not None:
        interaction_matrix = np.mean(np.abs(interaction_values), axis=0)
        interactions = interaction_matrix[np.ix_(order, order)].tolist()

    scatters = []
    if feature_values is not None:
        for feature_index in order[: min(top_n, 5)]:
            scatters.append(
                {
                    "feature_name": result.feature_names[int(feature_index)],
                    "x": np.asarray(feature_values[:, feature_index]).tolist(),
                    "y": np.asarray(shap_values[:, feature_index], dtype=float).tolist(),
                }
            )

    return {
        "title": title,
        "top_n": int(top_n),
        "feature_names": list(result.feature_names),
        "ordered_feature_names": np.asarray(result.feature_names)[full_order].tolist(),
        "ordered_feature_indices": np.asarray(full_order, dtype=int).tolist(),
        "top_feature_names": top_feature_names,
        "importance_values": np.asarray(importance[order], dtype=float).tolist(),
        "shap_values": shap_values,
        "feature_values": feature_values,
        "base_values": base_values,
        "interaction_values": interaction_values,
        "beeswarm": [
            {
                "feature_name": result.feature_names[int(feature_index)],
                "values": np.asarray(shap_values[:, feature_index], dtype=float).tolist(),
            }
            for feature_index in order
        ],
        "interaction_matrix": interactions,
        "scatter_plots": scatters,
        "feature_count": int(len(result.feature_names)),
    }
