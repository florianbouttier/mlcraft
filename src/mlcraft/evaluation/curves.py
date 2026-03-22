"""Curve calculations for evaluation reports."""

from __future__ import annotations

import numpy as np

from mlcraft.core.results import CurveData
from mlcraft.metrics.classification import _pr_curve_arrays, _roc_curve_arrays
from mlcraft.metrics.poisson import poisson_calibration_diagnostics


def roc_curve_data(y_true, y_score, *, sample_weight=None, name: str = "roc") -> CurveData:
    """Build ROC curve data from binary targets and scores.

    Args:
        y_true: Binary ground-truth array of shape `(n_samples,)`.
        y_score: Score or probability array of shape `(n_samples,)`.
        sample_weight: Optional per-row weights.
        name: Label stored in the resulting curve object.

    Returns:
        CurveData: ROC curve ready for rendering.
    """

    fpr, tpr = _roc_curve_arrays(y_true, y_score=y_score, sample_weight=sample_weight)
    return CurveData(name=name, x=fpr, y=tpr, x_label="False positive rate", y_label="True positive rate")


def pr_curve_data(y_true, y_score, *, sample_weight=None, name: str = "pr") -> CurveData:
    """Build precision-recall curve data from binary targets and scores.

    Args:
        y_true: Binary ground-truth array of shape `(n_samples,)`.
        y_score: Score or probability array of shape `(n_samples,)`.
        sample_weight: Optional per-row weights.
        name: Label stored in the resulting curve object.

    Returns:
        CurveData: Precision-recall curve ready for rendering.
    """

    recall, precision = _pr_curve_arrays(y_true, y_score=y_score, sample_weight=sample_weight)
    return CurveData(name=name, x=recall, y=precision, x_label="Recall", y_label="Precision")


def calibration_curve_data(y_true, y_score, *, sample_weight=None, n_bins: int = 10, name: str = "calibration") -> CurveData:
    """Build a calibration curve for binary classification scores.

    Args:
        y_true: Binary ground-truth array of shape `(n_samples,)`.
        y_score: Probability array of shape `(n_samples,)`.
        sample_weight: Optional per-row weights.
        n_bins: Number of quantile-based bins.
        name: Label stored in the resulting curve object.

    Returns:
        CurveData: Calibration curve ready for rendering.
    """

    y_true_arr = np.asarray(y_true, dtype=float)
    y_score_arr = np.asarray(y_score, dtype=float)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(y_score_arr, quantiles))
    if edges.shape[0] == 1:
        edges = np.array([y_score_arr.min(), y_score_arr.max() + 1e-12])
    bins = np.digitize(y_score_arr, edges[1:-1], right=True)
    pred_bins = []
    obs_bins = []
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    for bin_index in range(len(edges) - 1):
        mask = bins == bin_index
        if not np.any(mask):
            continue
        if weights is None:
            pred_bins.append(float(np.mean(y_score_arr[mask])))
            obs_bins.append(float(np.mean(y_true_arr[mask])))
        else:
            pred_bins.append(float(np.average(y_score_arr[mask], weights=weights[mask])))
            obs_bins.append(float(np.average(y_true_arr[mask], weights=weights[mask])))
    return CurveData(
        name=name,
        x=np.asarray(pred_bins, dtype=float),
        y=np.asarray(obs_bins, dtype=float),
        x_label="Predicted probability",
        y_label="Observed probability",
    )


def residual_distribution_data(y_true, y_pred, *, name: str = "residuals", bins: int = 20) -> CurveData:
    """Build a residual histogram for regression-like tasks.

    Args:
        y_true: Ground-truth array of shape `(n_samples,)`.
        y_pred: Prediction array of shape `(n_samples,)`.
        name: Label stored in the resulting curve object.
        bins: Number of histogram bins.

    Returns:
        CurveData: Residual histogram ready for rendering.
    """

    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    counts, edges = np.histogram(residuals, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return CurveData(name=name, x=centers, y=counts.astype(float), x_label="Residual", y_label="Count")


def poisson_calibration_curve(y_true, y_pred, *, exposure=None, name: str = "poisson_calibration") -> CurveData:
    """Build a calibration curve for Poisson rate predictions.

    Args:
        y_true: Observed counts of shape `(n_samples,)`.
        y_pred: Predicted rates of shape `(n_samples,)`.
        exposure: Optional exposure vector of shape `(n_samples,)`.
        name: Label stored in the resulting curve object.

    Returns:
        CurveData: Poisson calibration curve ready for rendering.
    """

    diagnostics = poisson_calibration_diagnostics(y_true, y_pred, exposure=exposure)
    return CurveData(
        name=name,
        x=diagnostics["predicted"],
        y=diagnostics["observed"],
        x_label="Predicted rate",
        y_label="Observed rate",
        metadata={"counts": diagnostics["counts"].tolist()},
    )
