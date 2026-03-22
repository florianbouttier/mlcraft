"""Regression metrics implemented with numpy."""

from __future__ import annotations

import numpy as np


def _as_arrays(y_true, y_pred, sample_weight=None):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    return yt, yp, sw


def _weighted_mean(values: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    if sample_weight is None:
        return float(np.mean(values))
    return float(np.average(values, weights=sample_weight))


def _weighted_median(values: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    if sample_weight is None:
        return float(np.median(values))
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = sample_weight[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = 0.5 * np.sum(sorted_weights)
    return float(sorted_values[np.searchsorted(cumulative, cutoff, side="left")])


def mae(y_true, y_pred, *, sample_weight=None, **_) -> float:
    """Compute mean absolute error.

    Args:
        y_true: Ground-truth array of shape `(n_samples,)`.
        y_pred: Prediction array of shape `(n_samples,)`.
        sample_weight: Optional per-row weights.

    Returns:
        float: Weighted or unweighted MAE.
    """

    yt, yp, sw = _as_arrays(y_true, y_pred, sample_weight)
    return _weighted_mean(np.abs(yt - yp), sw)


def mse(y_true, y_pred, *, sample_weight=None, **_) -> float:
    """Compute mean squared error.

    Args:
        y_true: Ground-truth array of shape `(n_samples,)`.
        y_pred: Prediction array of shape `(n_samples,)`.
        sample_weight: Optional per-row weights.

    Returns:
        float: Weighted or unweighted MSE.
    """

    yt, yp, sw = _as_arrays(y_true, y_pred, sample_weight)
    return _weighted_mean((yt - yp) ** 2, sw)


def rmse(y_true, y_pred, *, sample_weight=None, **_) -> float:
    """Compute root mean squared error.

    Args:
        y_true: Ground-truth array of shape `(n_samples,)`.
        y_pred: Prediction array of shape `(n_samples,)`.
        sample_weight: Optional per-row weights.

    Returns:
        float: Weighted or unweighted RMSE.
    """

    return float(np.sqrt(mse(y_true, y_pred, sample_weight=sample_weight)))


def r2(y_true, y_pred, *, sample_weight=None, **_) -> float:
    """Compute the coefficient of determination.

    Args:
        y_true: Ground-truth array of shape `(n_samples,)`.
        y_pred: Prediction array of shape `(n_samples,)`.
        sample_weight: Optional per-row weights.

    Returns:
        float: Weighted or unweighted `R^2` score.
    """

    yt, yp, sw = _as_arrays(y_true, y_pred, sample_weight)
    mean_true = _weighted_mean(yt, sw)
    numerator = np.sum(((yt - yp) ** 2) if sw is None else sw * ((yt - yp) ** 2))
    denominator = np.sum(((yt - mean_true) ** 2) if sw is None else sw * ((yt - mean_true) ** 2))
    if denominator == 0:
        return 0.0
    return float(1.0 - numerator / denominator)


def medae(y_true, y_pred, *, sample_weight=None, **_) -> float:
    """Compute median absolute error.

    Args:
        y_true: Ground-truth array of shape `(n_samples,)`.
        y_pred: Prediction array of shape `(n_samples,)`.
        sample_weight: Optional per-row weights.

    Returns:
        float: Weighted or unweighted median absolute error.
    """

    yt, yp, sw = _as_arrays(y_true, y_pred, sample_weight)
    return _weighted_median(np.abs(yt - yp), sw)


def mape(y_true, y_pred, *, sample_weight=None, epsilon: float = 1e-12, **_) -> float:
    """Compute mean absolute percentage error.

    Args:
        y_true: Ground-truth array of shape `(n_samples,)`.
        y_pred: Prediction array of shape `(n_samples,)`.
        sample_weight: Optional per-row weights.
        epsilon: Minimum absolute target value used to avoid division by zero.

    Returns:
        float: Weighted or unweighted MAPE on rows with non-zero targets.
    """

    yt, yp, sw = _as_arrays(y_true, y_pred, sample_weight)
    mask = np.abs(yt) > epsilon
    if not np.any(mask):
        return 0.0
    return _weighted_mean(np.abs((yt[mask] - yp[mask]) / yt[mask]), None if sw is None else sw[mask])
