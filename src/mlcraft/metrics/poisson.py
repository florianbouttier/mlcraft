"""Poisson metrics and diagnostics."""

from __future__ import annotations

import numpy as np


def _resolve_mu(y_pred, exposure=None) -> np.ndarray:
    pred = np.asarray(y_pred, dtype=float)
    if exposure is None:
        return np.clip(pred, 1e-12, None)
    exp = np.asarray(exposure, dtype=float)
    return np.clip(pred * exp, 1e-12, None)


def _observed_rate(y_true, exposure=None) -> np.ndarray:
    y = np.asarray(y_true, dtype=float)
    if exposure is None:
        return y
    exp = np.asarray(exposure, dtype=float)
    return y / np.clip(exp, 1e-12, None)


def _mean(values: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    if sample_weight is None:
        return float(np.mean(values))
    return float(np.average(values, weights=sample_weight))


def poisson_deviance(y_true, y_pred, *, sample_weight=None, exposure=None, **_) -> float:
    """Compute mean Poisson deviance.

    Args:
        y_true: Observed counts of shape `(n_samples,)`.
        y_pred: Predicted rates of shape `(n_samples,)`.
        sample_weight: Optional per-row weights.
        exposure: Optional exposure vector of shape `(n_samples,)`.

    Returns:
        float: Weighted or unweighted Poisson deviance.
    """

    y = np.asarray(y_true, dtype=float)
    mu = _resolve_mu(y_pred, exposure=exposure)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_term = np.where(y > 0, y * np.log(y / mu), 0.0)
    deviance = 2.0 * (ratio_term - (y - mu))
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    return _mean(deviance, weights)


def mae(y_true, y_pred, *, sample_weight=None, exposure=None, **_) -> float:
    """Compute mean absolute error for Poisson counts.

    Args:
        y_true: Observed counts of shape `(n_samples,)`.
        y_pred: Predicted rates of shape `(n_samples,)`.
        sample_weight: Optional per-row weights.
        exposure: Optional exposure vector of shape `(n_samples,)`.

    Returns:
        float: Weighted or unweighted absolute error on counts.
    """

    y = np.asarray(y_true, dtype=float)
    mu = _resolve_mu(y_pred, exposure=exposure)
    errors = np.abs(y - mu)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    return _mean(errors, weights)


def rmse(y_true, y_pred, *, sample_weight=None, exposure=None, **_) -> float:
    """Compute root mean squared error for Poisson counts.

    Args:
        y_true: Observed counts of shape `(n_samples,)`.
        y_pred: Predicted rates of shape `(n_samples,)`.
        sample_weight: Optional per-row weights.
        exposure: Optional exposure vector of shape `(n_samples,)`.

    Returns:
        float: Weighted or unweighted RMSE on counts.
    """

    y = np.asarray(y_true, dtype=float)
    mu = _resolve_mu(y_pred, exposure=exposure)
    errors = (y - mu) ** 2
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    return float(np.sqrt(_mean(errors, weights)))


def observed_mean(y_true, y_pred=None, *, sample_weight=None, exposure=None, **_) -> float:
    """Compute the observed mean count or rate.

    Args:
        y_true: Observed counts of shape `(n_samples,)`.
        y_pred: Unused placeholder kept for registry compatibility.
        sample_weight: Optional per-row weights.
        exposure: Optional exposure vector of shape `(n_samples,)`.

    Returns:
        float: Weighted or unweighted observed mean.
    """

    observed = _observed_rate(y_true, exposure=exposure)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    return _mean(observed, weights)


def predicted_mean(y_true, y_pred, *, sample_weight=None, exposure=None, **_) -> float:
    """Compute the mean predicted rate.

    Args:
        y_true: Unused placeholder kept for registry compatibility.
        y_pred: Predicted rates of shape `(n_samples,)`.
        sample_weight: Optional per-row weights.
        exposure: Unused placeholder kept for signature consistency.

    Returns:
        float: Weighted or unweighted mean predicted rate.
    """

    pred = np.asarray(y_pred, dtype=float)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    return _mean(pred, weights)


def poisson_calibration_diagnostics(y_true, y_pred, *, exposure=None, n_bins: int = 10, sample_weight=None) -> dict[str, np.ndarray]:
    """Aggregate Poisson calibration diagnostics by prediction bins.

    Args:
        y_true: Observed counts of shape `(n_samples,)`.
        y_pred: Predicted rates of shape `(n_samples,)`.
        exposure: Optional exposure vector of shape `(n_samples,)`.
        n_bins: Number of quantile-based bins.
        sample_weight: Optional per-row weights.

    Returns:
        dict[str, np.ndarray]: Binned predicted means, observed means, and
        sample counts.
    """

    pred = np.asarray(y_pred, dtype=float)
    observed = _observed_rate(y_true, exposure=exposure)
    if pred.size == 0:
        return {"bin_centers": np.array([]), "predicted": np.array([]), "observed": np.array([]), "counts": np.array([])}
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(pred, quantiles))
    if edges.shape[0] == 1:
        edges = np.array([pred.min(), pred.max() + 1e-12])
    bins = np.digitize(pred, edges[1:-1], right=True)
    predicted_bins = []
    observed_bins = []
    counts = []
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    for bin_index in range(len(edges) - 1):
        mask = bins == bin_index
        if not np.any(mask):
            continue
        w = None if weights is None else weights[mask]
        predicted_bins.append(_mean(pred[mask], w))
        observed_bins.append(_mean(observed[mask], w))
        counts.append(int(np.sum(mask)))
    return {
        "bin_centers": np.asarray(predicted_bins, dtype=float),
        "predicted": np.asarray(predicted_bins, dtype=float),
        "observed": np.asarray(observed_bins, dtype=float),
        "counts": np.asarray(counts, dtype=int),
    }
