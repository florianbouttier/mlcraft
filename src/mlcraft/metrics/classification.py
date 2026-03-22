"""Binary classification metrics implemented with numpy."""

from __future__ import annotations

import numpy as np


def _as_binary_arrays(y_true, y_pred=None, y_score=None, sample_weight=None):
    yt = np.asarray(y_true, dtype=int)
    yp = None if y_pred is None else np.asarray(y_pred, dtype=int)
    ys = None if y_score is None else np.asarray(y_score, dtype=float)
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    return yt, yp, ys, sw


def _mean(values: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    if sample_weight is None:
        return float(np.mean(values))
    return float(np.average(values, weights=sample_weight))


def _label_predictions(y_score: np.ndarray | None, y_pred: np.ndarray | None, threshold: float = 0.5) -> np.ndarray:
    if y_pred is not None:
        return np.asarray(y_pred, dtype=int)
    if y_score is None:
        raise ValueError("Either y_pred or y_score must be provided.")
    return (np.asarray(y_score, dtype=float) >= threshold).astype(int)


def _roc_curve_arrays(y_true, y_score, sample_weight=None) -> tuple[np.ndarray, np.ndarray]:
    yt, _, ys, sw = _as_binary_arrays(y_true, y_score=y_score, sample_weight=sample_weight)
    order = np.argsort(-ys, kind="mergesort")
    y_sorted = yt[order]
    score_sorted = ys[order]
    weight_sorted = np.ones_like(y_sorted, dtype=float) if sw is None else sw[order]
    tp = np.cumsum(weight_sorted * (y_sorted == 1))
    fp = np.cumsum(weight_sorted * (y_sorted == 0))
    tp_total = tp[-1] if tp.size else 0.0
    fp_total = fp[-1] if fp.size else 0.0
    if tp_total == 0 or fp_total == 0:
        base = np.array([0.0, 1.0], dtype=float)
        return base, base
    distinct = np.r_[True, np.diff(score_sorted) != 0]
    tpr = np.r_[0.0, tp[distinct] / tp_total, 1.0]
    fpr = np.r_[0.0, fp[distinct] / fp_total, 1.0]
    return fpr, tpr


def _pr_curve_arrays(y_true, y_score, sample_weight=None) -> tuple[np.ndarray, np.ndarray]:
    yt, _, ys, sw = _as_binary_arrays(y_true, y_score=y_score, sample_weight=sample_weight)
    order = np.argsort(-ys, kind="mergesort")
    y_sorted = yt[order]
    score_sorted = ys[order]
    weight_sorted = np.ones_like(y_sorted, dtype=float) if sw is None else sw[order]
    tp = np.cumsum(weight_sorted * (y_sorted == 1))
    fp = np.cumsum(weight_sorted * (y_sorted == 0))
    total_positive = tp[-1] if tp.size else 0.0
    if total_positive == 0:
        recall = np.array([0.0, 1.0], dtype=float)
        precision = np.array([1.0, 0.0], dtype=float)
        return recall, precision
    distinct = np.r_[True, np.diff(score_sorted) != 0]
    tp_d = tp[distinct]
    fp_d = fp[distinct]
    precision = np.r_[1.0, tp_d / np.maximum(tp_d + fp_d, 1e-12)]
    recall = np.r_[0.0, tp_d / total_positive]
    return recall, precision


def roc_auc(y_true, y_pred=None, *, y_score=None, sample_weight=None, **_) -> float:
    """Compute ROC AUC for binary classification.

    Args:
        y_true: Binary ground-truth array of shape `(n_samples,)`.
        y_pred: Optional score array used when `y_score` is omitted.
        y_score: Optional score or probability array of shape `(n_samples,)`.
        sample_weight: Optional per-row weights.

    Returns:
        float: ROC AUC score.
    """

    fpr, tpr = _roc_curve_arrays(y_true, y_score=y_score if y_score is not None else y_pred, sample_weight=sample_weight)
    return float(np.trapezoid(tpr, fpr))


def pr_auc(y_true, y_pred=None, *, y_score=None, sample_weight=None, **_) -> float:
    """Compute PR AUC for binary classification.

    Args:
        y_true: Binary ground-truth array of shape `(n_samples,)`.
        y_pred: Optional score array used when `y_score` is omitted.
        y_score: Optional score or probability array of shape `(n_samples,)`.
        sample_weight: Optional per-row weights.

    Returns:
        float: Precision-recall area under the curve.
    """

    recall, precision = _pr_curve_arrays(y_true, y_score=y_score if y_score is not None else y_pred, sample_weight=sample_weight)
    return float(np.trapezoid(precision, recall))


def logloss(y_true, y_pred=None, *, y_score=None, sample_weight=None, epsilon: float = 1e-12, **_) -> float:
    """Compute binary log loss.

    Args:
        y_true: Binary ground-truth array of shape `(n_samples,)`.
        y_pred: Optional score array used when `y_score` is omitted.
        y_score: Optional probability array of shape `(n_samples,)`.
        sample_weight: Optional per-row weights.
        epsilon: Probability clipping value used for numerical stability.

    Returns:
        float: Weighted or unweighted log loss.
    """

    yt, _, ys, sw = _as_binary_arrays(y_true, y_score=y_score if y_score is not None else y_pred, sample_weight=sample_weight)
    clipped = np.clip(ys, epsilon, 1.0 - epsilon)
    losses = -(yt * np.log(clipped) + (1 - yt) * np.log(1.0 - clipped))
    return _mean(losses, sw)


def accuracy(y_true, y_pred=None, *, y_score=None, sample_weight=None, threshold: float = 0.5, **_) -> float:
    """Compute binary accuracy.

    Args:
        y_true: Binary ground-truth array of shape `(n_samples,)`.
        y_pred: Optional hard-label predictions of shape `(n_samples,)`.
        y_score: Optional probability array used to derive labels.
        sample_weight: Optional per-row weights.
        threshold: Classification threshold applied to scores.

    Returns:
        float: Weighted or unweighted accuracy.
    """

    yt, yp, ys, sw = _as_binary_arrays(y_true, y_pred=y_pred, y_score=y_score, sample_weight=sample_weight)
    labels = _label_predictions(ys, yp, threshold=threshold)
    return _mean((labels == yt).astype(float), sw)


def precision(y_true, y_pred=None, *, y_score=None, sample_weight=None, threshold: float = 0.5, **_) -> float:
    """Compute binary precision.

    Args:
        y_true: Binary ground-truth array of shape `(n_samples,)`.
        y_pred: Optional hard-label predictions of shape `(n_samples,)`.
        y_score: Optional probability array used to derive labels.
        sample_weight: Optional per-row weights.
        threshold: Classification threshold applied to scores.

    Returns:
        float: Weighted or unweighted precision.
    """

    yt, yp, ys, sw = _as_binary_arrays(y_true, y_pred=y_pred, y_score=y_score, sample_weight=sample_weight)
    labels = _label_predictions(ys, yp, threshold=threshold)
    positive = labels == 1
    if not np.any(positive):
        return 0.0
    true_positive = yt[positive] == 1
    weights = None if sw is None else sw[positive]
    return _mean(true_positive.astype(float), weights)


def recall(y_true, y_pred=None, *, y_score=None, sample_weight=None, threshold: float = 0.5, **_) -> float:
    """Compute binary recall.

    Args:
        y_true: Binary ground-truth array of shape `(n_samples,)`.
        y_pred: Optional hard-label predictions of shape `(n_samples,)`.
        y_score: Optional probability array used to derive labels.
        sample_weight: Optional per-row weights.
        threshold: Classification threshold applied to scores.

    Returns:
        float: Weighted or unweighted recall.
    """

    yt, yp, ys, sw = _as_binary_arrays(y_true, y_pred=y_pred, y_score=y_score, sample_weight=sample_weight)
    labels = _label_predictions(ys, yp, threshold=threshold)
    actual_positive = yt == 1
    if not np.any(actual_positive):
        return 0.0
    recovered = labels[actual_positive] == 1
    weights = None if sw is None else sw[actual_positive]
    return _mean(recovered.astype(float), weights)


def f1(y_true, y_pred=None, *, y_score=None, sample_weight=None, threshold: float = 0.5, **_) -> float:
    """Compute binary F1 score.

    Args:
        y_true: Binary ground-truth array of shape `(n_samples,)`.
        y_pred: Optional hard-label predictions of shape `(n_samples,)`.
        y_score: Optional probability array used to derive labels.
        sample_weight: Optional per-row weights.
        threshold: Classification threshold applied to scores.

    Returns:
        float: Weighted or unweighted F1 score.
    """

    prec = precision(y_true, y_pred=y_pred, y_score=y_score, sample_weight=sample_weight, threshold=threshold)
    rec = recall(y_true, y_pred=y_pred, y_score=y_score, sample_weight=sample_weight, threshold=threshold)
    if prec + rec == 0:
        return 0.0
    return float(2.0 * prec * rec / (prec + rec))


def brier_score(y_true, y_pred=None, *, y_score=None, sample_weight=None, **_) -> float:
    """Compute the Brier score for binary probabilities.

    Args:
        y_true: Binary ground-truth array of shape `(n_samples,)`.
        y_pred: Optional score array used when `y_score` is omitted.
        y_score: Optional probability array of shape `(n_samples,)`.
        sample_weight: Optional per-row weights.

    Returns:
        float: Weighted or unweighted Brier score.
    """

    yt, _, ys, sw = _as_binary_arrays(y_true, y_score=y_score if y_score is not None else y_pred, sample_weight=sample_weight)
    return _mean((ys - yt) ** 2, sw)


def gini(y_true, y_pred=None, *, y_score=None, sample_weight=None, **_) -> float:
    """Compute the Gini coefficient derived from ROC AUC.

    Args:
        y_true: Binary ground-truth array of shape `(n_samples,)`.
        y_pred: Optional score array used when `y_score` is omitted.
        y_score: Optional probability array of shape `(n_samples,)`.
        sample_weight: Optional per-row weights.

    Returns:
        float: Gini coefficient.
    """

    auc = roc_auc(y_true, y_pred=y_pred, y_score=y_score, sample_weight=sample_weight)
    return float(2.0 * auc - 1.0)
