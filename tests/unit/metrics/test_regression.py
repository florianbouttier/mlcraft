import numpy as np

from mlcraft.metrics import regression


def test_regression_metrics_on_simple_case():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])
    assert regression.mae(y_true, y_pred) == 1.0 / 3.0
    assert regression.mse(y_true, y_pred) == 1.0 / 3.0
    assert round(regression.rmse(y_true, y_pred), 6) == round((1.0 / 3.0) ** 0.5, 6)
    assert regression.r2(y_true, y_pred) < 1.0
    assert regression.medae(y_true, y_pred) == 0.0


def test_regression_metrics_support_weights():
    y_true = np.array([1.0, 10.0])
    y_pred = np.array([2.0, 10.0])
    weights = np.array([3.0, 1.0])
    assert regression.mae(y_true, y_pred, sample_weight=weights) == 0.75

