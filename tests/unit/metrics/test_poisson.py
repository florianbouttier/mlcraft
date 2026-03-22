import numpy as np

from mlcraft.metrics import poisson


def test_poisson_metrics_support_exposure():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 1.0, 1.5])
    exposure = np.array([1.0, 2.0, 2.0])
    assert poisson.poisson_deviance(y_true, y_pred, exposure=exposure) >= 0
    assert poisson.predicted_mean(y_true, y_pred, exposure=exposure) == np.mean(y_pred)
    assert poisson.observed_mean(y_true, y_pred, exposure=exposure) == np.mean(y_true / exposure)


def test_poisson_calibration_diagnostics_return_bins():
    diagnostics = poisson.poisson_calibration_diagnostics(
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([0.2, 0.8, 1.5, 2.5]),
        n_bins=2,
    )
    assert diagnostics["predicted"].shape[0] >= 1

