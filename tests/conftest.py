import numpy as np
import pytest


@pytest.fixture
def regression_data():
    X = {
        "num_a": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        "num_b": np.array([0, 1, 0, 1, 0, 1]),
    }
    y = np.array([1.2, 2.1, 2.8, 4.2, 5.1, 5.9])
    return X, y


@pytest.fixture
def classification_data():
    X = {
        "num_a": np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.85, 0.05, 0.95]),
        "cat_a": np.array(["a", "a", "b", "b", "a", "b", "a", "b"], dtype=object),
    }
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=int)
    return X, y


@pytest.fixture
def poisson_data():
    X = {
        "x": np.array([0.2, 0.5, 0.7, 1.0, 1.5, 2.0]),
    }
    y = np.array([0, 1, 1, 2, 3, 5], dtype=float)
    exposure = np.array([1.0, 1.0, 1.5, 1.5, 2.0, 2.0], dtype=float)
    return X, y, exposure


@pytest.fixture
def temporal_data():
    X = {
        "date": np.array(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            dtype="datetime64[D]",
        ),
        "value": np.array([1, 2, 3, 4, 5]),
    }
    y = np.array([10, 20, 30, 40, 50])
    return X, y


@pytest.fixture
def btcusdt_parquet_data():
    pyarrow = pytest.importorskip("pyarrow.parquet")
    table = pyarrow.read_table("tests/btcusdt_regression_h4.parquet")
    X = {
        name: table[name].to_numpy(zero_copy_only=False)
        for name in table.column_names
        if name not in {"target_regression", "future_return"}
    }
    y_regression = table["target_regression"].to_numpy(zero_copy_only=False).astype(float)
    y_classification = (y_regression > 0.0).astype(int)
    return X, y_regression, y_classification
