import numpy as np

from mlcraft.data.adapters import fit_feature_adapter
from mlcraft.data.inference import infer_schema


def test_feature_adapter_transforms_columnar_data(classification_data):
    X, _ = classification_data
    schema = infer_schema(X)
    adapter = fit_feature_adapter(X, schema)
    matrix, metadata = adapter.transform(X, backend="xgboost")
    assert matrix.shape == (8, 2)
    assert metadata["categorical_indices"] == []


def test_feature_adapter_preserves_catboost_metadata(classification_data):
    X, _ = classification_data
    schema = infer_schema(X)
    adapter = fit_feature_adapter(X, schema)
    matrix, metadata = adapter.transform(X, backend="catboost")
    assert matrix.shape == (8, 2)
    assert metadata["categorical_indices"] == [1]
    assert matrix.dtype == object


def test_feature_adapter_adds_exposure_column(poisson_data):
    X, _, exposure = poisson_data
    schema = infer_schema(X)
    adapter = fit_feature_adapter(X, schema)
    matrix, _ = adapter.transform(X, backend="xgboost", exposure=exposure)
    assert matrix.shape[1] == 2
    assert np.allclose(matrix[:, -1], np.log(exposure))

