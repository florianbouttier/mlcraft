import numpy as np

from mlcraft.data.inference import InferenceOptions, infer_schema


def test_infer_schema_detects_core_types():
    data = {
        "ints": np.array([1, None, 2, 3], dtype=object),
        "floats": np.array([1.1, np.nan, 2.5, 3.2]),
        "bools": np.array([True, False, True, False]),
        "dates": np.array(["2024-01-01", None, "2024-01-03", "2024-01-04"], dtype=object),
        "cats": np.array(["a", "b", None, "a"], dtype=object),
    }
    schema = infer_schema(data)
    assert schema.get("ints").dtype.value == "integer"
    assert schema.get("floats").dtype.value == "float"
    assert schema.get("bools").dtype.value == "boolean"
    assert schema.get("dates").dtype.value == "datetime"
    assert schema.get("cats").dtype.value == "categorical"


def test_infer_schema_handles_quasi_empty_columns():
    data = {
        "sparse_float": np.array([None, None, None, 1.5], dtype=object),
        "sparse_int": np.array([None, None, None, 2], dtype=object),
    }
    schema = infer_schema(data)
    assert schema.get("sparse_float").dtype.value == "float"
    assert schema.get("sparse_int").dtype.value == "integer"
    assert schema.get("sparse_float").na_ratio == 0.75


def test_infer_schema_cardinality_and_serialization():
    data = {"cat": np.array(["a", "a", "b", None], dtype=object)}
    schema = infer_schema(data)
    assert schema.get("cat").cardinality == 2
    assert schema.to_dict()["columns"][0]["name"] == "cat"


def test_numeric_categorical_inference_can_be_enabled():
    data = {"code": np.array([1, 1, 2, 2, 1, 2])}
    schema = infer_schema(data, inference_options=InferenceOptions(categorical_infer_numeric=True, categorical_max_cardinality=3))
    assert schema.get("code").dtype.value == "categorical"

