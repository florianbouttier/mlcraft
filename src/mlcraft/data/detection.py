"""Low-level type detection helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from mlcraft.core.schema import ColumnDType


def is_na_mask(values: np.ndarray) -> np.ndarray:
    """Return a missing-value mask without coercing the whole array.

    Args:
        values: Column values to inspect.

    Returns:
        np.ndarray: Boolean mask of shape `(n_samples,)` where `True` marks a
        missing value.
    """

    array = np.asarray(values)
    if array.dtype.kind == "f":
        return np.isnan(array)
    if array.dtype.kind == "M":
        return np.isnat(array)
    if array.dtype.kind in {"i", "u", "b"}:
        return np.zeros(array.shape, dtype=bool)

    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, float):
            return np.isnan(value)
        if isinstance(value, np.datetime64):
            return np.isnat(value)
        return False

    return np.asarray([_is_missing(value) for value in array], dtype=bool)


def infer_cardinality(values: np.ndarray) -> int:
    """Return the number of distinct non-missing values.

    Args:
        values: Column values to inspect.

    Returns:
        int: Number of unique non-missing values.
    """

    array = np.asarray(values)
    mask = ~is_na_mask(array)
    if not np.any(mask):
        return 0
    filtered = array[mask]
    if filtered.dtype.kind == "M":
        return int(np.unique(filtered.astype("datetime64[ns]").astype("int64")).shape[0])
    return int(np.unique(filtered).shape[0])


def _all_instances(values: np.ndarray, expected: type) -> bool:
    return all(isinstance(value, expected) for value in values.tolist())


def detect_datetime_like(values: np.ndarray, *, sample_size: int = 50, min_success_ratio: float = 0.8) -> bool:
    """Detect simple datetime-like columns.

    Args:
        values: Non-missing values to inspect.
        sample_size: Maximum number of values sampled for string parsing.
        min_success_ratio: Minimum parse success ratio required to classify
            the column as datetime-like.

    Returns:
        bool: `True` when the column looks like datetime data.
    """

    array = np.asarray(values)
    if array.dtype.kind == "M":
        return True
    mask = ~is_na_mask(array)
    filtered = array[mask]
    if filtered.size == 0:
        return False
    sample = filtered[: min(sample_size, filtered.size)]
    if sample.dtype.kind in {"U", "S"} or sample.dtype == object:
        success = 0
        for value in sample.tolist():
            if isinstance(value, np.datetime64):
                success += 1
                continue
            if not isinstance(value, str):
                continue
            try:
                np.datetime64(value)
                success += 1
            except ValueError:
                continue
        return success / len(sample) >= min_success_ratio
    return False


def detect_categorical_like(
    values: np.ndarray,
    *,
    max_cardinality: int = 20,
    max_unique_ratio: float = 0.05,
    infer_numeric: bool = False,
) -> bool:
    """Detect categorical-like columns using simple heuristics.

    Args:
        values: Non-missing values to inspect.
        max_cardinality: Maximum unique count used by the heuristic.
        max_unique_ratio: Maximum unique ratio used by the heuristic.
        infer_numeric: Whether integer-like columns may be treated as
            categorical.

    Returns:
        bool: `True` when the column should be interpreted as categorical.
    """

    array = np.asarray(values)
    mask = ~is_na_mask(array)
    filtered = array[mask]
    if filtered.size == 0:
        return False
    cardinality = infer_cardinality(filtered)
    ratio = cardinality / filtered.size
    if array.dtype.kind in {"U", "S"}:
        return True
    if array.dtype == object:
        if all(isinstance(value, str) for value in filtered.tolist()):
            return True
        if infer_numeric:
            return cardinality <= max_cardinality or ratio <= max_unique_ratio
        return not np.issubdtype(np.asarray(filtered).dtype, np.number)
    if infer_numeric and array.dtype.kind in {"i", "u"}:
        return cardinality <= max_cardinality or ratio <= max_unique_ratio
    return False


def infer_primitive_dtype(
    values: np.ndarray,
    *,
    datetime_detection_enabled: bool = True,
    categorical_max_cardinality: int = 20,
    categorical_max_ratio: float = 0.05,
    categorical_infer_numeric: bool = False,
) -> ColumnDType:
    """Infer the semantic dtype of a column from observed values.

    Missing values are ignored for type detection so sparse numeric columns
    keep their intended dtype instead of collapsing to generic categorical
    object columns.

    Args:
        values: Column values to inspect.
        datetime_detection_enabled: Whether datetime heuristics are enabled.
        categorical_max_cardinality: Maximum unique count used by categorical
            heuristics.
        categorical_max_ratio: Maximum unique ratio used by categorical
            heuristics.
        categorical_infer_numeric: Whether low-cardinality integer columns may
            be treated as categorical.

    Returns:
        ColumnDType: Inferred semantic dtype.
    """

    array = np.asarray(values)
    mask = ~is_na_mask(array)
    filtered = array[mask]
    if filtered.size == 0:
        return ColumnDType.FLOAT
    if array.dtype.kind == "b" or _all_instances(filtered, bool):
        return ColumnDType.BOOLEAN
    if datetime_detection_enabled and detect_datetime_like(filtered):
        return ColumnDType.DATETIME
    if categorical_infer_numeric and detect_categorical_like(
        filtered,
        max_cardinality=categorical_max_cardinality,
        max_unique_ratio=categorical_max_ratio,
        infer_numeric=categorical_infer_numeric,
    ):
        return ColumnDType.CATEGORICAL
    if array.dtype.kind in {"i", "u"}:
        return ColumnDType.INTEGER
    if array.dtype.kind == "f":
        finite = filtered.astype(float)
        if np.all(np.isfinite(finite)) and np.all(np.equal(finite, np.floor(finite))):
            return ColumnDType.INTEGER
        return ColumnDType.FLOAT
    if array.dtype.kind in {"U", "S"}:
        return ColumnDType.CATEGORICAL
    if array.dtype == object:
        as_list = filtered.tolist()
        if all(isinstance(value, bool) for value in as_list):
            return ColumnDType.BOOLEAN
        if all(isinstance(value, (int, np.integer)) and not isinstance(value, bool) for value in as_list):
            return ColumnDType.INTEGER
        if all(isinstance(value, (int, np.integer, float, np.floating)) and not isinstance(value, bool) for value in as_list):
            numeric = np.asarray(as_list, dtype=float)
            if np.all(np.equal(numeric, np.floor(numeric))):
                return ColumnDType.INTEGER
            return ColumnDType.FLOAT
        if detect_categorical_like(
            filtered,
            max_cardinality=categorical_max_cardinality,
            max_unique_ratio=categorical_max_ratio,
            infer_numeric=categorical_infer_numeric,
        ):
            return ColumnDType.CATEGORICAL
        return ColumnDType.CATEGORICAL
    return ColumnDType.FLOAT
