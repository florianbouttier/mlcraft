"""Container helpers for column-oriented numpy data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mlcraft.typing import ColumnarData


def column_lengths(data: ColumnarData) -> dict[str, int]:
    """Return the row count of each column.

    Args:
        data: Column-oriented numpy mapping.

    Returns:
        dict[str, int]: Row counts keyed by column name.
    """

    return {name: len(np.asarray(values)) for name, values in data.items()}


def feature_names(data: ColumnarData | np.ndarray, *, feature_order: list[str] | None = None) -> list[str]:
    """Return feature names for a supported container.

    Args:
        data: Column mapping or 2D array.
        feature_order: Optional explicit names for a 2D array input.

    Returns:
        list[str]: Feature names in the order used by downstream code.

    Raises:
        ValueError: If `data` is a 2D array and `feature_order` has the wrong
            length.
    """

    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError("Expected a 2D array.")
        if feature_order is not None:
            if len(feature_order) != data.shape[1]:
                raise ValueError("feature_order length does not match array shape.")
            return list(feature_order)
        return [f"f{index}" for index in range(data.shape[1])]
    return list(data.keys())


def ensure_columnar_data(
    data: ColumnarData | np.ndarray,
    *,
    feature_order: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Normalize supported inputs into a column-oriented mapping.

    Args:
        data: Column mapping or 2D array to normalize.
        feature_order: Optional feature names for a 2D array input.

    Returns:
        dict[str, np.ndarray]: Mapping of column names to 1D arrays of shape
        `(n_samples,)`.

    Raises:
        TypeError: If the input is neither a mapping nor a 2D array.
        ValueError: If array dimensions are invalid or column lengths differ.

    Example:
        >>> data = ensure_columnar_data(np.array([[1.0, 2.0], [3.0, 4.0]]), feature_order=["a", "b"])
        >>> sorted(data)
        ['a', 'b']
    """

    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError("Expected a 2D array.")
        names = feature_names(data, feature_order=feature_order)
        return {name: np.asarray(data[:, index]) for index, name in enumerate(names)}
    if not isinstance(data, Mapping):
        raise TypeError("Expected a mapping of column arrays or a 2D numpy array.")
    result = {str(name): np.asarray(values) for name, values in data.items()}
    lengths = set(column_lengths(result).values())
    if len(lengths) > 1:
        raise ValueError("All columns must have the same length.")
    return result


def ensure_2d_array(
    data: ColumnarData | np.ndarray,
    *,
    feature_order: list[str] | None = None,
) -> np.ndarray:
    """Convert supported inputs into a 2D numpy array.

    Args:
        data: Column mapping or 2D array to normalize.
        feature_order: Optional feature order when `data` is a mapping.

    Returns:
        np.ndarray: Array of shape `(n_samples, n_features)`.

    Raises:
        ValueError: If the input array is not 2D.
    """

    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError("Expected a 2D array.")
        return data
    columnar = ensure_columnar_data(data)
    names = feature_order or list(columnar.keys())
    return np.column_stack([np.asarray(columnar[name]) for name in names])


def slice_rows(
    data: ColumnarData | np.ndarray | None,
    indices: np.ndarray,
) -> dict[str, np.ndarray] | np.ndarray | None:
    """Slice rows from arrays or column mappings.

    Args:
        data: Container to slice. `None` is returned unchanged.
        indices: Row indices to keep.

    Returns:
        dict[str, np.ndarray] | np.ndarray | None: Sliced container preserving
        the original shape convention.
    """

    if data is None:
        return None
    if isinstance(data, np.ndarray):
        return np.asarray(data)[indices]
    columnar = ensure_columnar_data(data)
    return {name: values[indices] for name, values in columnar.items()}
