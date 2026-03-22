"""Train/test split utilities."""

from __future__ import annotations

import numpy as np

from mlcraft.data.containers import ensure_columnar_data, slice_rows
from mlcraft.utils.random import normalize_random_state


def _resolve_test_count(n_samples: int, test_size: float | int) -> int:
    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("Float test_size must be between 0 and 1.")
        count = int(np.ceil(n_samples * test_size))
    else:
        count = int(test_size)
    if not 0 < count < n_samples:
        raise ValueError("test_size must leave at least one sample in each split.")
    return count


def train_test_split_random(data, y=None, *, test_size=0.2, random_state=None, shuffle=True):
    """Split data randomly into train and test partitions.

    Args:
        data: Feature data as a column mapping or 2D array.
        y: Optional target array of shape `(n_samples,)`.
        test_size: Test set size as a float ratio or an absolute row count.
        random_state: Optional random seed or generator.
        shuffle: Whether to shuffle rows before splitting.

    Returns:
        tuple: `(X_train, X_test)` when `y` is omitted, otherwise
        `(X_train, X_test, y_train, y_test)`.

    Example:
        >>> X_train, X_test, y_train, y_test = train_test_split_random(X, y, test_size=0.2, random_state=7)
        >>> len(y_train) + len(y_test) == len(y)
        True
    """

    columnar = ensure_columnar_data(data) if not isinstance(data, np.ndarray) else data
    n_samples = len(next(iter(columnar.values()))) if isinstance(columnar, dict) else columnar.shape[0]
    test_count = _resolve_test_count(n_samples, test_size)
    indices = np.arange(n_samples)
    if shuffle:
        rng = normalize_random_state(random_state)
        rng.shuffle(indices)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    X_train = slice_rows(data, train_idx)
    X_test = slice_rows(data, test_idx)
    if y is None:
        return X_train, X_test
    y_array = np.asarray(y)
    return X_train, X_test, y_array[train_idx], y_array[test_idx]


def train_test_split_time(data, y=None, *, time_column, test_size=0.2, ascending=True):
    """Split data chronologically with the newest rows in the test set.

    Args:
        data: Feature data as a column mapping or 2D array.
        y: Optional target array of shape `(n_samples,)`.
        time_column: Column name or array index containing the ordering key.
        test_size: Test set size as a float ratio or an absolute row count.
        ascending: Whether older rows come first in the ordering.

    Returns:
        tuple: `(X_train, X_test)` when `y` is omitted, otherwise
        `(X_train, X_test, y_train, y_test)`.

    Example:
        >>> X_train, X_test, y_train, y_test = train_test_split_time(X, y, time_column="date", test_size=0.2)
        >>> len(y_train) + len(y_test) == len(y)
        True
    """

    if isinstance(data, np.ndarray):
        if not isinstance(time_column, int):
            raise TypeError("time_column must be an integer index for 2D arrays.")
        time_values = np.asarray(data[:, time_column])
    else:
        columnar = ensure_columnar_data(data)
        time_values = np.asarray(columnar[time_column])
    order = np.argsort(time_values)
    if not ascending:
        order = order[::-1]
    n_samples = order.shape[0]
    test_count = _resolve_test_count(n_samples, test_size)
    train_idx = order[:-test_count]
    test_idx = order[-test_count:]
    X_train = slice_rows(data, train_idx)
    X_test = slice_rows(data, test_idx)
    if y is None:
        return X_train, X_test
    y_array = np.asarray(y)
    return X_train, X_test, y_array[train_idx], y_array[test_idx]
