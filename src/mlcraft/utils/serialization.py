"""Serialization helpers for numpy arrays, dataclasses, and enums."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import numpy as np


def to_serializable(value: Any, *, include_arrays: bool = True) -> Any:
    """Convert nested objects into JSON-friendly structures.

    Args:
        value: Object to serialize.
        include_arrays: Whether to inline array values instead of compact
            array metadata.

    Returns:
        Any: JSON-friendly representation of the input object.
    """

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: to_serializable(val, include_arrays=include_arrays) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_serializable(val, include_arrays=include_arrays) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item, include_arrays=include_arrays) for item in value]
    if isinstance(value, np.ndarray):
        if not include_arrays:
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        if value.dtype.kind == "M":
            return value.astype("datetime64[ns]").astype(str).tolist()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def from_serializable_array(value: Any, *, dtype: str | None = None) -> np.ndarray:
    """Convert serialized array content back into a numpy array.

    Args:
        value: Serialized array-like payload.
        dtype: Optional dtype override.

    Returns:
        np.ndarray: Reconstructed numpy array.
    """

    return np.asarray(value, dtype=dtype)
