"""Typing helpers shared across the package."""

from __future__ import annotations

from typing import Any, Literal, Mapping, MutableMapping, Union

import numpy as np
from numpy.typing import NDArray

Array1D = NDArray[Any]
Array2D = NDArray[Any]
ColumnarData = Union[Mapping[str, NDArray[Any]], MutableMapping[str, NDArray[Any]]]
IndexArray = NDArray[np.int_]
MetricName = str
BackendName = Literal["xgboost", "lightgbm", "catboost"]
TaskTypeName = Literal["regression", "classification", "poisson"]
