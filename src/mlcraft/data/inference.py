"""Schema inference based on column-oriented numpy arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from mlcraft.core.schema import ColumnRole, ColumnSchema, DataSchema
from mlcraft.data.containers import ensure_columnar_data
from mlcraft.data.detection import infer_cardinality, infer_primitive_dtype, is_na_mask


@dataclass
class InferenceOptions:
    """Configure schema inference heuristics.

    Args:
        datetime_detection_enabled: Whether simple datetime detection is
            enabled for string-like and object arrays.
        categorical_max_cardinality: Maximum unique count used by categorical
            heuristics when numeric inference is enabled.
        categorical_max_ratio: Maximum unique ratio used by categorical
            heuristics when numeric inference is enabled.
        categorical_infer_numeric: Whether low-cardinality integer columns can
            be tagged as categorical instead of numeric.
    """

    datetime_detection_enabled: bool = True
    categorical_max_cardinality: int = 20
    categorical_max_ratio: float = 0.05
    categorical_infer_numeric: bool = False


class SchemaInferer:
    """Infer a `DataSchema` from column-oriented numpy data.

    The inferer keeps inference logic separate from feature transformation so
    schema metadata can be reused by adapters, models, and reports.

    Args:
        inference_options: Optional heuristic overrides for type detection.

    Example:
        >>> inferer = SchemaInferer()
        >>> schema = inferer.infer({"x": np.array([1.0, np.nan, 2.0])})
        >>> schema.get("x").dtype.value
        'float'
    """

    def __init__(self, inference_options: InferenceOptions | None = None) -> None:
        self.inference_options = inference_options or InferenceOptions()

    def infer(
        self,
        data,
        *,
        roles: Mapping[str, ColumnRole | str] | None = None,
    ) -> DataSchema:
        """Infer schema metadata from column-oriented data.

        Args:
            data: Column mapping or 2D numpy array to inspect.
            roles: Optional semantic roles keyed by column name.

        Returns:
            DataSchema: Inferred dataset schema preserving column order.
        """

        columnar = ensure_columnar_data(data)
        columns: list[ColumnSchema] = []
        for name, values in columnar.items():
            mask = is_na_mask(values)
            dtype = infer_primitive_dtype(
                values,
                datetime_detection_enabled=self.inference_options.datetime_detection_enabled,
                categorical_max_cardinality=self.inference_options.categorical_max_cardinality,
                categorical_max_ratio=self.inference_options.categorical_max_ratio,
                categorical_infer_numeric=self.inference_options.categorical_infer_numeric,
            )
            role = roles.get(name) if roles else None
            columns.append(
                ColumnSchema(
                    name=name,
                    dtype=dtype,
                    nullable=bool(np.any(mask)),
                    na_ratio=float(mask.mean()) if mask.size else 0.0,
                    cardinality=infer_cardinality(values),
                    role=role,
                )
            )
        return DataSchema(columns=columns)


def infer_schema(data, *, roles=None, inference_options=None) -> DataSchema:
    """Infer a `DataSchema` from numpy-friendly inputs.

    Args:
        data: Column mapping or 2D array to inspect.
        roles: Optional semantic roles keyed by column name.
        inference_options: Optional `InferenceOptions` instance or dictionary
            of inference overrides.

    Returns:
        DataSchema: Inferred schema describing semantic dtypes, missing-value
        ratios, and optional roles.

    Example:
        >>> schema = infer_schema({"city": np.array(["Paris", None, "Lyon"], dtype=object)})
        >>> schema.get("city").dtype.value
        'categorical'
    """

    inferer = SchemaInferer(inference_options=InferenceOptions(**inference_options) if isinstance(inference_options, dict) else inference_options)
    return inferer.infer(data, roles=roles)
