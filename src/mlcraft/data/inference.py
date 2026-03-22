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
    datetime_detection_enabled: bool = True
    categorical_max_cardinality: int = 20
    categorical_max_ratio: float = 0.05
    categorical_infer_numeric: bool = False


class SchemaInferer:
    """Infer a DataSchema from column-oriented numpy data."""

    def __init__(self, inference_options: InferenceOptions | None = None) -> None:
        self.inference_options = inference_options or InferenceOptions()

    def infer(
        self,
        data,
        *,
        roles: Mapping[str, ColumnRole | str] | None = None,
    ) -> DataSchema:
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
    """Infer a DataSchema from numpy-friendly inputs."""

    inferer = SchemaInferer(inference_options=InferenceOptions(**inference_options) if isinstance(inference_options, dict) else inference_options)
    return inferer.infer(data, roles=roles)
