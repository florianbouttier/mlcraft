"""Feature adapters from column-oriented data to backend-friendly matrices."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mlcraft.core.schema import ColumnDType, DataSchema
from mlcraft.data.containers import ensure_columnar_data
from mlcraft.data.detection import is_na_mask


@dataclass
class FeatureAdapterConfig:
    """Configure how columnar features are converted for model backends.

    Args:
        feature_order: Optional explicit feature order. Defaults to schema
            order when omitted.
        categorical_missing_token: Placeholder used for missing categorical
            values during encoding.
        numeric_missing_value: Reserved numeric missing value placeholder.
        categorical_unknown_value: Code assigned to unseen categorical values.
        datetime_unit: Datetime precision used when datetimes are converted to
            integer timestamps.
    """

    feature_order: list[str] | None = None
    categorical_missing_token: str = "__MISSING__"
    numeric_missing_value: float = np.nan
    categorical_unknown_value: int = -1
    datetime_unit: str = "ns"


@dataclass
class FittedFeatureAdapter:
    """Store a fitted feature adapter ready to transform new data.

    Args:
        schema: Schema describing the source columns.
        config: Adapter configuration used during fitting.
        feature_order: Stable feature order used during transformation.
        categorical_maps: Encodings learned for categorical columns.
        categorical_indices: Feature indices treated as categorical by
            CatBoost.
    """

    schema: DataSchema
    config: FeatureAdapterConfig
    feature_order: list[str]
    categorical_maps: dict[str, dict[Any, int]] = field(default_factory=dict)
    categorical_indices: list[int] = field(default_factory=list)

    def transform(
        self,
        data,
        *,
        backend: str | None = None,
        exposure: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Transform raw features with the fitted adapter.

        Args:
            data: Column mapping or 2D array to transform.
            backend: Optional backend name used to adjust categorical handling.
            exposure: Optional exposure vector appended as log-exposure for
                Poisson workflows.

        Returns:
            tuple[np.ndarray, dict[str, Any]]: Transformed feature matrix of
            shape `(n_samples, n_features)` and backend-specific metadata.
        """

        return transform_feature_data(data, self, backend=backend, exposure=exposure)


def _to_numeric_array(values: np.ndarray, dtype: ColumnDType, unit: str) -> np.ndarray:
    if dtype == ColumnDType.BOOLEAN:
        mask = is_na_mask(values)
        array = np.asarray(values, dtype=object)
        result = np.empty(array.shape[0], dtype=float)
        result[mask] = np.nan
        result[~mask] = array[~mask].astype(bool).astype(float)
        return result
    if dtype == ColumnDType.DATETIME:
        array = np.asarray(values)
        if array.dtype.kind != "M":
            converted = np.asarray([np.datetime64(value) if value is not None else np.datetime64("NaT") for value in array], dtype=f"datetime64[{unit}]")
        else:
            converted = array.astype(f"datetime64[{unit}]")
        result = converted.astype("int64").astype(float)
        result[np.isnat(converted)] = np.nan
        return result
    mask = is_na_mask(values)
    array = np.asarray(values, dtype=object)
    result = np.empty(array.shape[0], dtype=float)
    result[mask] = np.nan
    if np.any(~mask):
        result[~mask] = np.asarray(array[~mask], dtype=float)
    return result


def fit_feature_adapter(
    data,
    schema: DataSchema,
    *,
    config: FeatureAdapterConfig | None = None,
) -> FittedFeatureAdapter:
    """Fit a feature adapter from schema-aware columnar data.

    Args:
        data: Column mapping or 2D array used to learn categorical encodings.
        schema: Dataset schema aligned with `data`.
        config: Optional adapter configuration overrides.

    Returns:
        FittedFeatureAdapter: Adapter carrying feature order and categorical
        encodings.

    Example:
        >>> X = {"city": np.array(["Paris", "Lyon"], dtype=object)}
        >>> schema = infer_schema(X)
        >>> adapter = fit_feature_adapter(X, schema)
        >>> adapter.feature_order
        ['city']
    """

    columnar = ensure_columnar_data(data)
    cfg = config or FeatureAdapterConfig()
    order = cfg.feature_order or schema.names()
    categorical_maps: dict[str, dict[Any, int]] = {}
    categorical_indices: list[int] = []

    for index, name in enumerate(order):
        column_schema = schema.get(name)
        if column_schema.dtype != ColumnDType.CATEGORICAL:
            continue
        categorical_indices.append(index)
        values = np.asarray(columnar[name], dtype=object)
        mask = is_na_mask(values)
        normalized = [cfg.categorical_missing_token if mask[pos] else values[pos] for pos in range(values.shape[0])]
        unique_values = list(dict.fromkeys(normalized))
        categorical_maps[name] = {value: code for code, value in enumerate(unique_values)}

    return FittedFeatureAdapter(
        schema=schema,
        config=cfg,
        feature_order=list(order),
        categorical_maps=categorical_maps,
        categorical_indices=categorical_indices,
    )


def transform_feature_data(
    data,
    fitted: FittedFeatureAdapter,
    *,
    backend: str | None = None,
    exposure: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Transform features for a target backend.

    Args:
        data: Column mapping or 2D array to transform.
        fitted: Fitted adapter holding feature order and encodings.
        backend: Optional backend name. `catboost` keeps native categorical
            columns, while other backends receive numeric encodings.
        exposure: Optional exposure vector appended as log-exposure.

    Returns:
        tuple[np.ndarray, dict[str, Any]]: Transformed feature matrix of shape
        `(n_samples, n_features)` and backend-specific metadata.
    """

    columnar = ensure_columnar_data(data)
    numeric_columns: list[np.ndarray] = []
    catboost_columns: list[np.ndarray] = []
    for name in fitted.feature_order:
        schema_column = fitted.schema.get(name)
        values = np.asarray(columnar[name])
        if schema_column.dtype == ColumnDType.CATEGORICAL:
            mapping = fitted.categorical_maps.get(name, {})
            values_obj = np.asarray(values, dtype=object)
            mask = is_na_mask(values_obj)
            normalized = np.asarray(
                [fitted.config.categorical_missing_token if mask[idx] else values_obj[idx] for idx in range(values_obj.shape[0])],
                dtype=object,
            )
            if backend == "catboost":
                catboost_columns.append(normalized.astype(object))
            else:
                encoded = np.asarray(
                    [mapping.get(item, fitted.config.categorical_unknown_value) for item in normalized.tolist()],
                    dtype=float,
                )
                numeric_columns.append(encoded)
            continue
        converted = _to_numeric_array(values, schema_column.dtype, fitted.config.datetime_unit)
        numeric_columns.append(converted)
        if backend == "catboost":
            catboost_columns.append(converted.astype(float))

    if exposure is not None:
        log_exposure = np.log(np.clip(np.asarray(exposure, dtype=float), 1e-12, None))
        numeric_columns.append(log_exposure.astype(float))
        if backend == "catboost":
            catboost_columns.append(log_exposure.astype(float))

    if backend == "catboost":
        matrix = np.empty((len(next(iter(columnar.values()))), len(catboost_columns)), dtype=object)
        for index, column in enumerate(catboost_columns):
            matrix[:, index] = column
        metadata = {"categorical_indices": list(fitted.categorical_indices)}
        return matrix, metadata

    matrix = np.column_stack(numeric_columns) if numeric_columns else np.empty((len(next(iter(columnar.values()))), 0))
    return matrix.astype(float, copy=False), {"categorical_indices": []}
