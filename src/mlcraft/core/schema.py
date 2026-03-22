"""Schema objects shared across the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from mlcraft.utils.serialization import to_serializable


class ColumnDType(str, Enum):
    """Enumerate the semantic dtypes supported by `mlcraft`."""

    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"


class ColumnRole(str, Enum):
    """Enumerate the semantic roles that a column can play."""

    FEATURE = "feature"
    TARGET = "target"
    WEIGHT = "weight"
    EXPOSURE = "exposure"
    ID = "id"


@dataclass
class ColumnSchema:
    """Describe one logical column in a dataset.

    The schema stores semantic information inferred from raw numpy arrays or
    declared explicitly by the caller. It keeps missing-value statistics and
    an optional business role so downstream components can reuse the same
    metadata without repeating configuration.

    Args:
        name: Column name as exposed to the rest of the package.
        dtype: Semantic dtype used by inference, adapters, and reporting.
        nullable: Whether at least one missing value is allowed in the column.
        na_ratio: Share of missing values in the observed sample.
        cardinality: Number of distinct non-missing values when available.
        role: Optional semantic role such as feature, target, or exposure.

    Example:
        >>> column = ColumnSchema(name="target", dtype="float", role="target")
        >>> column.dtype.value
        'float'
    """

    name: str
    dtype: ColumnDType | str
    nullable: bool = False
    na_ratio: float = 0.0
    cardinality: int | None = None
    role: ColumnRole | str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, ColumnDType):
            self.dtype = ColumnDType(str(self.dtype))
        if self.role is not None and not isinstance(self.role, ColumnRole):
            self.role = ColumnRole(str(self.role))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the column schema into a JSON-friendly dictionary.

        Returns:
            dict[str, Any]: Serialized column metadata.
        """

        return to_serializable(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ColumnSchema":
        """Build a `ColumnSchema` from serialized data.

        Args:
            payload: Serialized payload produced by `to_dict`.

        Returns:
            ColumnSchema: Reconstructed schema instance.
        """

        return cls(**payload)


@dataclass
class DataSchema:
    """Group column schemas for a full dataset.

    `DataSchema` acts as the shared contract between inference, adapters,
    model wrappers, and reporting. The object is intentionally lightweight so
    it can be serialized, versioned, and reused across runs.

    Args:
        columns: Ordered list of column descriptors.

    Example:
        >>> schema = DataSchema(columns=[ColumnSchema("age", "integer"), ColumnSchema("target", "float", role="target")])
        >>> schema.names()
        ['age', 'target']
    """

    columns: list[ColumnSchema] = field(default_factory=list)

    def __iter__(self) -> Iterable[ColumnSchema]:
        return iter(self.columns)

    def __len__(self) -> int:
        return len(self.columns)

    def names(self) -> list[str]:
        """Return column names in schema order.

        Returns:
            list[str]: Column names as stored in the schema.
        """

        return [column.name for column in self.columns]

    def get(self, name: str) -> ColumnSchema:
        """Return the schema entry for one column.

        Args:
            name: Name of the column to fetch.

        Returns:
            ColumnSchema: Matching column schema.

        Raises:
            KeyError: If the column is not present.
        """

        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(name)

    def has(self, name: str) -> bool:
        """Return whether a column exists in the schema.

        Args:
            name: Column name to check.

        Returns:
            bool: `True` when the column exists.
        """

        return any(column.name == name for column in self.columns)

    def by_role(self, role: ColumnRole | str) -> list[ColumnSchema]:
        """Return all columns matching a semantic role.

        Args:
            role: Role to match.

        Returns:
            list[ColumnSchema]: Matching columns in schema order.
        """

        target_role = role if isinstance(role, ColumnRole) else ColumnRole(str(role))
        return [column for column in self.columns if column.role == target_role]

    def feature_columns(self) -> list[ColumnSchema]:
        """Return feature columns, including columns with no explicit role.

        Returns:
            list[ColumnSchema]: Columns that can be treated as features.
        """

        return [column for column in self.columns if column.role in (None, ColumnRole.FEATURE)]

    def target_column(self) -> ColumnSchema | None:
        """Return the first target column when one is defined.

        Returns:
            ColumnSchema | None: Target column or `None` when absent.
        """

        targets = self.by_role(ColumnRole.TARGET)
        return targets[0] if targets else None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the dataset schema into a JSON-friendly dictionary.

        Returns:
            dict[str, Any]: Serialized dataset schema.
        """

        return {"columns": [column.to_dict() for column in self.columns]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DataSchema":
        """Build a `DataSchema` from serialized data.

        Args:
            payload: Serialized payload produced by `to_dict`.

        Returns:
            DataSchema: Reconstructed dataset schema.
        """

        return cls(columns=[ColumnSchema.from_dict(item) for item in payload.get("columns", [])])
