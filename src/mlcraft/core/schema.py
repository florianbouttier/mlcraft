"""Schema objects shared across the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from mlcraft.utils.serialization import to_serializable


class ColumnDType(str, Enum):
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"


class ColumnRole(str, Enum):
    FEATURE = "feature"
    TARGET = "target"
    WEIGHT = "weight"
    EXPOSURE = "exposure"
    ID = "id"


@dataclass
class ColumnSchema:
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
        return to_serializable(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ColumnSchema":
        return cls(**payload)


@dataclass
class DataSchema:
    columns: list[ColumnSchema] = field(default_factory=list)

    def __iter__(self) -> Iterable[ColumnSchema]:
        return iter(self.columns)

    def __len__(self) -> int:
        return len(self.columns)

    def names(self) -> list[str]:
        return [column.name for column in self.columns]

    def get(self, name: str) -> ColumnSchema:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(name)

    def has(self, name: str) -> bool:
        return any(column.name == name for column in self.columns)

    def by_role(self, role: ColumnRole | str) -> list[ColumnSchema]:
        target_role = role if isinstance(role, ColumnRole) else ColumnRole(str(role))
        return [column for column in self.columns if column.role == target_role]

    def feature_columns(self) -> list[ColumnSchema]:
        return [column for column in self.columns if column.role in (None, ColumnRole.FEATURE)]

    def target_column(self) -> ColumnSchema | None:
        targets = self.by_role(ColumnRole.TARGET)
        return targets[0] if targets else None

    def to_dict(self) -> dict[str, Any]:
        return {"columns": [column.to_dict() for column in self.columns]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DataSchema":
        return cls(columns=[ColumnSchema.from_dict(item) for item in payload.get("columns", [])])
