from mlcraft.core.schema import ColumnRole, ColumnSchema, DataSchema


def test_column_schema_serialization():
    schema = ColumnSchema(name="age", dtype="integer", nullable=True, na_ratio=0.1, cardinality=4, role="feature")
    payload = schema.to_dict()
    restored = ColumnSchema.from_dict(payload)
    assert restored.name == "age"
    assert restored.dtype.value == "integer"
    assert restored.role.value == "feature"


def test_data_schema_helpers():
    schema = DataSchema(
        columns=[
            ColumnSchema("id", "integer", role=ColumnRole.ID),
            ColumnSchema("target", "float", role=ColumnRole.TARGET),
            ColumnSchema("feature", "float", role=ColumnRole.FEATURE),
        ]
    )
    assert schema.names() == ["id", "target", "feature"]
    assert schema.target_column().name == "target"
    assert [column.name for column in schema.feature_columns()] == ["feature"]
    assert schema.get("id").role == ColumnRole.ID

