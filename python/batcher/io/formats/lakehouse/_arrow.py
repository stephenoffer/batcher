"""Normalize the Arrow types a lakehouse client hands back into the ones the engine speaks.

The engine's columnar contract is plain Arrow: `string`, `binary`, and the fixed-width
numerics. The table-format clients do not all produce that. delta-rs ≥ 1.x returns
`string_view`/`binary_view`; pyiceberg maps Iceberg's `StringType` to Arrow **`large_string`**
(a 64-bit offset variant). Both are legitimate Arrow, and neither is what the engine's
kernels are built for.

Left alone, the mismatch does not degrade — it *crashes*. A filter on a string column of an
ordinary Iceberg table raised ``Invalid comparison operation: LargeUtf8 == Utf8`` from the
Rust engine, because the column arrived as `large_string` and the literal as `string`. And
since every Iceberg table written by Spark or Flink carries pyiceberg's type mapping, that is
not an edge case: it is the normal case.

Normalizing here, at the connector boundary, is the same choice the FFI layer already makes
for narrow numerics (Int8/16/32 → Int64). The cast is zero-copy where Arrow can manage it and
cheap where it cannot, and it happens once per batch rather than once per comparison.
"""

from __future__ import annotations

import pyarrow as pa

__all__ = ["normalize_engine_types"]

#: The variant types a lakehouse client may return, and the plain type the engine speaks.
_NORMALIZE = (
    (pa.types.is_large_string, pa.string()),
    (pa.types.is_large_binary, pa.binary()),
    (pa.types.is_string_view, pa.string()),
    (pa.types.is_binary_view, pa.binary()),
)


def engine_schema(schema: pa.Schema) -> pa.Schema:
    """`schema` with every variant string/binary type replaced by the plain one."""
    fields = []
    for field in schema:
        fields.append(pa.field(field.name, _plain(field.type), nullable=field.nullable))
    return pa.schema(fields)


def _plain(dtype: pa.DataType) -> pa.DataType:
    for matches, replacement in _NORMALIZE:
        if matches(dtype):
            return replacement
    return dtype


def normalize_engine_types(table: pa.Table) -> pa.Table:
    """Cast `table`'s variant string/binary columns to the plain types the engine uses.

    A no-op — and no copy — when the table already speaks them, which is the common case.

    Args:
        table: A table as a lakehouse client produced it.

    Returns:
        The same table, typed for the engine.
    """
    target = engine_schema(table.schema)
    return table if target.equals(table.schema) else table.cast(target)
