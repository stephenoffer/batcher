"""Normalize the Arrow types a lakehouse client hands back into the ones the engine speaks.

The engine's columnar contract is plain Arrow: `string`, `binary`, the small-offset `list`,
and the fixed-width numerics. The table-format clients do not all produce that. delta-rs ≥ 1.x
returns `string_view`/`binary_view`; pyiceberg maps Iceberg's `StringType` to Arrow
**`large_string`** (a 64-bit offset variant) and every Iceberg `ListType` to **`large_list`**.
All are legitimate Arrow, and none is what the engine's kernels are built for.

Left alone, the mismatch does not degrade — it *crashes*. A filter on a string column of an
ordinary Iceberg table raised ``Invalid comparison operation: LargeUtf8 == Utf8`` from the
Rust engine, because the column arrived as `large_string` and the literal as `string`. And
since every Iceberg table written by Spark or Flink carries pyiceberg's type mapping, that is
not an edge case: it is the normal case.

The variant types **nest**, and so must the normalization. pyiceberg hands back a struct's
string field as `large_string`, a `list<string>` as `large_list<large_string>`, a map's
values likewise — so a top-level-only rewrite left every *nested* string/binary/list still in
its 64-bit form, and the same crash reappeared the moment a query touched it
(``s.field("name") == "x"`` → ``LargeUtf8 == Utf8``; ``tags.list.contains("a")`` →
``expected a Utf8 argument, got LargeList(... LargeUtf8)``). So `_plain` recurses through
`list`/`large_list`/`fixed_size_list`, `struct`, and `map`, normalizing each child — and
folds `large_list` down to the small-offset `list` the engine's kernels expect.

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
    """`schema` with every variant string/binary/list type replaced by the plain one."""
    return pa.schema([_plain_field(field) for field in schema])


def _plain_field(field: pa.Field) -> pa.Field:
    """`field` with its type (and any nested child types) normalized for the engine."""
    return pa.field(field.name, _plain(field.type), nullable=field.nullable)


def _plain(dtype: pa.DataType) -> pa.DataType:
    for matches, replacement in _NORMALIZE:
        if matches(dtype):
            return replacement
    # Variant types nest: a lakehouse client returns a struct's string field, a list's
    # element, or a map's value in its 64-bit form too, and the engine's kernels reject it
    # exactly as they reject a top-level one. Normalize each child, and fold `large_list`
    # down to the small-offset `list` the engine speaks.
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return pa.list_(_plain_field(dtype.field(0)))
    if pa.types.is_struct(dtype):
        return pa.struct([_plain_field(dtype.field(i)) for i in range(dtype.num_fields)])
    if pa.types.is_map(dtype):
        return pa.map_(_plain(dtype.key_type), _plain(dtype.item_type))
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
