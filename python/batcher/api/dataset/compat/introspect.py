"""The "what am I holding?" surface: `info`, `glimpse`, `memory_usage`, `collect_schema`.

A REPL user's first question about an unfamiliar dataset is what the columns are and
roughly what is in them, which is why pandas ships `info()` and Polars ships
`glimpse()`. Both are answered here from the schema plus a small bounded preview, so
neither one scans the whole relation: `info` executes a `count` (usually served from
metadata) and `glimpse` reads a single head slice.

`memory_usage` reports an *estimate* derived from the Arrow types and the row count,
not a measurement. Arrow's real footprint depends on dictionary encoding, validity
buffers, and string lengths, none of which are knowable without reading the data, so
the number is labelled an estimate everywhere it surfaces rather than implying a
precision it does not have.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.api.dataset.frame import Dataset

__all__ = [
    "build_collect_schema",
    "build_glimpse",
    "build_info",
    "build_memory_usage",
]

# Fallback width, in bytes, for a variable-width Arrow type whose per-value size
# cannot be known without reading the data (string, binary, list, struct).
_VARIABLE_WIDTH_ESTIMATE = 16


def build_collect_schema(ds: Dataset) -> dict[str, pa.DataType]:
    """The output schema as an ordered `{column: arrow_type}` mapping."""
    schema = ds.schema
    return dict(zip(schema.names, schema.types, strict=True))


def _estimated_width(dtype: pa.DataType) -> int:
    """Bytes per value for `dtype`, estimated for variable-width types."""
    import pyarrow as pa

    if pa.types.is_boolean(dtype):
        return 1
    try:
        bit_width = dtype.bit_width
    except (ValueError, AttributeError):
        return _VARIABLE_WIDTH_ESTIMATE
    return max(1, bit_width // 8)


def build_memory_usage(ds: Dataset) -> dict[str, int]:
    """An estimated in-memory size in bytes per column (see the module docstring)."""
    rows = ds.count()
    return {
        name: rows * _estimated_width(dtype) for name, dtype in build_collect_schema(ds).items()
    }


def build_info(ds: Dataset) -> None:
    """Print a pandas-style summary: row count, and each column's type and null count."""
    schema = build_collect_schema(ds)
    rows = ds.count()
    nulls = ds.null_count().to_pydict() if schema else {}

    print(f"Dataset: {rows} rows x {len(schema)} columns")
    if not schema:
        return
    name_width = max(len(n) for n in schema)
    type_width = max(len(str(t)) for t in schema.values())
    header = f"{'column'.ljust(name_width)}  {'dtype'.ljust(type_width)}  non-null"
    print(header)
    print("-" * len(header))
    for name, dtype in schema.items():
        null_count = _scalar_null_count(nulls, name)
        non_null = "?" if null_count is None else str(rows - null_count)
        print(f"{name.ljust(name_width)}  {str(dtype).ljust(type_width)}  {non_null}")
    total = sum(build_memory_usage(ds).values())
    print(f"estimated size: {total} bytes")


def _scalar_null_count(nulls: dict[str, Any], name: str) -> int | None:
    """The null count `null_count()` reported for `name`, or `None` if it did not."""
    values = nulls.get(name)
    if isinstance(values, list) and len(values) == 1 and isinstance(values[0], int):
        return values[0]
    return None


def build_glimpse(ds: Dataset, max_items_per_column: int) -> None:
    """Print a Polars-style transposed preview: one line per column with sample values."""
    schema = build_collect_schema(ds)
    if not schema:
        print("Dataset: 0 columns")
        return
    preview = ds.limit(max_items_per_column).to_pydict()
    name_width = max(len(n) for n in schema)
    type_width = max(len(str(t)) for t in schema.values())
    print(f"Dataset: {len(schema)} columns")
    for name, dtype in schema.items():
        values = ", ".join(repr(v) for v in preview.get(name, []))
        print(f"$ {name.ljust(name_width)} <{str(dtype).ljust(type_width)}> {values}")
