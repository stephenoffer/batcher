"""Static per-column byte widths derived from a column's Arrow type.

A column's width is a property of its dtype, known before any row is read: an
`int64` is 8 bytes, a `date32` is 4. Only the variable-length types (string,
binary, list, struct) genuinely need a measurement, and for those this module
returns a documented prior that a *learned* width overrides as soon as one exists.

This is the cold-start floor under the cost model's byte axes (broadcast
eligibility, memory, IO). Before it, an unmeasured relation was costed at a flat
`bytes_per_row` constant regardless of its schema — so a two-`int64`-column join
key (16 B/row) and a 20-column payload were both estimated at 64 B/row. That
single constant decided broadcast eligibility, and it mis-sized narrow relations
by ~4x, forfeiting the broadcast join that a small build side should always get.

Neutral layer: imports only `pyarrow`.
"""

from __future__ import annotations

import pyarrow as pa

__all__ = ["DEFAULT_VARLEN_BYTES", "column_bytes", "schema_row_bytes"]

# Prior for a variable-length column (string/binary/list/struct) with no measured
# width. Deliberately generous — over-estimating a payload column's width is the
# safe direction for the byte axes it feeds (it can only make the cost model more
# conservative about broadcasting a wide relation). A learned `avg_byte_width`
# replaces it the first time the column is actually measured.
DEFAULT_VARLEN_BYTES = 32.0

# Arrow's offset buffers cost 4 or 8 bytes per row on top of the value bytes.
_OFFSET_BYTES = 4.0


def column_bytes(dtype: pa.DataType, default_varlen: float = DEFAULT_VARLEN_BYTES) -> float:
    """Estimated bytes per row for a column of `dtype`.

    Fixed-width types are exact (the Arrow bit width). Variable-length types return
    `default_varlen` plus their offset-buffer cost, since their true width needs a
    measurement.

    Args:
        dtype: The column's Arrow type.
        default_varlen: Bytes to assume for a variable-length value.

    Returns:
        The estimated per-row width of the column, in bytes.
    """
    if pa.types.is_boolean(dtype):
        return 1.0
    if pa.types.is_dictionary(dtype):
        # The values live in a shared dictionary; a row costs only its index.
        return float(max(dtype.index_type.bit_width // 8, 1))
    if pa.types.is_string(dtype) or pa.types.is_binary(dtype):
        return default_varlen + _OFFSET_BYTES
    if pa.types.is_large_string(dtype) or pa.types.is_large_binary(dtype):
        return default_varlen + 2 * _OFFSET_BYTES
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return default_varlen + _OFFSET_BYTES
    if pa.types.is_struct(dtype):
        return sum(column_bytes(f.type, default_varlen) for f in dtype)
    try:
        bit_width = dtype.bit_width
    except (ValueError, NotImplementedError, AttributeError):
        # A type with no fixed bit width and no branch above (e.g. a nested union).
        return default_varlen
    return float(max(bit_width // 8, 1))


def schema_row_bytes(schema: pa.Schema, default_varlen: float = DEFAULT_VARLEN_BYTES) -> float:
    """Estimated bytes per row for a whole schema — the sum of its columns' widths.

    Args:
        schema: The Arrow schema to size.
        default_varlen: Bytes to assume for a variable-length value.

    Returns:
        The estimated per-row width of one row of `schema`, in bytes.
    """
    return sum(column_bytes(f.type, default_varlen) for f in schema)
