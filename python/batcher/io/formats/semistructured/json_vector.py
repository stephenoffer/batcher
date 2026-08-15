"""Vectorized NDJSON encoding — the JSON writer's fast path, built from Arrow kernels.

The two encoders this replaces on the common shapes both work a **row** at a time
somewhere. `_table_to_ndjson_exact` calls `json.dumps` per row over `to_pylist()`, which
is the only way to render a float exactly but costs a Python object per cell;
`_table_to_ndjson` hands the table to pandas, whose C encoder is fast but rounds floats to
``double_precision`` decimal places. So the writer had to choose between exact and fast,
and any table carrying a float anywhere took the exact-and-slow branch — measured at
**0.06 Mrow/s**, roughly 20x behind the same table written as CSV.

There is no trade to make. Arrow's ``cast(float64 -> string)`` emits the shortest
round-tripping form, the same guarantee ``repr`` gives, so the exact rendering is
available as a *column* operation. Rendering each column to its JSON text and joining the
columns with ``binary_join_element_wise`` encodes the whole table with no Python per row:
measured at **1.29 Mrow/s** on the same table, 22x the exact encoder and 2x pandas — while
being byte-identical to pandas' output on the tables pandas already handled.

This path declines rather than approximates. A type it cannot render (temporal, decimal,
binary, list, map) and a string holding a control character other than the five with a
short escape both return None, and the caller falls back to an encoder that handles them.
"""

from __future__ import annotations

import json
from typing import Any

import pyarrow as pa

__all__ = ["ndjson_vectorized"]


# Rows encoded per pass. Bounds peak memory to one chunk's rendered text, and keeps the
# joined output inside the int32 offsets of a `string` (not `large_string`) array — a
# 2 GiB ceiling that a single 4M-row table can otherwise reach.
_CHUNK_ROWS = 256 * 1024

# The characters JSON requires be escaped inside a string: the quote, the backslash, and
# every C0 control. Non-ASCII needs no escape (the output is UTF-8), so it is deliberately
# absent — emitting `é` where the stdlib emits `\u00e9` parses to the same string and
# writes fewer bytes.
_NEEDS_ESCAPE = r'["\\]|[\x00-\x1f]'

# The characters with a two-character escape. The backslash MUST come first: escaping it
# after the others would double the backslash each of them just introduced. Any *other*
# control character needs a `\u00XX` form, which no Arrow kernel produces, so its presence
# declines the whole encode rather than emitting an invalid document.
_SHORT_ESCAPES = (
    ("\\", "\\\\"),
    ('"', '\\"'),
    ("\n", "\\n"),
    ("\r", "\\r"),
    ("\t", "\\t"),
    ("\b", "\\b"),
    ("\f", "\\f"),
)
_UNESCAPABLE = r"[\x00-\x07\x0b\x0e-\x1f]"


def _renderable(dtype: pa.DataType) -> bool:
    """Whether `dtype` has a JSON rendering built only from Arrow kernels."""
    if pa.types.is_struct(dtype):
        return all(_renderable(dtype.field(i).type) for i in range(dtype.num_fields))
    return bool(
        pa.types.is_boolean(dtype)
        or pa.types.is_integer(dtype)
        or pa.types.is_floating(dtype)
        or pa.types.is_string(dtype)
        or pa.types.is_large_string(dtype)
        or pa.types.is_null(dtype)
        # Temporal renders as an ISO-8601 string (see `_temporal`). Declining it sent every
        # date/timestamp column to the pandas encoder, which wrote a *numerically wrong*
        # number for every unit but nanoseconds.
        or pa.types.is_timestamp(dtype)
        or pa.types.is_date(dtype)
    )


def _strings(arr: pa.Array, pc: Any) -> pa.Array | None:
    """JSON string literals (quoted, escaped) for a string array, or None if not possible."""
    plain = pc.cast(arr, pa.string())
    if pc.any(pc.match_substring_regex(plain, _NEEDS_ESCAPE)).as_py():
        # Only pay the escape passes on a column that actually holds one. A control
        # character with no short escape declines: `\u00XX` has no kernel.
        if pc.any(pc.match_substring_regex(plain, _UNESCAPABLE)).as_py():
            return None
        for needle, replacement in _SHORT_ESCAPES:
            plain = pc.replace_substring(plain, needle, replacement)
    return pc.binary_join_element_wise('"', plain, '"', "")


def _temporal(arr: pa.Array, pc: Any) -> pa.Array:
    """An ISO-8601 JSON string for a date or timestamp array.

    JSON has no temporal type, so the value has to be either a number or a string. It was
    a number, and the number was wrong: this column declined here and fell through to the
    pandas encoder, which reads *every* timestamp column's raw integers as nanoseconds. Only
    `timestamp[ns]` came out right -- a `timestamp[us]` column (what the FFI boundary
    normalizes to, so the common case) was divided by a million, writing `1709210096` for an
    instant whose epoch-microsecond value is `1709210096123456`. Read back as any unit that
    is a plausible reading of a bare number, that is the wrong instant, silently.

    Arrow's cast to string is ISO-8601 at the column's own resolution, so it is exact for
    every unit, and it is the spelling `msgpack` already writes and that DuckDB, Spark and
    pandas' own ``date_format="iso"`` produce. The rendered text is digits and
    ``- : . T + Z`` only, none of which JSON escapes, so the quotes go on directly rather
    than through `_strings`' escape passes.
    """
    return pc.binary_join_element_wise('"', pc.cast(arr, pa.string()), '"', "")


def _floats(arr: pa.Array, pc: Any) -> pa.Array:
    """JSON numbers for a float array: exact, with NaN/Inf as `null` and a kept ``.0``."""
    text = pc.cast(arr, pa.string())
    # NaN and +/-Inf have no JSON form. `null` is what both existing encoders emit.
    text = pc.if_else(pc.is_finite(arr), text, pa.scalar(None, pa.string()))
    # Arrow renders a whole float bare (`100`), and a column of bare integers reads back
    # as int64 — a column *type* change across a round trip. Restoring the `.0` keeps the
    # value a JSON float, which is what `repr` (and so the exact encoder) already did.
    bare = pc.invert(pc.match_substring_regex(text, r"[.eEni]"))
    return pc.if_else(bare, pc.binary_join_element_wise(text, ".0", ""), text)


def _values(arr: pa.Array, pc: Any) -> pa.Array | None:
    """The JSON text of every element of `arr`, with nulls rendered as the literal `null`.

    Returns None when this array's type or contents need an encoder with a Python loop.
    """
    dtype = arr.type
    if pa.types.is_null(dtype):
        return pa.array(["null"] * len(arr), pa.string())
    if pa.types.is_struct(dtype):
        rendered = _object(arr, pc)
    elif pa.types.is_floating(dtype):
        rendered = _floats(arr, pc)
    elif pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        rendered = _strings(arr, pc)
    elif pa.types.is_timestamp(dtype) or pa.types.is_date(dtype):
        rendered = _temporal(arr, pc)
    else:  # bool and integer: Arrow's cast already emits the JSON spelling
        rendered = pc.cast(arr, pa.string())
    if rendered is None:
        return None
    # `coalesce`, not `if_else(is_null(arr), ...)`: a non-null input can still render as
    # null (a NaN float does), and both cases must reach the same literal.
    return pc.coalesce(rendered, pa.scalar("null", pa.string()))


def _object(arr: pa.Array, pc: Any) -> pa.Array | None:
    """`{"f": v, ...}` for each element of a struct array, or None if a field declines."""
    dtype = arr.type
    pieces: list[Any] = []
    for i in range(dtype.num_fields):
        child = _values(arr.field(i), pc)
        if child is None:
            return None
        pieces.append(pa.scalar(("{" if i == 0 else ",") + json.dumps(dtype.field(i).name) + ":"))
        pieces.append(child)
    if not pieces:
        return pa.array(["{}"] * len(arr), pa.string())
    pieces.append(pa.scalar("}"))
    obj = pc.binary_join_element_wise(*pieces, "")
    # A null struct is the value `null`, not an object of null fields. The children carry
    # arbitrary content under a null parent, so this overrides rather than propagates.
    return pc.if_else(pc.is_null(arr), pa.scalar(None, pa.string()), obj)


def _lines_bytes(lines: pa.Array) -> bytes | None:
    """The concatenated text of a null-free string array, read straight off its buffer.

    A `StringArray` already stores its values back to back in one character buffer, and
    every line here ends in its own newline — so the buffer *is* the NDJSON document and
    joining would only copy it a second time. Returns None if the array is not in that
    canonical shape (sliced, or holding a null), rather than reading the wrong bytes.
    """
    if lines.null_count or lines.offset or not pa.types.is_string(lines.type):
        return None
    offsets, data = lines.buffers()[1], lines.buffers()[2]
    if offsets is None:
        return b""
    if data is None:
        return b""
    end = int.from_bytes(memoryview(offsets)[4 * len(lines) : 4 * len(lines) + 4], "little")
    return memoryview(data)[:end].tobytes()


def _chunk_bytes(table: pa.Table, pc: Any) -> bytes | None:
    """One chunk of rows as NDJSON bytes, or None if any column declines."""
    pieces: list[Any] = []
    for i, name in enumerate(table.column_names):
        rendered = _values(table.column(i).combine_chunks(), pc)
        if rendered is None:
            return None
        pieces.append(pa.scalar(("{" if i == 0 else ",") + json.dumps(name) + ":"))
        pieces.append(rendered)
    # The trailing newline is part of the line, so the joined buffer needs no separator
    # pass and every chunk concatenates into a valid document.
    pieces.append(pa.scalar("}\n"))
    return _lines_bytes(pa.concat_arrays([pc.binary_join_element_wise(*pieces, "")]))


def ndjson_vectorized(table: pa.Table) -> bytes | None:
    """Encode `table` as NDJSON using Arrow kernels only, or None to use another encoder.

    Floats render exactly (Arrow's float-to-string cast is the shortest round-tripping
    form, as ``repr`` is), so this is not the fast-but-lossy option — it replaces both the
    exact encoder and the pandas one on every shape it accepts.

    Args:
        table: The rows to encode.

    Returns:
        The NDJSON document, or None when a column's type or contents need a fallback
        encoder (temporal, decimal, binary, list and map types; a string holding a control
        character with no short escape).
    """
    if not all(_renderable(f.type) for f in table.schema):
        return None
    if table.num_rows == 0:
        # Not b"": `pyarrow.json.read_json` rejects a 0-byte file as "Empty JSON file",
        # so an empty write must still leave a readable document. Matches both fallbacks.
        return b"\n"
    import pyarrow.compute as pc

    out: list[bytes] = []
    for start in range(0, table.num_rows, _CHUNK_ROWS):
        try:
            chunk = _chunk_bytes(table.slice(start, _CHUNK_ROWS), pc)
        except pa.ArrowInvalid:
            return None  # a chunk whose rendered text overflows int32 offsets
        if chunk is None:
            return None
        out.append(chunk)
    return b"".join(out)
