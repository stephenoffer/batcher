"""CSV encoding with a null token, built from Arrow string kernels.

Arrow's CSV writer has no null representation: a null is an empty field, and every string
value is quoted. So the only way to write ``null_value="NULL"`` through it was to cast every
column to string and fill the nulls, which quoted every cell, the numbers and the token
included. That file did not read back. A quoted ``"NULL"`` is text rather than a null (the
reader keeps quoted values literal, so a country code ``"NA"`` survives), a numeric column
holding it could only be inferred as ``string``, and every column came back as text.

This encoder writes what DuckDB and pandas write instead: the token bare, numbers bare, and a
string quoted only when it has to be, meaning it is empty, holds the delimiter, a quote or a
line break, or equals the token itself. Each cell is rendered with the same cast the Arrow
writer uses, so a value's text is unchanged; only the quoting differs. It is all vectorized
Arrow kernels: rows are assembled with `binary_join_element_wise` and never touched in Python.
"""

from __future__ import annotations

import pyarrow as pa

from batcher._internal.errors import FormatError

__all__ = ["encode_with_null_token"]


def encode_with_null_token(
    table: pa.Table, *, delimiter: str, null_token: str, include_header: bool
) -> bytes:
    """`table` as CSV bytes, with nulls written as a bare `null_token`.

    Args:
        table: The rows to encode.
        delimiter: The field delimiter.
        null_token: The text a null is written as.
        include_header: Whether to start with a line of column names.

    Returns:
        The encoded CSV, one line per row, each ending in a newline.

    Raises:
        FormatError: If a column cannot be rendered as text (binary that is not UTF-8).
    """
    import pyarrow.compute as pc

    out = bytearray()
    if include_header:
        out += delimiter.join(_quote_name(n) for n in table.column_names).encode() + b"\n"
    if not table.num_rows or not table.num_columns:
        return bytes(out)
    cells = [
        _cells(table.column(i), name, delimiter, null_token)
        for i, name in enumerate(table.column_names)
    ]
    # `binary_join_element_wise` takes the separator last. Joining the row with an empty
    # string on "\n" appends the line ending without a Python pass over the rows.
    rows = pc.binary_join_element_wise(*cells, delimiter)
    lines = pc.binary_join_element_wise(rows, "", "\n")
    for chunk in lines.chunks if isinstance(lines, pa.ChunkedArray) else [lines]:
        out += _string_bytes(chunk)
    return bytes(out)


def _cells(column: pa.ChunkedArray, name: str, delimiter: str, token: str) -> pa.ChunkedArray:
    """One column's cells as CSV text, quoted where needed and nulls as `token`."""
    import pyarrow.compute as pc

    dtype = column.type
    if pa.types.is_dictionary(dtype):
        column = pc.cast(column, dtype.value_type)
        dtype = dtype.value_type
    try:
        text = pc.cast(column, pa.string())
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        raise FormatError(
            f"csv: column {name!r} ({dtype}) cannot be written as text with null_value=: "
            f"{exc}. Cast it to a string first, or write without null_value=."
        ) from exc
    if _is_texty(dtype):
        text = _quote_where_needed(text, delimiter, token)
    return pc.fill_null(text, token)


def _is_texty(dtype: pa.DataType) -> bool:
    return (
        pa.types.is_string(dtype)
        or pa.types.is_large_string(dtype)
        or pa.types.is_binary(dtype)
        or pa.types.is_large_binary(dtype)
    )


def _quote_where_needed(text: pa.ChunkedArray, delimiter: str, token: str) -> pa.ChunkedArray:
    """`text` with each value quoted when a bare rendering would read back differently.

    An empty value would read back as null, a delimiter, quote or line break would split or
    end the field, and a value equal to the token would read back as the null it spells.
    """
    import pyarrow.compute as pc

    needs = pc.or_(pc.equal(pc.utf8_length(text), 0), pc.equal(text, token))
    for special in {delimiter, '"', "\n", "\r"}:
        needs = pc.or_(needs, pc.match_substring(text, special))
    escaped = pc.replace_substring(text, '"', '""')
    quoted = pc.binary_join_element_wise('"', escaped, '"', "")
    return pc.if_else(needs, quoted, text)


def _quote_name(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _string_bytes(chunk: pa.Array) -> bytes:
    """The concatenated UTF-8 of every value in a null-free string `chunk`."""
    import numpy as np

    if not len(chunk):
        return b""
    width = np.int64 if pa.types.is_large_string(chunk.type) else np.int32
    offsets = np.frombuffer(chunk.buffers()[1], dtype=width)
    start = int(offsets[chunk.offset])
    stop = int(offsets[chunk.offset + len(chunk)])
    return chunk.buffers()[2].slice(start, stop - start).to_pybytes()
