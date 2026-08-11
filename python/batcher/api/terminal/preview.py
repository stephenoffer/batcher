"""Render a small result as a readable table for `Dataset.show`.

`show()` is the first thing anyone types in a REPL or a script, and it used to print
pyarrow's own `Table` repr — a *column-oriented* dump that lists each column's chunks on its
own line. That is a fine debugging view of an Arrow buffer and a poor view of data: to read
one row you scan several lines and count positions, and it degrades badly with column count.
Every neighbour prints rows (pandas, Polars, Spark, DuckDB, every SQL client), and the reason
is that a person reading a preview is reading rows.

So this renders rows, in ASCII, in the MySQL/Spark shape. ASCII rather than box-drawing
characters because a preview must be readable in whatever terminal, pipe, log file, or CI
transcript it lands in, and a mojibake table is worse than a plain one.

The whole module works on an already-materialized preview — `show()` pushes its `limit` into
the *plan*, so what arrives here is at most a screenful of rows. Touching those values in
Python is not a hot-path violation; it is the one place the values exist to be looked at.
"""

from __future__ import annotations

import pyarrow as pa

__all__ = ["render"]

#: Longest cell rendered in full. Past this the value is cut and marked, because one long
#: JSON blob or base64 string otherwise sets the width of every row on screen.
_MAX_CELL = 32
#: Total table width to aim for. Columns past it are dropped and counted, rather than
#: wrapping — a wrapped table is unreadable in exactly the terminal that made it wrap.
_MAX_WIDTH = 120
_ELLIPSIS = "..."


def render(table: pa.Table, *, limit: int) -> str:
    """The preview text for `table`, header and all.

    Args:
        table: The already-limited result to show.
        limit: The row cap `show()` was called with, so the footer can say whether the
            preview is the whole result or the front of it.

    Returns:
        The rendered table as a single string, without a trailing newline.
    """
    if table.num_columns == 0:
        return f"(no columns, {table.num_rows} rows)"
    names = list(table.column_names)
    types = [_type_name(table.schema.field(n).type) for n in names]
    rows = [[_cell(v) for v in row] for row in _row_values(table)]
    kept, dropped = _fit(names, types, rows)
    widths = _widths(kept, names, types, rows)
    lines = [_rule(widths), _row(names, kept, widths), _row(types, kept, widths), _rule(widths)]
    if rows:
        lines.extend(_row(r, kept, widths) for r in rows)
        lines.append(_rule(widths))
    lines.append(_footer(table, limit, dropped))
    return "\n".join(lines)


def _row_values(table: pa.Table) -> list[list[object]]:
    """`table` as a list of row value lists, in column order."""
    columns = [column.to_pylist() for column in table.columns]
    return [[column[i] for column in columns] for i in range(table.num_rows)]


def _type_name(dtype: pa.DataType) -> str:
    """A short name for `dtype` — the family, not the full parameterisation.

    A `struct<data: binary, shape: list<item: int32>, dtype: string>` is 56 characters of
    header that tells a reader less than `struct` does, and sets the column's width for
    every row beneath it.
    """
    text = str(dtype)
    for prefix in ("struct", "list", "large_list", "fixed_size_list", "map", "extension"):
        if text.startswith(prefix):
            return prefix
    return text if len(text) <= _MAX_CELL else text[: _MAX_CELL - len(_ELLIPSIS)] + _ELLIPSIS


def _cell(value: object) -> str:
    """One value as display text.

    `None` prints as ``null`` rather than as Python's ``None``: the column is nullable in
    Arrow terms and a reader comparing this against a SQL client should see the same word.
    """
    if value is None:
        return "null"
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, bool):
        return "true" if value else "false"
    tensor = _tensor_cell(value)
    if tensor is not None:
        return tensor
    text = str(value)
    if len(text) > _MAX_CELL:
        return text[: _MAX_CELL - len(_ELLIPSIS)] + _ELLIPSIS
    return text


def _tensor_cell(value: object) -> str | None:
    """``<uint8 2x2>`` for a variable-shape tensor row, else `None`.

    Recognized structurally, the same way `io.formats.ml.ragged.is_ragged_tensor_column`
    recognizes the column: a dict of exactly ``data``/``shape``/``dtype``. Rendered rather
    than dumped because the raw form is a base64-looking buffer that fills the row and tells
    the reader nothing — and this is a column type the multimodal path now produces routinely.
    """
    if not isinstance(value, dict) or set(value) != {"data", "shape", "dtype"}:
        return None
    shape = value.get("shape")
    if not isinstance(shape, list):
        return None
    return f"<{_dtype_name(value.get('dtype'))} {'x'.join(str(d) for d in shape)}>"


def _dtype_name(code: object) -> str:
    """``"uint8"`` for the stored ``"|u1"``.

    The column stores NumPy's `dtype.str` because it round-trips byte order exactly; nobody
    wants to read it. Imported lazily and only here, so a preview of ordinary columns never
    touches NumPy.
    """
    try:
        import numpy as np

        return np.dtype(str(code)).name
    except Exception:
        return str(code)


def _widths(kept: int, names: list[str], types: list[str], rows: list[list[str]]) -> list[int]:
    """Column widths over the header, the type row, and every previewed cell."""
    return [
        max(len(names[i]), len(types[i]), *(len(r[i]) for r in rows), 1)
        if rows
        else max(len(names[i]), len(types[i]))
        for i in range(kept)
    ]


def _fit(names: list[str], types: list[str], rows: list[list[str]]) -> tuple[int, int]:
    """How many leading columns fit in `_MAX_WIDTH`, and how many are dropped.

    At least one column is always kept, so a single very wide column still prints something
    rather than an empty frame.
    """
    total = 1
    for index in range(len(names)):
        cells = [len(names[index]), len(types[index]), *(len(r[index]) for r in rows)]
        total += max(cells) + 3
        if total > _MAX_WIDTH and index > 0:
            return index, len(names) - index
    return len(names), 0


def _rule(widths: list[int]) -> str:
    return "+" + "+".join("-" * (w + 2) for w in widths) + "+"


def _row(cells: list[str], kept: int, widths: list[int]) -> str:
    return "| " + " | ".join(cells[i].ljust(widths[i]) for i in range(kept)) + " |"


def _footer(table: pa.Table, limit: int, dropped: int) -> str:
    """The line under the table: what was shown, and what was left out.

    It says "first N rows" only when the preview actually filled its limit, because a
    preview that did not is the whole result and saying otherwise invites a second look for
    rows that are not there. The total row count is deliberately not printed: `show()` pushes
    its limit into the plan precisely so a billion-row source is never counted to preview ten
    rows of it.
    """
    rows = table.num_rows
    shown = _count(rows, "row")
    if rows >= limit:
        shown = f"first {shown}"
    columns = _count(table.num_columns, "column")
    if dropped:
        columns += f" ({dropped} not shown)"
    return f"[{shown} x {columns}]"


def _count(n: int, noun: str) -> str:
    """``"1 row"`` / ``"3 rows"`` — a footer that says "1 rows" reads as a bug in the tool."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"
