"""Arrow columns → the parameter sets a PEP 249 ``executemany`` binds.

This is the write path's mirror of `_arrow`, and it draws the columnar/row-shaped boundary
in the same place and for the same reason. A DB-API cursor is row-shaped: there is no way
to hand a driver a column. So the conversion happens **once per chunk**, column-wise —
`Array.to_pylist()` per column, then one `zip` — rather than once per row, and everything
above this line stays columnar.

`Table.to_pylist()` would have been the one-liner and is deliberately not used: it builds a
`dict` per row with every key re-hashed, costing several times what the column-wise
transpose costs and materializing the whole table at once regardless of the chunk size.

## Chunking is a correctness knob, not only a throughput one

A driver builds the whole parameter list in memory before it sends anything, and several
protocols cap how many placeholders one statement may carry — PostgreSQL's wire protocol
allows 65,535 parameters per statement, so a 10-column insert overflows at 6,554 rows.
Chunking by `rows_per_statement` keeps every execution under those ceilings and bounds the
driver's peak memory to one chunk.

## Nulls in a key column

A null bound into ``WHERE k = ?`` matches nothing, because SQL equality with null is
unknown rather than true. An ``update``/``delete``/``upsert`` row whose key is null
therefore affects no rows — silently, on every database. `null_key_rows` counts them so the
sink can say so instead of leaving the user to discover it from a row count.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from batcher._internal.errors import BackendError
from batcher.io.formats.sql.dbapi._statements import Statement

__all__ = ["duplicate_key_rows", "latest_per_key", "null_key_rows", "parameter_chunks"]


def _column(table: pa.Table, name: str) -> pa.ChunkedArray:
    """The named column, or a `BackendError` naming what the frame actually holds."""
    index = table.schema.get_field_index(name)
    if index < 0:
        raise BackendError(
            f"column {name!r} is not in the data being written ({table.schema.names}). "
            "A SQL write binds the frame's columns by name."
        )
    return table.column(index)


def parameter_chunks(
    table: pa.Table, statement: Statement, *, rows_per_statement: int
) -> Iterator[list[Any]]:
    """Yield `statement`'s parameter sets, at most `rows_per_statement` per yield.

    Each yielded element is what `cursor.executemany` takes: a list of tuples for a
    positional `paramstyle`, or a list of dicts keyed by the statement's synthetic
    parameter names for a name-based one.

    Args:
        table: The rows to write.
        statement: The statement whose `columns` fix the per-row binding order.
        rows_per_statement: The maximum rows bound into one ``executemany`` call.

    Yields:
        One chunk of parameter sets, in table order.

    Raises:
        BackendError: If the statement names a column the table does not have, or
            `rows_per_statement` is not positive.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.io.formats.sql.dbapi._bind import parameter_chunks
            >>> from batcher.io.formats.sql.dbapi._statements import insert
            >>> t = pa.table({"id": [1, 2, 3], "amt": [1.5, 2.5, 3.5]})
            >>> stmt = insert("t", ("id", "amt"), dialect="sqlite", paramstyle="qmark")
            >>> list(parameter_chunks(t, stmt, rows_per_statement=2))
            [[(1, 1.5), (2, 2.5)], [(3, 3.5)]]
    """
    if rows_per_statement < 1:
        raise BackendError(f"rows_per_statement must be >= 1, got {rows_per_statement}")
    columns = [_column(table, name) for name in statement.columns]
    for offset in range(0, table.num_rows, rows_per_statement):
        length = min(rows_per_statement, table.num_rows - offset)
        values = [column.slice(offset, length).to_pylist() for column in columns]
        rows = zip(*values, strict=True) if values else ()
        if statement.positional:
            yield [tuple(row) for row in rows]
        else:
            names = statement.names
            yield [dict(zip(names, row, strict=True)) for row in rows]


def null_key_rows(table: pa.Table, key_columns: tuple[str, ...]) -> int:
    """How many rows hold a null in any of `key_columns`.

    Args:
        table: The rows about to be written.
        key_columns: The columns bound into a ``WHERE`` clause or a conflict target.

    Returns:
        The number of rows that cannot match a target row, because SQL equality against
        null is unknown rather than true.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.io.formats.sql.dbapi._bind import null_key_rows
            >>> null_key_rows(pa.table({"id": [1, None, 3]}), ("id",))
            1
    """
    if not key_columns or table.num_rows == 0:
        return 0
    mask: pa.ChunkedArray | None = None
    for name in key_columns:
        is_null = pc.is_null(_column(table, name))
        mask = is_null if mask is None else pc.or_(mask, is_null)
    return 0 if mask is None else int(pc.sum(pc.cast(mask, pa.int64())).as_py() or 0)


def duplicate_key_rows(table: pa.Table, key_columns: tuple[str, ...]) -> int:
    """How many rows repeat a key an earlier row of `table` already holds.

    A keyed DML write binds one statement per row, so a repeated key is resolved by
    statement order: the row that happens to run last wins. Frame order is not a property
    a distributed write fixes, so this is counted and reported rather than trusted.

    Args:
        table: The rows about to be written.
        key_columns: The columns identifying a row.

    Returns:
        ``num_rows - distinct keys``: zero when every key is unique.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.io.formats.sql.dbapi._bind import duplicate_key_rows
            >>> duplicate_key_rows(pa.table({"id": [1, 1, 2]}), ("id",))
            1
    """
    if not key_columns or table.num_rows < 2:
        return 0
    keys = pa.table({f"k{i}": _column(table, n) for i, n in enumerate(key_columns)})
    groups = keys.group_by(keys.schema.names).aggregate([]).num_rows
    return table.num_rows - groups


def latest_per_key(
    table: pa.Table, key_columns: tuple[str, ...], sequence_by: tuple[str, ...]
) -> pa.Table:
    """Keep, per key, the row with the greatest `sequence_by` value, in frame order.

    This is what makes a repeated key's winner a property of the data rather than of the
    order rows reached the statement: the change with the highest sequence is the one
    applied, however the frame was split. A null sequence value sorts below every non-null
    one, so it never beats a non-null change -- the rule `Dataset.distinct(keep="last",
    order_by=...)` applies on a single sequence column, which keeps the per-shard and the
    whole-frame dedupe in agreement there. A tie falls back to frame order.

    Args:
        table: The rows about to be written.
        key_columns: The columns identifying a row.
        sequence_by: The columns ordering two changes to one key, most significant first.

    Returns:
        `table` with one row per key, the survivors in their original relative order.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.io.formats.sql.dbapi._bind import latest_per_key
            >>> t = pa.table({"id": [1, 1, 2], "seq": [2, 1, 1], "v": ["new", "old", "x"]})
            >>> latest_per_key(t, ("id",), ("seq",)).to_pydict()
            {'id': [1, 2], 'seq': [2, 1], 'v': ['new', 'x']}
    """
    if not key_columns or not sequence_by or table.num_rows < 2:
        return table
    index = "__batcher_row"
    keyed = pa.table(
        {f"k{i}": _column(table, n) for i, n in enumerate(key_columns)}
        | {f"s{i}": _column(table, n) for i, n in enumerate(sequence_by)}
        | {index: pa.array(range(table.num_rows), type=pa.int64())}
    )
    ordered = keyed.sort_by(
        [(f"s{i}", "ascending") for i in range(len(sequence_by))] + [(index, "ascending")],
        null_placement="at_start",
    )
    # `last` in the order just imposed; single-threaded, so `last` means last-in-order.
    winners = pa.TableGroupBy(
        ordered, [f"k{i}" for i in range(len(key_columns))], use_threads=False
    ).aggregate([(index, "last")])
    survivors = winners.column(f"{index}_last")
    return table.take(pc.take(survivors, pc.sort_indices(survivors)))
