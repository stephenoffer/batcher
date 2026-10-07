"""The PEP 249 cursor: run one statement through a `Session`, then stream its rows.

`execute` hands the statement and its parameters to `Session.sql`, which binds each value
into the parsed query as a typed literal. Nothing is spliced into the SQL text. The result
is the lazy `Dataset` that call returns, and the cursor does no work until the first fetch.
Rows are then produced one Arrow batch at a time from `Dataset.iter_batches`, so
`fetchmany` over a large result holds one batch, not the whole relation, and `fetchall` is
the only call that gathers everything.

Python tuples are what PEP 249 returns, so a fetch converts a batch into them. That is the
client boundary, the same place `Dataset.to_pylist` converts. A caller that wants the
columns stays in Arrow with `fetch_arrow_table`.

**Statements without a result set.** ``CREATE``, ``DROP``, ``INSERT``, ``UPDATE``,
``DELETE``, ``MERGE``, ``ALTER``, ``USE`` and ``SET`` leave `description` as None, as PEP
249 requires. `Session.sql` returns the target's new state for those, and the cursor does
not expose it as rows. A DML statement with ``RETURNING`` does produce rows. `rowcount` is
-1 for every statement, because a session DML statement is a plan rewrite that has not run
yet. The count it would report does not exist when `execute` returns.

This is the `dbapi` layer.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from batcher.dbapi.errors import InterfaceError, ProgrammingError, translate

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset
    from batcher.dbapi.connection import Connection

__all__ = ["Cursor"]

#: Leading keywords of a statement that changes the session or its catalog.
_WRITES = frozenset(
    {"CREATE", "DROP", "INSERT", "UPDATE", "DELETE", "MERGE", "ALTER", "TRUNCATE", "COPY"}
)
#: Leading keywords of a statement with no result set.
_NO_ROWS = _WRITES | {"USE", "SET"}
_DML = frozenset({"INSERT", "UPDATE", "DELETE", "MERGE"})

#: One entry of `Cursor.description`: name, type code, and five fields Batcher leaves None.
Column = tuple[str, pa.DataType, None, None, None, None, None]


def classify(operation: str) -> tuple[bool, bool]:
    """Whether `operation` produces a result set, and whether it writes.

    Read from the statement's leading keyword, which every SQL dialect places first, plus
    a ``RETURNING`` clause for DML. Only called after the session accepted the statement,
    so a tokenizer failure here cannot hide a syntax error.

    Args:
        operation: One SQL statement.

    Returns:
        ``(returns_rows, writes)``.
    """
    import sqlglot

    try:
        words = [token.text.upper() for token in sqlglot.tokenize(operation)]
    except Exception:  # an exotic dialect's token; the session already parsed it
        return True, False
    if not words:
        return False, False
    first = words[0]
    returning = first in _DML and "RETURNING" in words
    return (first not in _NO_ROWS or returning), first in _WRITES


class Cursor:
    """A PEP 249 cursor over a Batcher `Session`, created by `Connection.cursor`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher import dbapi
            >>> s = bt.Session()
            >>> _ = s.register("t", bt.from_pydict({"v": [1, 2, 3]}))
            >>> with dbapi.connect(s) as conn:
            ...     cur = conn.cursor()
            ...     _ = cur.execute("SELECT v FROM t WHERE v >= ? ORDER BY v", [2])
            ...     cur.fetchall()
            [(2,), (3,)]
    """

    #: How many rows `fetchmany` returns when called without a size (PEP 249's default).
    arraysize: int = 1

    def __init__(self, connection: Connection) -> None:
        """Create a cursor bound to `connection`; use `Connection.cursor` instead.

        Args:
            connection: The connection whose session runs this cursor's statements.
        """
        self._connection = connection
        self._closed = False
        self._description: list[Column] | None = None
        self._result: Dataset | None = None
        self._batches: Iterator[pa.RecordBatch] | None = None
        self._rows: deque[tuple[Any, ...]] = deque()

    # --- PEP 249 attributes ------------------------------------------------
    @property
    def connection(self) -> Connection:
        """The connection this cursor was created from.

        Returns:
            The `Connection`.

        Examples:
            .. doctest::

                >>> from batcher import dbapi
                >>> conn = dbapi.connect()
                >>> conn.cursor().connection is conn
                True
        """
        return self._connection

    @property
    def description(self) -> list[Column] | None:
        """One 7-tuple per result column, or None after a statement with no result set.

        Each tuple is ``(name, type_code, None, None, None, None, None)``. The type code is
        the column's pyarrow `DataType` and compares equal to the matching PEP 249 type
        object, such as `dbapi.NUMBER`. Batcher does not report display size, internal size,
        precision, scale or nullability here.

        Returns:
            The column descriptions, or None.

        Examples:
            .. doctest::

                >>> from batcher import dbapi
                >>> cur = dbapi.connect().cursor()
                >>> _ = cur.execute("SELECT 1 AS n, 'a' AS s")
                >>> [(d[0], d[1] == dbapi.NUMBER) for d in cur.description]
                [('n', True), ('s', False)]
        """
        return self._description

    @property
    def rowcount(self) -> int:
        """Always -1: Batcher does not know a statement's row count when `execute` returns.

        A query's rows are produced as they are fetched, and a DML statement on a session
        table is a plan rewrite that runs on a later read, so no count exists yet. PEP 249
        defines -1 as "not determinable", which is the truthful answer.

        Returns:
            -1.

        Examples:
            .. doctest::

                >>> from batcher import dbapi
                >>> cur = dbapi.connect().cursor()
                >>> cur.execute("SELECT 1").rowcount
                -1
        """
        return -1

    # --- execution -----------------------------------------------------------
    def execute(
        self, operation: str, parameters: Sequence[Any] | Mapping[str, Any] | None = None
    ) -> Cursor:
        """Run one SQL statement, binding `parameters` to its placeholders.

        A sequence fills ``?`` (the module's `paramstyle`, ``qmark``) or ``$1``-style
        placeholders; a mapping fills ``$name`` ones, and ``:name`` ones where the
        session's dialect reads them (DuckDB's reads ``name: expr`` in a select list as an
        alias, so write ``$name`` there). Each value is bound
        into the parsed statement as a typed literal and never spliced into the text.

        Args:
            operation: One SQL statement.
            parameters: Values for its placeholders, or None when it has none.

        Returns:
            This cursor, so ``cur.execute(...).fetchall()`` reads in one line.

        Raises:
            InterfaceError: The cursor or its connection is closed.
            ProgrammingError: A syntax error, an unknown table or column, or a bad binding.
            NotSupportedError: A construct Batcher does not translate.

        Examples:
            .. doctest::

                >>> from batcher import dbapi
                >>> cur = dbapi.connect().cursor()
                >>> cur.execute("SELECT $a + $b AS total", {"a": 1, "b": 2}).fetchone()
                (3,)
        """
        self._check_open()
        self._reset()
        if parameters is not None and (
            isinstance(parameters, (str, bytes)) or not isinstance(parameters, Sequence | Mapping)
        ):
            raise ProgrammingError(
                f"parameters must be a sequence or a mapping, got {type(parameters).__name__}"
            )
        session = self._connection.session
        try:
            result = session.sql(operation, params=parameters)
        except Exception as exc:
            raise translate(exc) from exc
        returns_rows, writes = classify(operation)
        if writes:
            self._connection._note_write()
        if returns_rows:
            self._result = result
            self._description = [
                (field.name, field.type, None, None, None, None, None) for field in result.schema
            ]
        return self

    def executemany(
        self, operation: str, seq_of_parameters: Sequence[Sequence[Any] | Mapping[str, Any]]
    ) -> Cursor:
        """Run `operation` once per parameter set, in order.

        Each run is a separate statement with its own effect, as with `execute`. Nothing
        groups them, so when one fails the ones before it stay applied. A result set is
        refused here, as PEP 249 leaves one undefined: use `execute` for a query.

        Args:
            operation: One SQL statement.
            seq_of_parameters: One parameter sequence or mapping per run.

        Returns:
            This cursor.

        Raises:
            ProgrammingError: `operation` produces a result set.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher import dbapi
                >>> s = bt.Session()
                >>> _ = s.sql("CREATE TABLE t AS SELECT 0 AS v")
                >>> cur = dbapi.connect(s).cursor()
                >>> _ = cur.executemany("INSERT INTO t VALUES (?)", [[1], [2]])
                >>> cur.execute("SELECT SUM(v) AS s FROM t").fetchone()
                (3,)
        """
        self._check_open()
        if classify(operation)[0]:
            raise ProgrammingError(
                "executemany() runs statements without a result set; use execute() for a query"
            )
        for parameters in seq_of_parameters:
            self.execute(operation, parameters)
        return self

    # --- fetching ------------------------------------------------------------
    def fetchone(self) -> tuple[Any, ...] | None:
        """The next row, or None when the result is exhausted.

        Returns:
            One row as a tuple, or None.

        Raises:
            ProgrammingError: The last statement produced no result set.

        Examples:
            .. doctest::

                >>> from batcher import dbapi
                >>> cur = dbapi.connect().cursor()
                >>> _ = cur.execute("SELECT 7 AS n")
                >>> cur.fetchone(), cur.fetchone()
                ((7,), None)
        """
        rows = self.fetchmany(1)
        return rows[0] if rows else None

    def fetchmany(self, size: int | None = None) -> list[tuple[Any, ...]]:
        """Up to `size` more rows; fewer, or none, once the result runs out.

        Args:
            size: How many rows to return; `arraysize` when omitted.

        Returns:
            The rows, as tuples.

        Raises:
            ProgrammingError: The last statement produced no result set.

        Examples:
            .. doctest::

                >>> from batcher import dbapi
                >>> cur = dbapi.connect().cursor()
                >>> _ = cur.execute("SELECT * FROM range(5) t(i) ORDER BY i")
                >>> cur.fetchmany(2), cur.fetchmany(4)
                ([(0,), (1,)], [(2,), (3,), (4,)])
        """
        want = self.arraysize if size is None else size
        if want < 0:
            raise ProgrammingError(f"fetchmany() size must be non-negative, got {want}")
        self._check_result()
        while len(self._rows) < want and self._pull():
            pass
        return [self._rows.popleft() for _ in range(min(want, len(self._rows)))]

    def fetchall(self) -> list[tuple[Any, ...]]:
        """Every remaining row.

        Returns:
            The rows, as tuples.

        Raises:
            ProgrammingError: The last statement produced no result set.

        Examples:
            .. doctest::

                >>> from batcher import dbapi
                >>> cur = dbapi.connect().cursor()
                >>> cur.execute("SELECT 1 AS a UNION ALL SELECT 2 ORDER BY a").fetchall()
                [(1,), (2,)]
        """
        self._check_result()
        while self._pull():
            pass
        rows = list(self._rows)
        self._rows.clear()
        return rows

    def fetch_arrow_table(self) -> pa.Table:
        """Every remaining row as one pyarrow `Table`, without converting to Python values.

        Not part of PEP 249. DuckDB and ADBC cursors offer the same call, and it is the way
        to read a large result through this adapter without building a tuple per row.

        Returns:
            The remaining rows, with the result's schema.

        Raises:
            ProgrammingError: The last statement produced no result set.

        Examples:
            .. doctest::

                >>> from batcher import dbapi
                >>> cur = dbapi.connect().cursor()
                >>> cur.execute("SELECT 1 AS a").fetch_arrow_table().to_pydict()
                {'a': [1]}
        """
        result = self._check_result()
        if self._batches is None and not self._rows:
            self._batches = iter(())
            try:
                return result.to_arrow()
            except Exception as exc:
                raise translate(exc) from exc
        head = pa.Table.from_pylist(
            [dict(zip(result.schema.names, row, strict=True)) for row in self._rows],
            schema=result.schema,
        )
        self._rows.clear()
        tail: list[pa.RecordBatch] = []
        while self._batches is not None:
            batch = self._next_batch()
            if batch is None:
                break
            tail.append(batch)
        return pa.concat_tables([head, pa.Table.from_batches(tail, schema=result.schema)])

    # --- PEP 249 no-ops, iteration, lifecycle --------------------------------
    def setinputsizes(self, sizes: Any) -> None:
        """Accept and ignore input-size hints, as PEP 249 permits.

        Args:
            sizes: Ignored.

        Examples:
            .. doctest::

                >>> from batcher import dbapi
                >>> dbapi.connect().cursor().setinputsizes([None])
        """

    def setoutputsize(self, size: Any, column: Any = None) -> None:
        """Accept and ignore output-size hints, as PEP 249 permits.

        Args:
            size: Ignored.
            column: Ignored.

        Examples:
            .. doctest::

                >>> from batcher import dbapi
                >>> dbapi.connect().cursor().setoutputsize(1024)
        """

    def close(self) -> None:
        """Close the cursor and release its result; any later call raises `InterfaceError`.

        Examples:
            .. doctest::

                >>> from batcher import dbapi
                >>> cur = dbapi.connect().cursor()
                >>> cur.close()
                >>> cur.closed
                True
        """
        self._reset()
        self._closed = True

    @property
    def closed(self) -> bool:
        """Whether `close` has run on this cursor or its connection.

        Returns:
            True once closed.

        Examples:
            .. doctest::

                >>> from batcher import dbapi
                >>> dbapi.connect().cursor().closed
                False
        """
        return self._closed

    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        """Iterate the remaining rows, one at a time.

        Yields:
            Each remaining row, as a tuple.

        Examples:
            .. doctest::

                >>> from batcher import dbapi
                >>> cur = dbapi.connect().cursor()
                >>> list(cur.execute("SELECT 1 AS a UNION ALL SELECT 2 ORDER BY a"))
                [(1,), (2,)]
        """
        while (row := self.fetchone()) is not None:
            yield row

    def __enter__(self) -> Cursor:
        """Return this cursor; the ``with`` block closes it on exit."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the cursor."""
        self.close()

    # --- internals -------------------------------------------------------------
    def _check_open(self) -> None:
        if self._closed:
            raise InterfaceError("the cursor is closed")
        self._connection._check_open()

    def _check_result(self) -> Dataset:
        self._check_open()
        if self._result is None:
            raise ProgrammingError("the last statement produced no result set to fetch")
        return self._result

    def _reset(self) -> None:
        """Drop the current result, closing its batch stream so the engine can stop."""
        close = getattr(self._batches, "close", None)
        if close is not None:
            close()
        self._batches = None
        self._result = None
        self._description = None
        self._rows.clear()

    def _next_batch(self) -> pa.RecordBatch | None:
        """The next batch of the result, opening the stream on first use; None at the end."""
        if self._batches is None:
            self._batches = iter(self._check_result().iter_batches())
        try:
            return next(self._batches)
        except StopIteration:
            self._batches = iter(())
            return None
        except Exception as exc:
            raise translate(exc) from exc

    def _pull(self) -> bool:
        """Convert one more batch into rows; False once the result is exhausted."""
        batch = self._next_batch()
        if batch is None:
            return False
        if batch.num_columns == 0:
            self._rows.extend(() for _ in range(batch.num_rows))
        else:
            self._rows.extend(zip(*(c.to_pylist() for c in batch.columns), strict=True))
        return True
