"""Run an Ibis expression on Batcher, through Ibis's public SQL compiler and `Session.sql`.

Ibis compiles an expression to SQL with ``ibis.to_sql(expr, dialect=...)``, a documented,
public function that needs no backend. Batcher's SQL front end reads DuckDB's dialect, so
`to_dataset` compiles to ``dialect="duckdb"`` and hands the text to `Session.sql`. The tables
the expression names are resolved by name in the session, and the result is a lazy `Dataset`:
`Dataset.to_arrow`, `iter_batches` or a write runs it, and nothing passes through pandas.

`table` goes the other way. It describes a session table to Ibis as an *unbound* table,
``ibis.table(schema, name=...)``, with the schema converted by ``ibis.Schema.from_pyarrow``,
so an expression can be written against it without a connection.

**Why not a registered Ibis backend.** An ``ibis.backends`` entry point needs a subclass of
Ibis's backend base classes, which are internal and change between Ibis releases. This
bridge uses only the three documented calls above, so it does not break when those internals
move. The cost is that ``ibis.batcher.connect()`` does not exist; you call `to_dataset`.

**The subset.** Whatever SQL Ibis emits for an expression must be SQL Batcher translates.
Projections, filters, computed columns, ``group_by``/``aggregate`` with the common
reductions, ``order_by``, ``limit``, joins, ``distinct`` and unions are the documented
subset. A construct outside it raises `SQLUnsupportedError` naming the construct.

Not yet verified against a live Ibis installation; see tests/PENDING_VERIFICATION.md.

This is the `integrations` layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher._internal.optional import require

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset
    from batcher.api.sql_session import Session

__all__ = ["table", "to_dataset"]


def _ibis() -> Any:
    return require(
        "ibis", feature="The Batcher Ibis bridge", provides="ibis-framework", extra="ibis"
    )


def _session(session: Session | None) -> Session:
    from batcher.api.session.sql import current_session
    from batcher.api.sql_session import Session

    if session is None:
        return current_session()
    if not isinstance(session, Session):
        raise PlanError(f"expected a bt.Session, got {type(session).__name__}")
    return session


def table(name: str, session: Session | None = None) -> Any:
    """An unbound Ibis table with the name and schema of the session table `name`.

    Not yet verified against a live Ibis installation; see tests/PENDING_VERIFICATION.md.

    Args:
        name: A table or view registered in the session.
        session: The session to read it from. None uses `bt.current_session()`.

    Returns:
        An ``ibis.Table`` expression, unbound to any backend.

    Raises:
        PlanError: `name` is not a table in the session.
        MissingDependencyError: Ibis is not installed (the ``ibis`` extra).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.integrations import ibis as bt_ibis
            >>> s = bt.Session()
            >>> _ = s.register("orders", bt.from_pydict({"id": [1, 2], "amount": [5.0, 7.5]}))
            >>> orders = bt_ibis.table("orders", s)  # doctest: +SKIP
            >>> expr = orders.filter(orders.amount > 6).select("id")  # doctest: +SKIP
            >>> bt_ibis.to_dataset(expr, s).to_pydict()  # doctest: +SKIP
            {'id': [2]}
    """
    ibis = _ibis()
    schema = _session(session).table(name).schema
    return ibis.table(ibis.Schema.from_pyarrow(schema), name=name)


def to_dataset(expr: Any, session: Session | None = None) -> Dataset:
    """Compile the Ibis expression `expr` to SQL and run it on `session`, lazily.

    Each table `expr` names must be registered in the session under that name. Not yet
    verified against a live Ibis installation; see tests/PENDING_VERIFICATION.md.

    Args:
        expr: An Ibis table or scalar expression over unbound tables.
        session: The session that resolves the tables. None uses `bt.current_session()`.

    Returns:
        A lazy `Dataset` of the result.

    Raises:
        SQLUnsupportedError: Ibis emitted SQL outside the subset Batcher translates.
        PlanError: A table the expression names is not in the session.
        MissingDependencyError: Ibis is not installed (the ``ibis`` extra).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.integrations import ibis as bt_ibis
            >>> s = bt.Session()
            >>> _ = s.register("t", bt.from_pydict({"g": ["a", "b", "a"], "v": [1, 2, 3]}))
            >>> t = bt_ibis.table("t", s)  # doctest: +SKIP
            >>> expr = t.group_by("g").aggregate(total=t.v.sum()).order_by("g")  # doctest: +SKIP
            >>> bt_ibis.to_dataset(expr, s).to_pydict()  # doctest: +SKIP
            {'g': ['a', 'b'], 'total': [4, 2]}
    """
    sql = _ibis().to_sql(expr, dialect="duckdb")
    return _session(session).sql(str(sql))
