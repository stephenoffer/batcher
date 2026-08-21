"""Shared helpers for SQL/warehouse sources — query rewriting and import guards.

Every relational backend (ADBC, ConnectorX, Snowflake, BigQuery, ClickHouse,
ODBC) shares the same control-plane shaping: a single logical query, optional
projection (rewrite the SELECT column list), optional predicate (append a
WHERE), and a deferred optional-dependency import that raises a typed
`BackendError` telling the user which extra to install. Centralizing these keeps
each backend module small and the behavior identical across backends.

Nothing here touches a row: query strings are shaped in Python, the data plane
stays Arrow-only on the worker.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any

import pyarrow as pa

from batcher.io.formats.sql.uri import quote_identifier

__all__ = [
    "apply_predicate",
    "apply_projection",
    "connection_fingerprint",
    "count_query",
    "identifier_quoter",
    "probe_is_typed",
    "push_down",
    "pushed_sql",
    "require_module",
    "schema_probe",
    "wrap_subquery",
]

#: Connection-kwarg names that hold authentication material rather than identity.
#:
#: These are excluded from the fingerprint for a practical reason: rotating a password
#: must not change which relation the learned statistics belong to. Including them would
#: orphan a table's accumulated stats every rotation, silently returning the optimizer to
#: cold estimates on a schedule.
_NON_IDENTIFYING = frozenset(
    {"password", "passwd", "token", "access_token", "secret", "api_key", "passphrase", "pwd"}
)


def connection_fingerprint(material: Mapping[str, Any]) -> str:
    """A short, stable, non-secret discriminator for a database connection.

    `identity()` is the key a source's learned statistics are stored under, so two sources
    with the same identity are treated as the same relation. Keyed on the query alone,
    ``SELECT * FROM orders`` against **production** and against **staging** collide — and
    the failure is invisible: Kyber applies the billion-row table's cardinalities to the
    thousand-row one, picks a plan for the wrong data, and nothing errors. This makes the
    connection part of the key so those are different relations.

    The digest is `sha256`, not `hash()`: Python salts `hash()` per process, so an identity
    built on it would differ on every run and no statistic would ever be reused — the
    feedback loop would look like it worked while never actually improving a plan.

    Args:
        material: Connection kwargs (host, port, database, path, credentials …).

    Returns:
        A 12-character hex digest, or ``"-"`` when there is nothing identifying.
    """
    identifying = {
        key: str(value) for key, value in material.items() if key.lower() not in _NON_IDENTIFYING
    }
    if not identifying:
        return "-"
    blob = json.dumps(identifying, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def require_module(module: str, *, extra: str) -> ModuleType:
    """Import a database driver, or raise a typed install hint naming its extra.

    A thin adapter over `_internal.optional.require`, the engine's one optional-dependency
    guard, kept under this name because thirteen SQL backends call it. It was a *fourth* copy
    of that guard — beside `nosql.base._driver`, `dbapi.source._import_driver` and `require`
    itself — and each copy phrased the failure differently while `require`'s already carried
    the `install` field and was already an `ImportError`, so ``except ImportError`` around an
    optional read worked in some backends and not others.

    Args:
        module: Importable driver module name, e.g. ``"snowflake.connector"``.
        extra: The Batcher extra that installs it.

    Returns:
        The imported module.

    Raises:
        MissingDependencyError: If the driver is not installed. It is both a `BackendError`
            and an `ImportError`, so handlers written against either spelling still catch it.
    """
    from batcher._internal.optional import require

    return require(module, feature=f"The {extra} source", provides=module, extra=extra)


def wrap_subquery(query: str, *, table: str | None = None) -> str:
    """Normalize a logical read into a parenthesized derived table.

    A ``table=`` read becomes ``SELECT * FROM <table>``; an arbitrary query is
    wrapped as ``(<query>) AS _bc`` so projection/predicate can be layered on it
    uniformly without parsing the inner SQL.
    """
    inner = query if query is not None else f"SELECT * FROM {table}"
    return f"(\n{inner}\n) AS _bc"


def _identity(name: str) -> str:
    """The default identifier rendering: verbatim, as every caller had before quoting."""
    return name


def apply_projection(
    query: str,
    projection: list[str] | None,
    *,
    table: str | None = None,
    quote: Callable[[str], str] = _identity,
) -> str:
    """Rewrite a read to select only `projection` columns.

    Returns ``SELECT <cols> FROM (<query>) AS _bc``; with no projection returns
    ``SELECT * FROM (<query>) AS _bc``.

    `quote` delimits each column for the target dialect. Emitted verbatim by default,
    which is what a caller that cannot identify its dialect must keep doing — and which
    breaks on three ordinary names. A reserved word (``order``, ``user``, ``date``) is a
    syntax error; a name holding a space is worse, because ``SELECT my col`` parses as the
    column ``my`` aliased to ``col`` and returns the wrong column under the right name.
    See `uri.quote_identifier`.
    """
    cols = ", ".join(quote(c) for c in projection) if projection else "*"
    return f"SELECT {cols} FROM {wrap_subquery(query, table=table)}"


def schema_probe(query: str | None, *, table: str | None = None) -> str:
    """`query` shaped to return its column types and **no rows**.

    Every SQL source here answered `schema()` by running the user's query in full and taking
    `.schema` off the materialized Arrow table. The column names of a billion-row join cost
    the billion-row join — and because the plan needs the schema *before* it executes, an
    ordinary ``read(...).filter(...).collect()`` submitted the whole query **twice**. On a
    warehouse that bills per query or per byte scanned, the schema lookup is a second full
    invoice for a result that is then discarded.

    ``WHERE 1 = 0`` is the portable spelling: it is valid on every backend here, and each
    one's planner folds it to an empty scan before touching storage, so the round trip
    returns the schema having read nothing. The result set is empty but fully typed, which
    is exactly what `schema()` needs.

    Args:
        query: The base read, or ``None`` when reading `table` directly.
        table: The table name, when the read is a plain table rather than a query.

    Returns:
        SQL returning zero rows with the query's full schema.
    """
    return f"SELECT * FROM {wrap_subquery(query, table=table)} WHERE 1 = 0"


#: Alias the row-count query gives its single value.
#:
#: No leading underscore, deliberately: Oracle rejects an unquoted identifier that starts
#: with one, so ``AS _bc_n`` is a syntax error there rather than a portable alias. The
#: caller reads the value positionally anyway — servers fold an unquoted alias's case
#: (Oracle and Snowflake upper-case it), so selecting it back by the name written here
#: would fail on exactly the backends that accepted the query.
COUNT_COLUMN = "bc_n"


def count_query(query: str | None, *, table: str | None = None) -> str:
    """`query` shaped to return its row count and nothing else.

    The sibling of `schema_probe`, and it exists for the same reason: `ds.count()` on a
    warehouse relation had no way to ask the server how many rows there are, so it counted
    them the only way it could — by reading the relation. Worse than it sounds, because a
    ``COUNT(*)`` needs no columns at all and the projection that would have narrowed the
    read is empty, which `apply_projection` renders as ``SELECT *``. So counting a table
    transferred every column of every row of it, to return one integer.

    ``SELECT COUNT(*) FROM (…)`` is ANSI and needs no dialect gate the way a row cap does:
    every backend these connectors reach accepts it, and the server answers from an index
    or its own statistics rather than a scan.

    Args:
        query: The base read, or ``None`` when counting `table` directly.
        table: The table name, when the read is a plain table rather than a query.

    Returns:
        SQL returning one row with one column, `COUNT_COLUMN`.
    """
    return f"SELECT COUNT(*) AS {COUNT_COLUMN} FROM {wrap_subquery(query, table=table)}"


def probe_is_typed(schema: pa.Schema) -> bool:
    """True iff a probe's schema carries real types rather than untyped empty columns.

    A probe is only a valid substitute for the full read if the backend types an empty
    result set from its *query metadata*. Every backend here does — that is what makes
    ``WHERE 1 = 0`` the standard schema-discovery idiom — but a driver that instead infers
    types from the rows it saw would hand back `null`-typed columns for a zero-row result.

    That failure would be silent and severe: `schema()` is what the engine types its
    operators against, so a `null` where an `int64` belongs is the same broken contract the
    CSV and Avro fixes were about, reintroduced by the very change meant to make schema
    lookup cheap. So the probe result is checked, and a caller that sees `False` falls back
    to the full read — slow, which is merely what it did before, rather than wrong.

    Args:
        schema: The schema the zero-row probe returned.

    Returns:
        True if the probe can be trusted; False if the caller should fall back.
    """
    return bool(schema) and not any(pa.types.is_null(f.type) for f in schema)


def apply_predicate(sql: str, predicate: str | None) -> str:
    """Append a SQL predicate as an outer ``WHERE`` on an already-shaped read."""
    if not predicate:
        return sql
    return f"SELECT * FROM (\n{sql}\n) AS _bcp WHERE {predicate}"


def push_down(
    query: str,
    predicate: dict | None = None,
    projection: list[str] | None = None,
    *,
    table: str | None = None,
    extra_where: str | None = None,
    limit: int | None = None,
    order_by: tuple[tuple[str, bool, bool], ...] | None = None,
    quote: Callable[[str], str] = _identity,
) -> str:
    """`query` with Kyber's pushed projection and predicate folded into the SQL itself.

    This is the whole of "connect a SQL connector to the optimizer". The `WHERE` and the column
    list have to execute **in the database**, because that is the only place they can avoid
    work: a predicate applied after the result set has crossed the wire has already cost the
    scan, the network, and the driver's memory. On a TB table the difference is not a constant
    factor — it is whether the query runs at all.

    It has to happen at **split-planning** time, not at read time, for the same reason. A split
    is a picklable locator that a worker rebuilds a reader from; if the pushdown lives anywhere
    but inside the split's own query, the worker reconstructs an *unfiltered* read and the
    server never hears about the filter. That is precisely how every SQL connector here was
    behaving on the distributed path: correct results, and the entire table pulled per worker.

    A predicate the translator cannot express returns None from `to_sql_where` and is simply not
    pushed — the engine's `Filter` re-checks every row regardless, so an unpushed predicate is
    always correct and merely slower.

    **The predicate has to be applied *below* the projection, not above it.** Kyber pushes the
    two independently, and it routinely pushes a projection that does not include the column the
    predicate filters on — ``select("id").filter(col("country") == "US")`` narrows to ``id`` and
    filters on ``country``. Projecting first produced
    ``SELECT * FROM (SELECT id FROM …) WHERE country = 'US'``, where ``country`` no longer
    exists: not a slow query but a hard ``no such column`` error from the server, on every
    connector that shares this helper. Filtering first and projecting the result is both correct
    and the order the database would have chosen anyway.

    Args:
        query: The base read (a user query, or `SELECT * FROM table` when `table` is given).
        predicate: The predicate IR Kyber pushed to this scan.
        projection: The columns Kyber pushed to this scan.
        table: The table name, when the read is a plain table rather than a query.
        extra_where: A ready-made SQL fragment ANDed with the pushed predicate — used by
            range partitioning to give each split its own slice of the key space. It is
            applied at the same depth as the predicate, and for the same reason: a
            partition fragment layered above the projection would reference a partition
            column the projection had already dropped.
        limit: The most rows the plan needs from this read (`PhysicalPlan.source_limits`),
            appended as a trailing ``LIMIT``. Outermost, unlike the predicate: it counts
            the rows the read *returns*, so it has to sit above the filter rather than
            below it. Callers pass this only for a backend whose dialect accepts the
            clause (`uri.supports_limit_clause`).
        order_by: The ordering a pushed row cap is taken in, one
            ``(column, descending, nulls_first)`` per key — a *top-N* rather than a
            prefix. Emitted with an explicit ``NULLS`` clause, without which the server's
            "first n" and the engine's disagree wherever they place a null, and the read
            returns the wrong rows rather than merely extra ones. Callers pass this only
            for a dialect that accepts the clause (`uri.supports_nulls_ordering`).
        quote: How to delimit a column name for the target dialect
            (`uri.quote_identifier`). Applies to the projection and to the pushed
            predicate alike, since a reserved-word column breaks both.

    Returns:
        The SQL to send to the server.
    """
    from batcher.io.predicate import to_sql_where

    where = to_sql_where(predicate, quote=quote) if predicate is not None else None
    if extra_where is not None:
        # Parenthesized: the fragment may contain a bare OR (`k < 5 OR k IS NULL`), which
        # would otherwise bind looser than the AND and silently widen the partition to
        # every row with a NULL key — duplicating those rows across every split.
        where = f"({where}) AND ({extra_where})" if where else f"({extra_where})"
    if where is None:
        shaped = apply_projection(query, projection, table=table, quote=quote)
    else:
        filtered = f"SELECT * FROM {wrap_subquery(query, table=table)} WHERE {where}"
        shaped = apply_projection(filtered, projection, quote=quote)
    return _capped(_ordered(shaped, order_by, quote), limit)


def identifier_quoter(dialect: str | None) -> Callable[[str], str]:
    """How to delimit an identifier for `dialect`; verbatim when it is unknown.

    A projection reaches the server as a column list, and three ordinary names break
    unquoted: a reserved word (``order``, ``user``, ``key``), a name holding a space
    (which parses as a column *aliased* to the second word, returning the wrong column
    under the right name), and an unaliased aggregate from a user's own query.

    An unknown dialect renders verbatim rather than guessing a delimiter: ODBC names a
    driver rather than a dialect, and quoting with the wrong character turns a working
    query into a syntax error.

    Args:
        dialect: The connection-URI scheme naming the SQL dialect, or empty when it
            cannot be known.

    Returns:
        A callable rendering one identifier for that dialect.
    """
    if not dialect:
        return _identity
    known = dialect
    return lambda name: quote_identifier(name, known)


def pushed_sql(
    query: str,
    *,
    predicate: dict | None = None,
    projection: list[str] | None = None,
    limit: int | None = None,
    ordering: tuple[tuple[str, bool, bool], ...] | None = None,
    extra_where: str | None = None,
    table: str | None = None,
    quote: Callable[[str], str] = _identity,
    supports_ordering: bool = False,
    supports_limit: bool = False,
) -> str:
    """`query` with every part of the plan this backend can push folded into the SQL.

    A thin capability gate in front of `push_down`, kept in one place because the rule it
    encodes is the one pushdown that can be *wrong* rather than merely incomplete. An
    ordered cap is only sound if the ordering goes with it: a dialect that takes ``LIMIT``
    but cannot spell ``NULLS FIRST|LAST`` must drop the cap too, or the server returns its
    own idea of the first n -- the wrong rows, silently, with no error anywhere.

    Args:
        query: The one logical query the source reads.
        predicate: Kyber's pushed predicate, or None.
        projection: The pushed column list, or None for every column.
        limit: The pushed row cap, applied only when `supports_limit`.
        ordering: The pushed sort keys as `(column, descending, nulls_first)`, applied
            only when `supports_ordering`.
        extra_where: A partitioning range clause ANDed with the predicate.
        table: The table to select from when `query` names one.
        quote: How to delimit an identifier, from `identifier_quoter`.
        supports_ordering: Whether the dialect accepts an explicit ``NULLS`` clause.
        supports_limit: Whether the dialect accepts a ``LIMIT`` clause.

    Returns:
        The SQL one split executes, pushdown included.
    """
    ordered = ordering if (ordering and supports_ordering) else None
    capped = None if (ordering and ordered is None) else (limit if supports_limit else None)
    return push_down(
        query,
        predicate,
        projection,
        table=table,
        extra_where=extra_where,
        limit=capped,
        order_by=ordered,
        quote=quote,
    )


def _ordered(
    sql: str,
    order_by: tuple[tuple[str, bool, bool], ...] | None,
    quote: Callable[[str], str],
) -> str:
    """`sql` with a trailing ``ORDER BY``, when the cap above it is a top-N.

    The ``NULLS`` clause is always explicit. Left to the server's default it is not a
    style question: SQLite sorts nulls first on an ascending order where PostgreSQL and
    DuckDB sort them last, so the same ``ORDER BY k LIMIT 2`` asks two servers for
    different rows, and only one of them matches what the engine would have computed.
    """
    if not order_by:
        return sql
    keys = ", ".join(
        f"{quote(column)} {'DESC' if descending else 'ASC'} "
        f"NULLS {'FIRST' if nulls_first else 'LAST'}"
        for column, descending, nulls_first in order_by
    )
    return f"{sql} ORDER BY {keys}"


def _capped(sql: str, limit: int | None) -> str:
    """`sql` with a trailing row cap, when the plan asked for one."""
    return sql if limit is None else f"{sql} LIMIT {int(limit)}"
