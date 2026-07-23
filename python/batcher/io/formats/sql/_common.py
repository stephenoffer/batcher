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
import importlib
import json
from collections.abc import Mapping
from types import ModuleType
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError

__all__ = [
    "apply_predicate",
    "apply_projection",
    "connection_fingerprint",
    "probe_is_typed",
    "push_down",
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
    """Import an optional driver module, or raise a typed install hint.

    Args:
        module: The importable module name (e.g. ``"adbc_driver_manager"``).
        extra: The Batcher extra that provides it (e.g. ``"sql"``).

    Returns:
        The imported module.

    Raises:
        BackendError: If the module is not installed, with a `pip install`
            instruction for the relevant extra.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised only without the driver
        raise BackendError(
            f"{module!r} is required for this source; install it with "
            f"pip install 'batcher-engine[{extra}]'"
        ) from exc


def wrap_subquery(query: str, *, table: str | None = None) -> str:
    """Normalize a logical read into a parenthesized derived table.

    A ``table=`` read becomes ``SELECT * FROM <table>``; an arbitrary query is
    wrapped as ``(<query>) AS _bc`` so projection/predicate can be layered on it
    uniformly without parsing the inner SQL.
    """
    inner = query if query is not None else f"SELECT * FROM {table}"
    return f"(\n{inner}\n) AS _bc"


def apply_projection(query: str, projection: list[str] | None, *, table: str | None = None) -> str:
    """Rewrite a read to select only `projection` columns.

    Returns ``SELECT <cols> FROM (<query>) AS _bc``; with no projection returns
    ``SELECT * FROM (<query>) AS _bc``. Column names are emitted verbatim — the
    caller is responsible for trusted/identifier-safe column names (they come
    from the plan's projection-pushdown, not user free-text).
    """
    cols = ", ".join(projection) if projection else "*"
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

    Returns:
        The SQL to send to the server.
    """
    from batcher.io.predicate import to_sql_where

    where = to_sql_where(predicate) if predicate is not None else None
    if extra_where is not None:
        # Parenthesized: the fragment may contain a bare OR (`k < 5 OR k IS NULL`), which
        # would otherwise bind looser than the AND and silently widen the partition to
        # every row with a NULL key — duplicating those rows across every split.
        where = f"({where}) AND ({extra_where})" if where else f"({extra_where})"
    if where is None:
        return apply_projection(query, projection, table=table)
    filtered = f"SELECT * FROM {wrap_subquery(query, table=table)} WHERE {where}"
    return apply_projection(filtered, projection)
