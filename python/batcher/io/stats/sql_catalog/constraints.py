"""Statistics a table's *declared constraints* prove, rather than a sample estimates.

Every other probe in this package reads a figure `ANALYZE` measured from a fraction of the
rows, so nothing they produce may answer a query. A schema constraint is the opposite kind
of fact: the database refuses every write that would violate it, so "this column holds no
nulls" and "these values are distinct" are guarantees rather than measurements — the same
standing as a Parquet footer's null count, and available from a table nobody has ever
analyzed.

Both map onto a `ColumnStat` facet the optimizer already reasons from, which is what makes
them worth a query:

  - **NOT NULL** -> `null_count=0` at EXACT. `IS NOT NULL` filters vanish
    (`zonemap_pruning._is_null_status`), `count(col)` answers from the row count, and
    `orderings_satisfy` may relax null placement (`RelStats.non_null_columns`).
  - **PRIMARY KEY / single-column UNIQUE** -> `ndv = rows`. A join against such a column is
    at most 1:N, which is the difference between an estimated join cardinality and a known
    one, and `Distinct`/`GROUP BY` over it are provably no-ops
    (`identities._a_key_column_is_unique`).

Only *single-column* constraints are read. A composite key says nothing about any one of
its columns alone — `UNIQUE (order_id, line_no)` does not make `order_id` distinct — and
recording it per column would claim exactly that.
"""

from __future__ import annotations

from typing import Any

from batcher.io.stats.sql_catalog.probes import RunRows, _at, _to_int
from batcher.plan.stats import ColumnStat, Provenance

__all__ = ["constraint_column_stats"]


#: Dialects whose `information_schema` follows the ANSI shape these two queries assume.
#: Oracle and SQLite have no `information_schema` and are handled separately below.
_ANSI_INFORMATION_SCHEMA = frozenset(
    {"postgres", "redshift", "mysql", "snowflake", "clickhouse", "sqlserver", "duckdb"}
)


def constraint_column_stats(
    run_rows: RunRows,
    dialect: str,
    table: str,
    row_count: int | None,
    *,
    rows_exact: bool = False,
) -> dict[str, ColumnStat]:
    """Per-column `ColumnStat` from the table's *declared constraints*, or an empty dict.

    Unlike every other probe in this module, these are not estimates. A `NOT NULL` or a
    single-column `PRIMARY KEY`/`UNIQUE` is enforced on every write, so the derived facets
    are EXACT and may answer an exact query — subject to the row count they are resolved
    against also being exact, which is why `rows_exact` is a separate argument rather than
    inferred.

    Composite constraints are ignored on purpose: `UNIQUE (a, b)` constrains the *pair* and
    says nothing about either column alone.

    Args:
        run_rows: Runs a catalog query and returns its rows, each a positional sequence.
        dialect: The catalog dialect (see `dialect_for_driver`).
        table: The unqualified table name.
        row_count: The table's row count, which a uniqueness constraint resolves into a
            distinct count. Without it, uniqueness contributes nothing.
        rows_exact: Whether `row_count` is itself exact. A distinct count derived from an
            estimate (Postgres ``reltuples``) is tagged SKETCH, so it sharpens join
            cardinality without ever answering a `count_distinct`.

    Returns:
        A ``{column_name: ColumnStat}`` mapping (possibly empty).
    """
    not_null = _not_null_columns(run_rows, dialect, table)
    unique = _unique_columns(run_rows, dialect, table)
    out: dict[str, ColumnStat] = {}
    for name in not_null | unique:
        # A UNIQUE column that also permits nulls is *not* distinct-per-row: most dialects
        # allow many null rows under one unique index, so `ndv` would over-count by the
        # nulls. Resolving it against the row count is only sound for the NOT NULL case,
        # which is every primary key.
        distinct = row_count if (name in unique and name in not_null) else None
        if name in not_null:
            null_count: float | None = 0.0
            null_prov: Provenance | None = Provenance.EXACT
        else:
            null_count, null_prov = None, None
        if distinct is None:
            ndv: float | None = None
            ndv_prov: Provenance | None = None
        else:
            ndv = float(distinct)
            ndv_prov = Provenance.EXACT if rows_exact else Provenance.SKETCH
        if null_count is None and ndv is None:
            continue
        out[name] = ColumnStat(
            null_count=null_count,
            ndv=ndv,
            null_count_provenance=null_prov,
            ndv_provenance=ndv_prov,
        )
    return out


#: dialect -> query listing the table's NOT NULL column names, one per row.
_NOT_NULL_QUERIES: dict[str, str] = {
    "ansi": (
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = '{table}' AND is_nullable = 'NO'"
    ),
    "oracle": (
        "SELECT column_name FROM all_tab_columns "
        "WHERE table_name = UPPER('{table}') AND nullable = 'N'"
    ),
}

#: dialect -> query listing columns covered by a single-column PK/UNIQUE constraint.
#: The `HAVING count(*) = 1` is what restricts the answer to single-column constraints;
#: without it a composite key would report each of its columns as individually unique.
_UNIQUE_QUERIES: dict[str, str] = {
    "ansi": (
        "SELECT MIN(k.column_name) FROM information_schema.table_constraints c "
        "JOIN information_schema.key_column_usage k "
        "ON c.constraint_name = k.constraint_name "
        "AND c.table_name = k.table_name "
        "WHERE c.table_name = '{table}' "
        "AND c.constraint_type IN ('PRIMARY KEY', 'UNIQUE') "
        "GROUP BY c.constraint_name HAVING COUNT(*) = 1"
    ),
    "oracle": (
        "SELECT MIN(k.column_name) FROM all_constraints c "
        "JOIN all_cons_columns k ON c.constraint_name = k.constraint_name "
        "WHERE c.table_name = UPPER('{table}') AND c.constraint_type IN ('P', 'U') "
        "GROUP BY c.constraint_name HAVING COUNT(*) = 1"
    ),
}


def _constraint_dialect(dialect: str) -> str | None:
    """The constraint-query family `dialect` belongs to, or None when it has none."""
    if dialect in _ANSI_INFORMATION_SCHEMA:
        return "ansi"
    return dialect if dialect in _NOT_NULL_QUERIES else None


def _not_null_columns(run_rows: RunRows, dialect: str, table: str) -> frozenset[str]:
    """Names of `table`'s NOT NULL columns, or empty when the catalog cannot say."""
    if dialect == "sqlite":
        return _sqlite_table_info(run_rows, table)[0]
    family = _constraint_dialect(dialect)
    if family is None:
        return frozenset()
    return _first_column_names(run_rows, _NOT_NULL_QUERIES[family].format(table=table))


def _unique_columns(run_rows: RunRows, dialect: str, table: str) -> frozenset[str]:
    """Names of `table`'s single-column PK/UNIQUE columns, or empty when unavailable."""
    if dialect == "sqlite":
        return _sqlite_table_info(run_rows, table)[1]
    family = _constraint_dialect(dialect)
    if family is None:
        return frozenset()
    return _first_column_names(run_rows, _UNIQUE_QUERIES[family].format(table=table))


def _sqlite_table_info(run_rows: RunRows, table: str) -> tuple[frozenset[str], frozenset[str]]:
    """SQLite's `(not_null, single_column_pk)` from ``PRAGMA table_info``.

    SQLite has no `information_schema`; the pragma is the equivalent, returning
    ``(cid, name, type, notnull, dflt_value, pk)`` per column. `pk` is the column's
    1-based position in the primary key (0 when it is not part of one), so a composite
    key is recognized by more than one column reporting a non-zero `pk` — and excluded,
    exactly as the `HAVING COUNT(*) = 1` excludes it for every other dialect.

    A `UNIQUE` constraint is deliberately not read here: it needs a second pragma per index
    (``index_list`` then ``index_info``), which is a query fan-out rather than the single
    probe every other dialect costs, for the rarer half of the same fact.
    """
    try:
        rows = run_rows(f"PRAGMA table_info('{table}')")
    except Exception:
        return frozenset(), frozenset()
    not_null: set[str] = set()
    key: list[str] = []
    for row in rows or ():
        name = _at(row, 1)
        if name is None:
            continue
        if _to_int(_at(row, 3)):
            not_null.add(str(name))
        if _to_int(_at(row, 5)):
            key.append(str(name))
    return frozenset(not_null), frozenset(key) if len(key) == 1 else frozenset()


def _first_column_names(run_rows: RunRows, sql: str) -> frozenset[str]:
    """The first field of every row of `sql`, as a set of names. Empty on any failure."""
    try:
        rows = run_rows(sql)
    except Exception:
        return frozenset()
    return frozenset(str(row[0]) for row in rows or () if row and row[0] is not None)


def _merge_column_stats(
    sampled: dict[str, ColumnStat], declared: dict[str, ColumnStat]
) -> dict[str, ColumnStat]:
    """Fold constraint-derived facets into the sampled ones, letting the exact fact win.

    The two probes overlap: `pg_stats` reports a sampled `null_frac` for a column whose
    schema already declares it NOT NULL, and an `n_distinct` of -1 for one already declared
    a primary key. Where they overlap the declared fact is strictly better — exact rather
    than sampled, and current rather than as-of-last-`ANALYZE` — so it replaces the sampled
    one. Every facet the constraint says nothing about (`mcv`, `quantiles`, and a `null_count`
    for a nullable column) is carried through untouched.
    """
    import dataclasses

    out = dict(sampled)
    for name, fact in declared.items():
        base = out.get(name)
        if base is None:
            out[name] = fact
            continue
        changes: dict[str, Any] = {}
        if fact.null_count_provenance is Provenance.EXACT:
            changes["null_count"] = fact.null_count
            changes["null_count_provenance"] = Provenance.EXACT
        if fact.ndv is not None:
            changes["ndv"] = fact.ndv
            changes["ndv_provenance"] = fact.ndv_provenance
        out[name] = dataclasses.replace(base, **changes) if changes else base
    return out
