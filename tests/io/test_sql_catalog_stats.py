"""Catalog-derived SQL statistics: dialect mapping, row/byte/column probes, and wiring.

`io.stats.sql_catalog` reads a table's row count, on-disk size, and per-column
null/ndv/mcv/quantile statistics straight from a database's system catalog, so a
warehouse table reaches Kyber with the same facets a Parquet footer supplies. The pure
extraction functions are exercised with mock query callbacks (they must map every
dialect's catalog shape correctly without a live server), and the end-to-end wiring is
pinned against SQLite — a real SQL engine from the standard library, so
`DBAPISource.statistics()` is a genuine catalog round trip with no optional dependency.

Three invariants are load-bearing and asserted directly:

1. **No *sampled* catalog stat answers an exact query.** Row-count exactness follows the
   engine's own guarantee (Snowflake maintains it, Postgres `reltuples` drifts), and every
   `pg_stats` column stat is a sampled `ANALYZE` estimate, so `exact_rows` and the
   per-facet provenance gates must never claim EXACT for an estimate.
2. **A *declared* one does.** A NOT NULL or single-column PRIMARY KEY is enforced on every
   write, so it is EXACT — and where the two overlap on one column, the declared fact
   replaces the sampled one. The gating that separates the two families is the point of
   the constraint tests: a distinct count resolved against an estimated row count inherits
   that estimate's inexactness, and a nullable UNIQUE column declares no distinct count at
   all, because many null rows may sit under one unique index.
3. **A `query=` read declares nothing.** Only a base `table=` has a catalog entry.
"""

from __future__ import annotations

import sqlite3

import pytest

from batcher.io.formats.sql.dbapi import DBAPISource
from batcher.io.stats import (
    catalog_byte_size,
    catalog_column_stats,
    catalog_row_count,
    constraint_column_stats,
    dialect_for_driver,
    sql_statistics,
)
from batcher.plan.stats import Provenance

pytestmark = pytest.mark.unit


# --- dialect resolution ---------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("psycopg2", "postgres"),
        ("psycopg", "postgres"),
        ("postgresql", "postgres"),
        ("postgresql+psycopg", "postgres"),
        ("adbc_driver_postgresql", "postgres"),
        ("cockroachdb", "postgres"),
        ("sqlite3", "sqlite"),
        ("adbc_driver_sqlite", "sqlite"),
        ("clickhouse_connect", "clickhouse"),
        ("snowflake.connector", "snowflake"),
        ("pymysql", "mysql"),
        ("mariadb", "mysql"),
        ("cx_Oracle", "oracle"),
        ("oracledb", "oracle"),
        ("pyodbc", "sqlserver"),
        ("duckdb", "duckdb"),
        ("trino", "trino"),
        ("redshift_connector", "redshift"),
        ("some_unknown_driver", None),
        ("", None),
        (None, None),
    ],
)
def test_dialect_for_driver(name, expected) -> None:
    """Every supported driver/scheme resolves to its catalog dialect; the rest to None."""
    assert dialect_for_driver(name) == expected


# --- row count ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dialect", "value", "expect_rows", "expect_exact"),
    [
        ("snowflake", 999, 999, True),
        ("clickhouse", 500, 500, True),
        ("sqlserver", 1234, 1234, True),
        ("postgres", 12345.0, 12345, False),
        ("mysql", 42, 42, False),
        ("oracle", 88, 88, False),
        ("redshift", 7, 7, False),
        ("duckdb", 64, 64, False),
    ],
)
def test_catalog_row_count(dialect, value, expect_rows, expect_exact) -> None:
    """Each dialect's catalog query yields the right count and exactness."""
    stats = catalog_row_count(lambda _sql: value, dialect, "t")
    assert stats is not None
    assert stats.row_count == expect_rows
    assert stats.exact_rows is expect_exact


def test_catalog_row_count_unknown_dialect_is_none() -> None:
    """An unmapped dialect probes nothing rather than running a wrong query."""
    assert catalog_row_count(lambda _sql: 1, "nosuchdb", "t") is None


def test_catalog_row_count_failure_is_none() -> None:
    """A catalog query that raises (no permission, missing view) degrades to None."""

    def boom(_sql: str) -> int:
        raise RuntimeError("permission denied")

    assert catalog_row_count(boom, "postgres", "t") is None


def test_sqlite_stat1_parsing() -> None:
    """SQLite's whitespace-joined stat string is parsed to its leading row count."""
    stats = catalog_row_count(lambda _sql: "1000 4 1", "sqlite", "t")
    assert stats is not None
    assert stats.row_count == 1000
    assert stats.exact_rows is False  # sqlite_stat1 drifts until re-ANALYZEd


def test_sqlite_unanalyzed_is_none() -> None:
    """A table never ANALYZEd has no sqlite_stat1 row, so no count is claimed."""
    assert catalog_row_count(lambda _sql: None, "sqlite", "t") is None


# --- byte size ------------------------------------------------------------------


@pytest.mark.parametrize("dialect", ["postgres", "mysql", "clickhouse", "snowflake", "redshift"])
def test_catalog_byte_size(dialect) -> None:
    """A dialect with a size catalog returns a positive byte count."""
    assert catalog_byte_size(lambda _sql: 8192, dialect, "t") == 8192


def test_catalog_byte_size_zero_is_none() -> None:
    """A zero/empty size is reported as None, not a misleading zero."""
    assert catalog_byte_size(lambda _sql: 0, "postgres", "t") is None


def test_catalog_byte_size_unknown_dialect() -> None:
    """A dialect with no size catalog (trino here) yields None without a query."""
    called = []
    catalog_byte_size(lambda sql: called.append(sql), "trino", "t")
    assert called == []


# --- column statistics (Postgres pg_stats) --------------------------------------


def _pg_rows() -> list[tuple]:
    """Two `pg_stats` rows: a skewed text column and a fully-distinct key."""
    return [
        # attname, null_frac, n_distinct, most_common_vals, most_common_freqs, histogram
        ("country", 0.25, 3.0, "{US,FR,DE}", "{0.5,0.3,0.2}", None),
        ("id", 0.0, -1.0, None, None, "{1,250,500,750,1000}"),
    ]


def test_pg_column_stats_facets() -> None:
    """pg_stats maps onto null_count, ndv, mcv, and a quantile grid."""
    cols = catalog_column_stats(lambda _sql: _pg_rows(), "postgres", "t", 1000)

    country = cols["country"]
    assert country.null_count == 250.0  # 0.25 * 1000
    assert country.ndv == 3.0
    assert country.mcv == {"US": 0.5, "FR": 0.3, "DE": 0.2}

    key = cols["id"]
    assert key.ndv == 1000.0  # n_distinct=-1 (all distinct) resolved against row count
    assert key.quantiles == {
        "probs": [0.0, 0.25, 0.5, 0.75, 1.0],
        "values": [1.0, 250.0, 500.0, 750.0, 1000.0],
    }


def test_pg_column_stats_never_exact() -> None:
    """Every catalog column stat is a sample, so no facet may answer an exact query."""
    cols = catalog_column_stats(lambda _sql: _pg_rows(), "postgres", "t", 1000)
    country = cols["country"]
    assert country.provenance is Provenance.SKETCH
    assert not country.ndv_is_exact
    assert not country.null_count_is_exact


def test_pg_column_stats_list_valued_arrays() -> None:
    """A driver that returns real Python lists (not `{..}` text) parses identically."""
    rows = [("c", 0.1, 2.0, ["a", "b"], [0.7, 0.3], None)]
    cols = catalog_column_stats(lambda _sql: rows, "postgres", "t", 100)
    assert cols["c"].mcv == {"a": 0.7, "b": 0.3}


def test_pg_column_stats_non_postgres_is_empty() -> None:
    """The `pg_stats` shape is read for Postgres and its kin only.

    Oracle and MySQL have their own catalogs and their own row shapes (asserted below);
    a dialect with neither must yield an empty map rather than mis-parse someone else's
    columns positionally.
    """
    assert catalog_column_stats(lambda _sql: _pg_rows(), "clickhouse", "t", 1000) == {}


def test_pg_column_stats_failure_is_empty() -> None:
    """A pg_stats query that raises degrades to an empty map, never propagating."""

    def boom(_sql: str) -> list:
        raise RuntimeError("relation pg_stats does not exist")

    assert catalog_column_stats(boom, "postgres", "t", 1000) == {}


# --- composition ----------------------------------------------------------------


def test_sql_statistics_composes_all_facets() -> None:
    """`sql_statistics` folds row count, byte size, and columns into one record."""

    def run_scalar(sql: str) -> object:
        if "reltuples" in sql:
            return 1000.0
        if "pg_total_relation_size" in sql:
            return 999_999
        return None

    stats = sql_statistics("postgres", "t", run_scalar=run_scalar, run_rows=lambda _sql: _pg_rows())
    assert stats is not None
    assert stats.row_count == 1000
    assert stats.exact_rows is False
    assert stats.byte_size == 999_999
    assert set(stats.columns) == {"country", "id"}


def test_sql_statistics_all_missing_is_none() -> None:
    """When the catalog yields nothing at all, no statistics record is produced."""
    assert sql_statistics("oracle", "t", run_scalar=lambda _sql: None) is None


# --- end-to-end wiring against SQLite -------------------------------------------


@pytest.fixture
def analyzed_db(tmp_path):
    """A SQLite database with a table that has been ANALYZEd (so sqlite_stat1 exists)."""
    path = tmp_path / "w.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE ev (id INTEGER, country TEXT)")
    conn.executemany(
        "INSERT INTO ev VALUES (?, ?)",
        [(i, "US" if i % 2 else "FR") for i in range(200)],
    )
    conn.execute("ANALYZE")
    conn.commit()
    conn.close()
    return path


def test_dbapi_statistics_from_sqlite_catalog(analyzed_db) -> None:
    """`DBAPISource(table=...)` reads its row count from the live SQLite catalog."""
    src = DBAPISource(module="sqlite3", connect_kwargs={"database": str(analyzed_db)}, table="ev")
    stats = src.statistics()
    assert stats is not None
    assert stats.row_count == 200
    assert stats.exact_rows is False  # sqlite_stat1 is advisory


def test_dbapi_statistics_query_read_is_none(analyzed_db) -> None:
    """A `query=` read has no catalog entry, so it declares no statistics."""
    src = DBAPISource(
        module="sqlite3",
        connect_kwargs={"database": str(analyzed_db)},
        query="SELECT id FROM ev WHERE country = 'US'",
    )
    assert src.statistics() is None


def test_dbapi_statistics_borrowed_connection(analyzed_db) -> None:
    """A borrowed (already-open) connection resolves its dialect from the connection type."""
    conn = sqlite3.connect(analyzed_db)
    try:
        src = DBAPISource(connection=conn, table="ev")
        stats = src.statistics()
        assert stats is not None
        assert stats.row_count == 200
    finally:
        conn.close()  # Batcher never closes a borrowed connection; the test owns it.


# --- declared constraints (NOT NULL, PRIMARY KEY / UNIQUE) ----------------------
#
# These are the one family here that is *not* a sample, so the assertions below are about
# exactness as much as extraction: a constraint is enforced on every write, so it may
# answer a query, while everything else in this module may only inform cost.


def _ansi_rows(not_null: list[str], unique: list[str]):
    """A `run_rows` that answers the two ANSI constraint queries and nothing else."""

    def run(sql: str):
        if "is_nullable" in sql:
            return [(c,) for c in not_null]
        if "constraint_type" in sql:
            return [(c,) for c in unique]
        return []

    return run


def test_constraint_not_null_is_exact() -> None:
    """A NOT NULL column declares `null_count=0` at EXACT — a constraint, not a sample."""
    stats = constraint_column_stats(_ansi_rows(["id"], []), "postgres", "t", 100)
    assert stats["id"].null_count == 0
    assert stats["id"].null_count_is_exact


def test_constraint_primary_key_gives_exact_ndv() -> None:
    """A single-column PK that is also NOT NULL has `ndv == rows`, EXACT when rows are."""
    stats = constraint_column_stats(
        _ansi_rows(["id"], ["id"]), "postgres", "t", 100, rows_exact=True
    )
    assert stats["id"].ndv == 100
    assert stats["id"].ndv_is_exact


def test_constraint_ndv_from_estimated_rows_is_not_exact() -> None:
    """Resolved against `reltuples`, the distinct count informs cost but answers nothing."""
    stats = constraint_column_stats(
        _ansi_rows(["id"], ["id"]), "postgres", "t", 100, rows_exact=False
    )
    assert stats["id"].ndv == 100
    assert not stats["id"].ndv_is_exact


def test_constraint_nullable_unique_declares_no_ndv() -> None:
    """A UNIQUE column that permits nulls is not distinct-per-row — most dialects allow
    many null rows under one unique index, so `ndv == rows` would over-count."""
    stats = constraint_column_stats(
        _ansi_rows([], ["email"]), "postgres", "t", 100, rows_exact=True
    )
    assert "email" not in stats


def test_constraint_composite_key_is_ignored() -> None:
    """`UNIQUE (a, b)` constrains the pair, so neither column alone becomes distinct.

    The ANSI query filters composites out server-side with `HAVING COUNT(*) = 1`; this
    pins the *consequence* — that a two-column key contributes no ndv — by answering the
    unique query with nothing, which is what that HAVING produces.
    """
    stats = constraint_column_stats(_ansi_rows(["a", "b"], []), "postgres", "t", 100)
    assert stats["a"].ndv is None
    assert stats["b"].ndv is None


def test_constraint_unknown_dialect_runs_no_query() -> None:
    """A dialect with no constraint catalog probes nothing rather than guessing a query."""
    called: list[str] = []

    def run(sql: str):
        called.append(sql)
        return []

    assert constraint_column_stats(run, "bigquery", "t", 100) == {}
    assert called == []


def test_constraint_failure_is_tolerated() -> None:
    """A permission error on the catalog yields no statistics, never an exception."""

    def run(_sql: str):
        raise RuntimeError("permission denied for information_schema")

    assert constraint_column_stats(run, "postgres", "t", 100) == {}


def test_sqlite_pragma_declares_not_null_and_key(analyzed_db) -> None:
    """SQLite has no information_schema; `PRAGMA table_info` carries the same two facts."""
    conn = sqlite3.connect(analyzed_db)
    conn.execute("CREATE TABLE k (id INTEGER PRIMARY KEY, name TEXT NOT NULL, note TEXT)")
    try:
        stats = constraint_column_stats(
            lambda sql: list(conn.execute(sql)), "sqlite", "k", 50, rows_exact=True
        )
    finally:
        conn.close()
    # `INTEGER PRIMARY KEY` is SQLite's rowid alias: `pk=1` but `notnull=0`, so it is
    # unique-but-nullable by the pragma's own account and declares no ndv.
    assert stats["name"].null_count == 0
    assert stats["name"].null_count_is_exact
    assert "note" not in stats


def test_declared_fact_overrides_the_sampled_one() -> None:
    """Where `pg_stats` and the schema overlap, the constraint wins — exact beats sampled.

    `pg_stats` reports a *sampled* null fraction for a column the schema already declares
    NOT NULL. Both describe the same column; only one of them can answer `has_nulls`.
    """

    def run(sql: str):
        if "pg_stats" in sql:
            return [("id", 0.02, -1.0, None, None, None)]
        if "is_nullable" in sql:
            return [("id",)]
        if "constraint_type" in sql:
            return [("id",)]
        return []

    stats = sql_statistics("clickhouse", "t", run_scalar=lambda _s: 100, run_rows=run)
    assert stats is not None
    # ClickHouse's `system.tables` count is transactionally maintained, hence exact.
    assert stats.exact_rows
    id_stat = stats.columns["id"]
    assert id_stat.null_count == 0
    assert id_stat.null_count_is_exact


# --- per-column catalogs beyond Postgres ----------------------------------------


def test_oracle_column_stats() -> None:
    """`all_tab_col_statistics` carries a distinct count, a null count and a width."""
    rows = [("ID", 1000, 0, 8), ("NAME", 12, 5, 24)]
    cols = catalog_column_stats(lambda _sql: rows, "oracle", "t", 1000)
    assert cols["ID"].ndv == 1000.0
    assert cols["NAME"].null_count == 5.0
    assert cols["NAME"].avg_bytes == 24.0
    # Oracle's figures come from DBMS_STATS sampling, so none may answer an exact query.
    assert not cols["NAME"].ndv_is_exact
    assert not cols["NAME"].null_count_is_exact


def test_mysql_index_cardinality_is_a_distinct_count() -> None:
    """An index's CARDINALITY estimates the distinct values of the column it leads."""
    cols = catalog_column_stats(lambda _sql: [("user_id", 5000)], "mysql", "t", 100000)
    assert cols["user_id"].ndv == 5000.0
    assert not cols["user_id"].ndv_is_exact


def test_mysql_zero_cardinality_is_discarded() -> None:
    """MySQL reports 0 for a never-analyzed table, and a 0 ndv estimates every equality
    at zero rows — worse than having no estimate at all."""
    assert catalog_column_stats(lambda _sql: [("c", 0)], "mysql", "t", 100) == {}


def test_column_stats_dialect_without_a_catalog() -> None:
    """A dialect with no column catalog runs no query rather than a wrong one."""
    called: list[str] = []
    assert catalog_column_stats(lambda s: called.append(s) or [], "snowflake", "t", 10) == {}
    assert called == []


def test_dbapi_statistics_carry_sqlite_constraints(tmp_path) -> None:
    """The constraint probe end to end, through a real connection rather than a mock.

    Pins the exactness split the whole family turns on. A NOT NULL column's zero null
    count is EXACT — the constraint holds however stale `sqlite_stat1` is — while the
    primary key's distinct count is resolved against that estimated row count and is
    therefore SKETCH. A column with no constraint contributes nothing at all.
    """
    path = tmp_path / "c.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE users (id INTEGER NOT NULL PRIMARY KEY, email TEXT NOT NULL, bio TEXT)"
    )
    conn.executemany(
        "INSERT INTO users VALUES (?, ?, ?)", [(i, f"u{i}@x", None) for i in range(200)]
    )
    conn.execute("ANALYZE")
    conn.commit()
    conn.close()

    src = DBAPISource(module="sqlite3", connect_kwargs={"database": str(path)}, table="users")
    stats = src.statistics()
    assert stats is not None
    assert stats.exact_rows is False  # sqlite_stat1 is advisory

    assert stats.columns["email"].null_count == 0
    assert stats.columns["email"].null_count_is_exact
    assert stats.columns["id"].ndv == 200.0
    assert not stats.columns["id"].ndv_is_exact  # resolved against an estimated row count
    assert "bio" not in stats.columns  # nullable, unconstrained → nothing to declare
