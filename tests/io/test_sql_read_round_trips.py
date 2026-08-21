"""How many times a warm SQL query goes to the server, and why that number is 1.

A point lookup against a database is dominated by round trips, not by rows, so the count is
the measurement that matters. Before the work this file pins, one *warm* lookup issued
**seven** statements, of which one was the query:

```text
SELECT stat FROM sqlite_stat1 WHERE tbl = 'orders' LIMIT 1   -- catalog row count
SELECT SUM(pgsize) FROM dbstat WHERE name = 'orders'         -- catalog byte size
PRAGMA table_info('orders')                                  -- catalog column stats
PRAGMA table_info('orders')                                  -- ...asked for a second time
SELECT * FROM (SELECT * FROM orders) AS _bc WHERE 1 = 0      -- schema probe
SELECT * FROM (SELECT * FROM orders) AS _bc                  -- schema fallback, UNCAPPED
SELECT "id", "amt" FROM ... WHERE "id" = 12345               -- the query
```

Four separate defects, none of which any test could see, because every one of them produced
the right answer:

1. **The schema fallback had no cap.** `probe_is_typed` is `False` for essentially every PEP
   249 driver — PEP 249 exposes four coarse type singletons and most drivers report nothing
   for a zero-row result — so the "fallback" was *the* path, on every schema lookup, and it
   submitted `SELECT * FROM t`. The split reads one batch and abandons the cursor, which
   bounds the *client*; a default psycopg2 cursor is client-side and buffers the entire
   result set before `fetchmany` returns, so typing the columns of a large table pulled the
   whole table over the wire.
2. **Nothing cached `schema()`**, though `nosql.ScanSource` has cached its own all along.
3. **Nothing cached `statistics()`**, so three catalog queries and a connect ran on every
   terminal op — about a quarter of a point lookup's wall clock.
4. **The catalog session re-ran identical SQL.** `catalog_column_stats` and
   `constraint_column_stats` both ask SQLite for `PRAGMA table_info`, and neither knows about
   the other.

End to end the lookup went from 4.33 ms to 2.65 ms, and the 1,000-row scan from 7.53 ms to
5.01 ms. What is asserted here is the round-trip **count**, because that is what regresses
silently: every one of these defects returns correct rows.
"""

from __future__ import annotations

import sqlite3

import pytest

import batcher as bt
from batcher.io.formats.sql.dbapi import DBAPISource
from batcher.io.formats.sql.dbapi import source as source_module

pytestmark = pytest.mark.io


class _CountingCursor:
    def __init__(self, cursor, log: list[str]) -> None:
        self._cursor, self._log = cursor, log

    def execute(self, sql, *args, **kwargs):
        self._log.append(" ".join(sql.split()))
        return self._cursor.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class _CountingConnection:
    def __init__(self, connection, log: list[str]) -> None:
        self._connection, self._log = connection, log

    def cursor(self):
        return _CountingCursor(self._connection.cursor(), self._log)

    def __getattr__(self, name):
        return getattr(self._connection, name)


@pytest.fixture
def counted(tmp_path, monkeypatch):
    """A real SQLite database, plus the list of statements Batcher sends it."""
    path = tmp_path / "orders.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, amt REAL)")
    conn.executemany("INSERT INTO orders VALUES (?, ?)", [(i, float(i)) for i in range(2_000)])
    conn.execute("ANALYZE")
    conn.commit()
    conn.close()

    log: list[str] = []
    real_connect = source_module._connect
    monkeypatch.setattr(
        source_module,
        "_connect",
        lambda module, kwargs: _CountingConnection(real_connect(module, kwargs), log),
    )
    return f"sqlite:///{path}", log


def test_a_warm_point_lookup_makes_exactly_one_round_trip(counted) -> None:
    """The assertion the seven-statement version would have failed."""
    uri, log = counted
    ds = bt.read.sql(None, uri=uri, table="orders")
    ds.filter(bt.col("id") == 12).to_pydict()  # warm: schema and statistics are probed here
    log.clear()
    assert ds.filter(bt.col("id") == 12).to_pydict() == {"id": [12], "amt": [12.0]}
    assert len(log) == 1, f"a warm lookup sent {len(log)} statements: {log}"
    assert "WHERE" in log[0], "the one statement sent was not the query"


def test_the_first_query_probes_the_catalog_but_never_scans_the_table(counted) -> None:
    """A cold query may probe; what it may not do is submit an uncapped read."""
    uri, log = counted
    bt.read.sql(None, uri=uri, table="orders").filter(bt.col("id") == 12).to_pydict()
    uncapped = [
        sql
        for sql in log
        if sql.startswith("SELECT * FROM ( SELECT * FROM orders )")
        and "WHERE" not in sql
        and "LIMIT" not in sql
    ]
    assert not uncapped, f"an unbounded read was submitted: {uncapped}"


def test_the_schema_fallback_is_capped_where_the_dialect_allows_it(tmp_path) -> None:
    """`probe_is_typed` is False for nearly every PEP 249 driver, so this is the live path."""
    source = DBAPISource(uri=f"sqlite:///{tmp_path / 'x.db'}", table="orders")
    assert source.supports_limit
    assert source._sampling_sql().rstrip().endswith(f"LIMIT {source.batch_size}")


def test_an_unknown_dialect_submits_exactly_what_it_always_did(tmp_path) -> None:
    """A cap the server cannot parse turns a working read into a syntax error."""
    source = DBAPISource(module="pyodbc", connect_kwargs={}, table="orders")
    assert not source.supports_limit
    assert "LIMIT" not in source._sampling_sql()


def test_the_schema_is_probed_once_per_source(counted) -> None:
    uri, log = counted
    source = DBAPISource(uri=uri, table="orders")
    first = source.schema()
    log.clear()
    assert source.schema() == first
    assert log == [], f"the schema was re-probed: {log}"


def test_the_statistics_are_probed_once_per_source(counted) -> None:
    uri, log = counted
    source = DBAPISource(uri=uri, table="orders")
    first = source.statistics()
    assert first is not None and first.row_count == 2_000
    log.clear()
    assert source.statistics().row_count == 2_000
    assert log == [], f"the catalog was re-probed: {log}"


def test_a_cached_statistic_can_never_answer_an_exact_count() -> None:
    """The rule `api.source_stats` already states, applied to the live catalog probe.

    A Snowflake or SQL Server catalog count is transactional, so the *first* read of it may
    answer a terminal. A cached one may not: the table can have grown since, and an exact
    count that is stale is a wrong answer rather than a slow plan.
    """
    from batcher.plan.source_stats import SourceStatistics

    source = DBAPISource(module="sqlite3", connect_kwargs={}, table="orders")
    object.__setattr__(source, "_stats_cache", SourceStatistics(row_count=99, exact_rows=True))
    cached = source.statistics()
    assert cached.row_count == 99, "the cached numbers are still returned"
    assert cached.exact_rows is False, "a cached count must not be able to answer count()"


def test_the_catalog_session_answers_identical_sql_once(counted) -> None:
    """One connection, one transaction, one answer — so re-running it buys nothing."""
    uri, log = counted
    source = DBAPISource(uri=uri, table="orders")
    source.statistics()
    repeated = [sql for sql in log if log.count(sql) > 1]
    assert not repeated, f"the same catalog query was sent more than once: {set(repeated)}"


def test_a_cached_probe_does_not_change_what_the_source_is(tmp_path) -> None:
    """`identity()` and the plan cache key on what a source *is*, never on what it holds."""
    uri = f"sqlite:///{tmp_path / 'y.db'}"
    a = DBAPISource(uri=uri, table="orders")
    b = DBAPISource(uri=uri, table="orders")
    object.__setattr__(a, "_schema_cache", None)
    assert a == b and a.identity() == b.identity()


def test_a_held_dataset_re_reads_a_table_that_grew_under_it(tmp_path) -> None:
    """The exact correctness risk the statistics cache introduces, end to end.

    Cache a catalog count on a source, add rows to the table through that same source's
    sink, and ask the *held* `Dataset` for its count again. It must read the table rather
    than answer from the cached number — which is what marking a cached statistic advisory
    buys, and the only reason the cache is safe.
    """
    path = tmp_path / "orders.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, amt REAL)")
    conn.executemany("INSERT INTO orders VALUES (?, ?)", [(i, float(i)) for i in range(100)])
    conn.execute("ANALYZE")
    conn.commit()
    conn.close()
    uri = f"sqlite:///{path}"

    held = bt.read.sql(None, uri=uri, table="orders")
    assert held.count() == 100
    bt.from_pydict({"id": list(range(100, 150)), "amt": [1.0] * 50}).write.sql(
        "orders", uri=uri, mode="append", key_columns="id"
    )
    assert held.count() == 150, "a cached statistic answered a stale count()"
