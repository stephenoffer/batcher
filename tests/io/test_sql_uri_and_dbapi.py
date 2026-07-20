"""Connection-URI routing, DB-API 2.0 reads, and the pushdown-ordering regression.

Three things are pinned here, all of which were live defects:

1. `push_down` composed the projection *outside* the predicate, so a query that pushed
   both — the common `select(a).filter(b == …)` shape — referenced a column the
   projection had already removed. That is a hard ``no such column`` error from the
   server, on five connectors at once, not a slow query. `sqlite3` is a real SQL engine
   from the standard library, so these assertions run the generated SQL rather than
   pattern-matching it: a string assertion would not have caught the bug, because the
   *string* looked perfectly reasonable.

2. `bt.read.sql(query, uri=…)` raised `TypeError` — `ADBCSource` had no `uri` parameter
   at all, while the docstring and the docs both advertised one. The doctest that would
   have caught it was marked ``+SKIP``.

3. `DBAPISource` reads any PEP 249 driver at batch granularity. `sqlite3` is again the
   oracle: it ships with Python, so this is a genuine end-to-end database round trip
   with no optional dependency and no mocking.
"""

from __future__ import annotations

import pickle
import re
import sqlite3

import pyarrow as pa
import pytest

from batcher._internal.errors import BackendError
from batcher.io.formats.sql._common import push_down
from batcher.io.formats.sql.adbc import ADBCSource
from batcher.io.formats.sql.dbapi import DBAPISource
from batcher.io.formats.sql.partition import range_predicates
from batcher.io.formats.sql.uri import known_schemes, parse_uri, redact_uri

pytestmark = pytest.mark.unit

#: A predicate on `country`, in the IR shape Kyber pushes to a scan.
_COUNTRY_IS_US = {
    "e": "binary",
    "op": "eq",
    "left": {"e": "col", "name": "country"},
    "right": {"e": "lit", "value": {"str": "US"}},
}


@pytest.fixture
def db(tmp_path):
    """A real SQLite database with nulls, duplicates, and a filterable column."""
    path = tmp_path / "warehouse.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE ev (id INTEGER, country TEXT, amt REAL)")
    conn.executemany(
        "INSERT INTO ev VALUES (?, ?, ?)",
        [(1, "US", 9.5), (2, "FR", 3.0), (3, "US", None), (4, None, 1.5)],
    )
    conn.commit()
    conn.close()
    return path


# --- 1. push_down ordering ------------------------------------------------------


@pytest.mark.parametrize(
    ("predicate", "projection", "expected"),
    [
        (_COUNTRY_IS_US, ["id"], [(1,), (3,)]),
        (_COUNTRY_IS_US, None, [(1, "US", 9.5), (3, "US", None)]),
        (None, ["id"], [(1,), (2,), (3,), (4,)]),
        (None, None, [(1, "US", 9.5), (2, "FR", 3.0), (3, "US", None), (4, None, 1.5)]),
    ],
    ids=["projection+predicate", "predicate-only", "projection-only", "neither"],
)
def test_pushed_sql_executes_and_is_correct(db, predicate, projection, expected) -> None:
    """The generated SQL runs on a real engine and returns the right rows.

    The first case is the regression: filtering on `country` while projecting only `id`
    used to emit ``SELECT id FROM (…) WHERE country = 'US'`` and fail outright.
    """
    sql = push_down(None, predicate, projection, table="ev")
    conn = sqlite3.connect(db)
    try:
        assert conn.execute(sql).fetchall() == expected
    finally:
        conn.close()


def test_pushed_sql_filters_below_the_projection(db) -> None:
    """The predicate is applied beneath the projection, not layered on top of it."""
    sql = push_down(None, _COUNTRY_IS_US, ["id"], table="ev")
    assert sql.index("WHERE country = 'US'") > sql.index("SELECT id")


def test_pushdown_works_through_a_user_subquery(db) -> None:
    """A user-supplied query, not just a `table=`, is filtered and projected correctly."""
    sql = push_down("SELECT id, country FROM ev", _COUNTRY_IS_US, ["id"])
    conn = sqlite3.connect(db)
    try:
        assert conn.execute(sql).fetchall() == [(1,), (3,)]
    finally:
        conn.close()


# --- 2. connection URIs ---------------------------------------------------------


@pytest.mark.parametrize(
    ("uri", "backend", "driver"),
    [
        ("postgresql://h/db", "adbc", "adbc_driver_postgresql"),
        ("postgres://h/db", "adbc", "adbc_driver_postgresql"),
        ("postgresql+psycopg2://h/db", "adbc", "adbc_driver_postgresql"),
        ("sqlite:///local.db", "adbc", "adbc_driver_sqlite"),
        ("grpc+tls://flight:443", "adbc", "adbc_driver_flightsql"),
        ("mysql://h/db", "connectorx", None),
        ("mysql+pymysql://h/db", "connectorx", None),
        ("redshift://h/db", "connectorx", None),
    ],
)
def test_uri_routes_to_the_right_backend(uri, backend, driver) -> None:
    parsed = parse_uri(uri)
    assert (parsed.backend, parsed.driver) == (backend, driver)


def test_uri_keeps_the_transport_suffix_but_drops_the_dbapi_suffix() -> None:
    """``+psycopg2`` names a DBAPI driver and is noise here; ``+tls`` names a transport."""
    assert parse_uri("postgresql+psycopg2://h/db").scheme == "postgresql"
    assert parse_uri("grpc+tls://h:443").scheme == "grpc+tls"


def test_inline_password_is_lifted_out_of_the_carried_uri() -> None:
    """The URI that reaches logs and splits must not contain the password."""
    parsed = parse_uri("postgresql://alice:hunter2@db:5432/app")
    assert parsed.password == "hunter2"
    assert "hunter2" not in parsed.uri
    assert parsed.uri == "postgresql://alice@db:5432/app"
    assert parsed.username == "alice"
    assert parsed.database == "app"


def test_explicit_password_overrides_an_inline_one() -> None:
    parsed = parse_uri("postgresql://alice:inline@db/app", password="env:PGPASSWORD")
    assert parsed.password == "env:PGPASSWORD"
    assert "inline" not in parsed.uri


def test_query_string_becomes_options() -> None:
    assert parse_uri("snowflake://acct/db?warehouse=WH&role=R").options == {
        "warehouse": "WH",
        "role": "R",
    }


def test_file_backed_scheme_keeps_its_path_as_a_locator() -> None:
    """``sqlite:///data.db`` addresses a file; that path is not a database name."""
    parsed = parse_uri("sqlite:///data.db")
    assert parsed.database is None
    assert parsed.uri == "sqlite:///data.db"


@pytest.mark.parametrize("bad", ["not-a-uri", "", "mongodb://h/db", "ftp://h/x"])
def test_unroutable_uri_raises_with_guidance(bad) -> None:
    with pytest.raises(BackendError):
        parse_uri(bad)


def test_unsupported_scheme_error_lists_what_is_supported() -> None:
    """A scheme with no route at all gets the full list of what can be routed."""
    with pytest.raises(BackendError, match="postgresql"):
        parse_uri("ftp://h/db")


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("mongodb://h/db", "bt.read.table('mongo'"),
        ("mongodb+srv://h/db", "bt.read.table('mongo'"),
        ("databricks://ws/t", "bt.read.databricks("),
        ("teradata://h/db", "bt.read.table('odbc'"),
    ],
)
def test_a_readable_store_with_no_uri_spelling_names_the_call_that_works(uri, expected) -> None:
    """Batcher reads these — just not from a URI. Say so, rather than listing SQL schemes.

    Answering a MongoDB URI with "supported schemes are postgresql, mysql, …" is true and
    useless: it reads as "Batcher cannot do MongoDB", when the only problem is spelling.
    """
    with pytest.raises(BackendError, match=re.escape(expected)):
        parse_uri(uri)


@pytest.mark.parametrize(
    ("uri", "backend"),
    [
        # PostgreSQL wire protocol — `adbc_driver_postgresql` connects to all of these.
        ("cockroachdb://h:26257/db", "adbc"),
        ("cockroach://h:26257/db", "adbc"),
        ("timescaledb://h/metrics", "adbc"),
        ("alloydb://h/db", "adbc"),
        ("greenplum://h/db", "adbc"),
        ("yugabytedb://h/db", "adbc"),
        ("yugabyte://h/db", "adbc"),
        ("risingwave://h/db", "adbc"),
        ("materialize://h/db", "adbc"),
        ("questdb://h:8812/qdb", "adbc"),
        ("crate://h:5432/doc", "adbc"),
        ("cratedb://h:5432/doc", "adbc"),
        # MySQL wire protocol — ConnectorX's MySQL reader connects to all of these.
        ("singlestore://h/db", "connectorx"),
        ("memsql://h/db", "connectorx"),
        ("tidb://h:4000/db", "connectorx"),
        ("starrocks://h:9030/db", "connectorx"),
        ("doris://h:9030/db", "connectorx"),
        ("percona://h/db", "connectorx"),
    ],
)
def test_wire_compatible_databases_route_to_the_compatible_driver(uri, backend) -> None:
    """A database that speaks a supported wire protocol is readable, not "unsupported".

    Wire compatibility is all the driver needs: it connects, executes, and gets Arrow
    back. Without these entries a user holding a working ``cockroachdb://`` URI was told
    the database was unsupported, when the only thing missing was the scheme name.
    """
    assert parse_uri(uri).backend == backend


def test_postgres_wire_databases_all_use_the_postgres_driver() -> None:
    assert parse_uri("cockroachdb://h/db").driver == "adbc_driver_postgresql"
    assert parse_uri("timescaledb://h/db").driver == parse_uri("postgresql://h/db").driver


def test_no_scheme_is_claimed_by_two_backends_at_once() -> None:
    """A scheme in both maps would route by whichever check ran first — silently.

    This nearly happened: adding `presto` to the ConnectorX set while it was also listed
    as an ODBC-only alternative made the alternative unreachable dead guidance. Presto
    and Trino diverged after the fork and ConnectorX ships a *Trino* reader, so routing
    Presto there would have been a guess presented as support.
    """
    from batcher.io.formats.sql.uri import (
        _ADBC_DRIVERS,
        _ALTERNATIVE_ROUTES,
        _CONNECTORX_SCHEMES,
    )

    assert not set(_ADBC_DRIVERS) & _CONNECTORX_SCHEMES
    routable = set(_ADBC_DRIVERS) | _CONNECTORX_SCHEMES
    assert not routable & set(_ALTERNATIVE_ROUTES), (
        "a routable scheme must not also be listed as unroutable-with-an-alternative"
    )


def test_presto_is_still_directed_to_odbc() -> None:
    with pytest.raises(BackendError, match=re.escape("bt.read.table('odbc'")):
        parse_uri("presto://h:8080/hive")


def test_redact_uri() -> None:
    assert redact_uri("postgresql://a:pw@h:5432/db") == "postgresql://a:***@h:5432/db"
    assert redact_uri("postgresql://h/db") == "postgresql://h/db"


def test_known_schemes_are_sorted_and_nonempty() -> None:
    schemes = known_schemes()
    assert schemes == tuple(sorted(schemes))
    assert "postgresql" in schemes and "mysql" in schemes


# --- ADBCSource accepts a URI ---------------------------------------------------


def test_adbc_source_accepts_a_uri() -> None:
    """The regression: this raised `TypeError: unexpected keyword argument 'uri'`."""
    source = ADBCSource(query="SELECT 1", uri="postgresql://alice@db:5432/app")
    assert source.driver == "adbc_driver_postgresql"
    assert source.db_kwargs["uri"] == "postgresql://alice@db:5432/app"


def test_adbc_uri_password_reference_survives_pickling_unresolved(monkeypatch) -> None:
    """An `env:` reference must still be a reference on the far side of a pickle."""
    monkeypatch.setenv("PGPASSWORD", "s3cret")
    source = ADBCSource(query="SELECT 1", uri="postgresql://a@h/db", password="env:PGPASSWORD")
    blob = pickle.dumps(source)
    assert b"s3cret" not in blob
    assert pickle.loads(blob).db_kwargs["password"] == "env:PGPASSWORD"


def test_adbc_credentials_are_absent_from_repr() -> None:
    source = ADBCSource(query="SELECT 1", uri="postgresql://a:literal@h/db")
    assert "literal" not in repr(source)


def test_explicit_driver_still_wins_over_the_uri() -> None:
    source = ADBCSource(query="SELECT 1", uri="postgresql://h/db", driver="custom_driver")
    assert source.driver == "custom_driver"


def test_adbc_uri_with_no_adbc_driver_names_the_alternative() -> None:
    with pytest.raises(BackendError, match="connectorx"):
        ADBCSource(query="SELECT 1", uri="mysql://h/db")


def test_adbc_without_any_connection_raises() -> None:
    with pytest.raises(BackendError, match="uri="):
        ADBCSource(query="SELECT 1")


def test_adbc_legacy_positional_construction_still_works() -> None:
    """`driver=`/`db_kwargs=` predates `uri=` and must keep working unchanged."""
    source = ADBCSource("adbc_driver_sqlite", {"uri": ":memory:"}, "SELECT 1")
    assert source.driver == "adbc_driver_sqlite"


# --- 3. DB-API 2.0 --------------------------------------------------------------


def _source(db, **kw) -> DBAPISource:
    return DBAPISource("sqlite3", {"database": str(db)}, **kw)


def test_dbapi_reads_a_real_database(db) -> None:
    table = pa.Table.from_batches(_source(db, table="ev").read())
    assert table.to_pydict() == {
        "id": [1, 2, 3, 4],
        "country": ["US", "FR", "US", None],
        "amt": [9.5, 3.0, None, 1.5],
    }


def test_dbapi_infers_types_from_data_when_the_driver_reports_none(db) -> None:
    """`sqlite3` reports no type codes, so types come from a real batch — not a guess."""
    assert _source(db, table="ev").schema() == pa.schema(
        [("id", pa.int64()), ("country", pa.string()), ("amt", pa.float64())]
    )


def test_dbapi_batches_at_batch_size_and_keeps_one_schema(db) -> None:
    """Every batch shares batch 1's schema, so an all-null later batch cannot retype."""
    batches = _source(db, table="ev", batch_size=2).read()
    assert [b.num_rows for b in batches] == [2, 2]
    assert len({b.schema for b in batches}) == 1


def test_dbapi_pushes_the_predicate_to_the_server(db) -> None:
    rows = pa.Table.from_batches(_source(db, table="ev").read(predicate=_COUNTRY_IS_US))
    assert rows.column("id").to_pylist() == [1, 3]


def test_dbapi_applies_projection(db) -> None:
    table = pa.Table.from_batches(_source(db, table="ev").read(projection=["country"]))
    assert table.column_names == ["country"]


def test_dbapi_split_carries_the_pushdown_and_is_picklable(db) -> None:
    split = _source(db, table="ev").splits(predicate=_COUNTRY_IS_US, projection=["id"])[0]
    assert "WHERE country = 'US'" in split.sql
    assert pickle.loads(pickle.dumps(split)).identity() == split.identity()


def test_dbapi_empty_relation_keeps_its_column_names(db) -> None:
    """No rows means no inferable types, but the columns still exist."""
    schema = _source(db, query="SELECT id, country FROM ev WHERE 0 = 1").schema()
    assert schema.names == ["id", "country"]


def test_dbapi_schema_override_is_authoritative(db) -> None:
    declared = pa.schema([("id", pa.int32()), ("country", pa.string()), ("amt", pa.float32())])
    assert _source(db, table="ev", schema_override=declared).schema() == declared


def test_dbapi_iter_batches_streams(db) -> None:
    assert sum(b.num_rows for b in _source(db, table="ev", batch_size=1).iter_batches()) == 4


def test_dbapi_requires_query_or_table(db) -> None:
    with pytest.raises(BackendError, match="query="):
        _source(db)


def test_dbapi_rejects_a_nonsense_batch_size(db) -> None:
    with pytest.raises(BackendError, match="batch_size"):
        _source(db, table="ev", batch_size=0)


def test_dbapi_missing_driver_names_the_users_package() -> None:
    source = DBAPISource("definitely_not_a_driver", {}, table="t")
    with pytest.raises(BackendError, match="pip install definitely_not_a_driver"):
        source.read()


def test_dbapi_credentials_are_absent_from_repr(db) -> None:
    assert str(db) not in repr(_source(db, table="ev"))


def test_dbapi_is_registered() -> None:
    from batcher.io.formats.base import SOURCES

    assert SOURCES.get("dbapi") is DBAPISource


# --- 4. bulk parallel range partitioning ---------------------------------------


@pytest.fixture
def skewed_db(tmp_path):
    """A table whose keys include NULLs and values far outside any sane bound.

    Those are precisely the rows a range-partitioned read loses when the partition
    predicates are merely *disjoint* rather than disjoint **and exhaustive**.
    """
    path = tmp_path / "skewed.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (id INTEGER, country TEXT)")
    keys = [-500, -1, 0, 1, 5, 50, 99, 100, 101, 5000]
    conn.executemany(
        "INSERT INTO t VALUES (?, ?)",
        [(k, "US" if k % 2 else "FR") for k in keys] + [(None, "US"), (None, "FR")],
    )
    conn.commit()
    conn.close()
    return path


def _all_rows(splits) -> list[dict]:
    rows = [row for split in splits for batch in split.read() for row in batch.to_pylist()]
    return sorted(rows, key=lambda r: (r.get("id") is None, r.get("id")))


@pytest.mark.parametrize("num_partitions", [1, 2, 3, 4, 8, 16])
def test_partitioned_read_equals_unpartitioned_read(skewed_db, num_partitions) -> None:
    """The invariant: partitioning changes parallelism, never the result set.

    Bounds of ``[0, 100]`` are deliberately wrong for this data — keys run from -500 to
    5000 and include NULLs. Every one of those rows must still come back exactly once.
    """
    source = DBAPISource("sqlite3", {"database": str(skewed_db)}, table="t")
    expected = _all_rows(source.splits())

    partitioned = DBAPISource(
        "sqlite3",
        {"database": str(skewed_db)},
        table="t",
        partition_on="id",
        lower_bound=0,
        upper_bound=100,
        num_partitions=num_partitions,
    )
    splits = partitioned.splits()
    assert len(splits) == num_partitions
    assert _all_rows(splits) == expected


def test_partitioning_composes_with_predicate_and_projection(skewed_db) -> None:
    """All three pushdowns at once — the shape most likely to compose wrongly."""
    source = DBAPISource(
        "sqlite3",
        {"database": str(skewed_db)},
        table="t",
        partition_on="id",
        lower_bound=0,
        upper_bound=100,
        num_partitions=4,
    )
    rows = _all_rows(source.splits(predicate=_COUNTRY_IS_US, projection=["id"]))
    unfiltered = _all_rows(DBAPISource("sqlite3", {"database": str(skewed_db)}, table="t").splits())
    assert rows == [{"id": r["id"]} for r in unfiltered if r["country"] == "US"]


def test_partition_fragment_is_parenthesized_against_the_predicate(skewed_db) -> None:
    """`A AND k < 5 OR k IS NULL` would bind as `(A AND …) OR (k IS NULL)`.

    That reads every NULL-keyed row into *every* split — duplication, not loss, and
    invisible unless the parentheses are asserted.
    """
    source = DBAPISource(
        "sqlite3",
        {"database": str(skewed_db)},
        table="t",
        partition_on="id",
        lower_bound=0,
        upper_bound=100,
        num_partitions=2,
    )
    sql = source.splits(predicate=_COUNTRY_IS_US)[0].sql
    assert "(country = 'US') AND (id < 50.0 OR id IS NULL)" in sql


def test_partition_on_requires_bounds() -> None:
    with pytest.raises(BackendError, match="lower_bound"):
        DBAPISource("sqlite3", {}, table="t", partition_on="id")


def test_adbc_range_partitions_when_the_driver_cannot(skewed_db) -> None:
    """ADBC falls back to range partitioning; each split carries its own slice."""
    source = ADBCSource(
        table="t",
        uri="postgresql://h/db",
        partition_on="id",
        lower_bound=0,
        upper_bound=100,
        num_partitions=4,
    )
    fragments = [split.sql for split in source.splits()]
    assert len(fragments) == 4
    assert "id < 25.0 OR id IS NULL" in fragments[0]
    assert "id >= 75.0" in fragments[-1]


def test_range_predicates_cover_every_row_exactly_once() -> None:
    """Unit-level: first partition unbounded below, last unbounded above, NULLs placed."""
    fragments = range_predicates("k", 0, 100, 3)
    assert fragments[0] == "k < 33.333333333333336 OR k IS NULL"
    assert fragments[-1] == "k >= 66.66666666666667"


def test_single_partition_means_no_filter() -> None:
    assert range_predicates("k", 0, 100, 1) == [None]


def test_degenerate_bounds_collapse_to_one_partition() -> None:
    """Equal bounds would make every cut point coincide — N-1 empty queries."""
    assert range_predicates("k", 5, 5, 8) == [None]


@pytest.mark.parametrize(("lo", "hi", "n"), [(0, 100, 0), (0, 100, -1), (100, 0, 4)])
def test_invalid_partitioning_is_rejected(lo, hi, n) -> None:
    with pytest.raises(BackendError):
        range_predicates("k", lo, hi, n)


# --- 5. streaming, and the distributed == single-node invariant ------------------


_PARTITION_SCHEMA = pa.schema([("id", pa.int64())])


class _RecordBatchStream:
    """A stand-in for Arrow's `RecordBatchReader`: a schema up front, batches on demand.

    The schema is available *without* consuming a batch — that is the whole property a
    streaming `schema()` depends on, so the fake has to model it rather than expose the
    schema only after the data has been read.
    """

    schema = _PARTITION_SCHEMA

    def __init__(self, log: list[str]) -> None:
        self._log = log

    def __iter__(self):
        for start in (0, 2):
            self._log.append("batch")
            yield pa.RecordBatch.from_arrays(
                [pa.array([start + 1, start + 2])], schema=_PARTITION_SCHEMA
            )


class _PartitionReader:
    """Stands in for the object an ADBC cursor returns for a FlightSQL partition."""

    def __init__(self, log: list[str]) -> None:
        self._log = log

    def fetch_arrow_table(self) -> pa.Table:
        self._log.append("fetch_arrow_table")
        return pa.table({"id": [1, 2, 3, 4]})

    def fetch_record_batch(self) -> _RecordBatchStream:
        self._log.append("fetch_record_batch")
        return _RecordBatchStream(self._log)


class _PartitionCursor:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    def adbc_read_partition(self, descriptor: bytes) -> _PartitionReader:
        return _PartitionReader(self._log)


class _PartitionConn:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    def cursor(self) -> _PartitionCursor:
        return _PartitionCursor(self._log)

    def close(self) -> None:
        self._log.append("close")


def _partition_split(monkeypatch, log: list[str]):
    from batcher.io.formats.sql.adbc import _ADBCPartitionSplit

    monkeypatch.setattr(
        "batcher.io.formats.sql.adbc._connect",
        lambda driver, db_kwargs, conn_kwargs: _PartitionConn(log),
    )
    return _ADBCPartitionSplit("d", {}, None, b"desc-0", 0)


def test_flightsql_partition_streams_rather_than_materializing(monkeypatch) -> None:
    """`iter_batches` on a partition must not pull the whole partition first.

    This was ``yield from self.read(...)``, so the streaming entry point called
    `fetch_arrow_table` and materialized an entire worker-sized partition before
    yielding anything — defeating every caller that chose `iter_batches` to avoid
    exactly that.
    """
    log: list[str] = []
    split = _partition_split(monkeypatch, log)

    first = next(iter(split.iter_batches()))

    assert first.num_rows == 2
    assert "fetch_record_batch" in log
    assert "fetch_arrow_table" not in log
    # Only the first batch was produced — the rest of the partition is still unread.
    assert log.count("batch") == 1


def test_flightsql_partition_schema_does_not_download_the_partition(monkeypatch) -> None:
    """A schema lookup must read the stream header, not the partition's rows."""
    log: list[str] = []
    split = _partition_split(monkeypatch, log)

    assert split.schema() == _PARTITION_SCHEMA
    assert "fetch_arrow_table" not in log
    assert log.count("batch") == 0


def test_flightsql_partition_closes_its_connection(monkeypatch) -> None:
    log: list[str] = []
    split = _partition_split(monkeypatch, log)
    list(split.iter_batches())
    assert log.count("close") == 1


def test_dbapi_iter_batches_is_lazy(db) -> None:
    """The first batch must arrive without the rest of the relation being read."""
    source = DBAPISource("sqlite3", {"database": str(db)}, table="ev", batch_size=1)
    batches = source.iter_batches()
    assert next(batches).num_rows == 1
    batches.close()


def test_partitioned_splits_survive_the_trip_to_a_worker(skewed_db) -> None:
    """Invariant: single-node == distributed.

    A split is a locator, not data — it is pickled and rebuilt on a worker that shares no
    state with the driver. Reading every split *after* a pickle round trip must reproduce
    the single-node relation exactly, NULL keys and out-of-bounds keys included.
    """
    single = _all_rows(DBAPISource("sqlite3", {"database": str(skewed_db)}, table="t").splits())

    partitioned = DBAPISource(
        "sqlite3",
        {"database": str(skewed_db)},
        table="t",
        partition_on="id",
        lower_bound=0,
        upper_bound=100,
        num_partitions=8,
    )
    shipped = [pickle.loads(pickle.dumps(split)) for split in partitioned.splits()]
    assert _all_rows(shipped) == single


# --- 6. identity: the key learned statistics are stored under ---------------------


def test_the_same_query_against_two_databases_is_two_relations() -> None:
    """Prod and staging must not share a learned-statistics key.

    Keyed on the query alone, ``SELECT * FROM orders`` against production and against
    staging were literally the same string. Kyber would then plan the thousand-row
    staging table using the billion-row production table's cardinalities — a bad plan
    from good code, with nothing anywhere reporting an error.
    """
    prod = DBAPISource("sqlite3", {"database": "/data/prod.db"}, table="orders")
    staging = DBAPISource("sqlite3", {"database": "/data/staging.db"}, table="orders")
    assert prod.identity() != staging.identity()

    live = ADBCSource(table="orders", uri="postgresql://prod-db:5432/app")
    spare = ADBCSource(table="orders", uri="postgresql://staging-db:5432/app")
    assert live.identity() != spare.identity()


def test_identity_is_stable_across_processes() -> None:
    """A per-process salt would silently disable the learned-stats loop entirely.

    `hash()` is salted per interpreter, so an identity built on it changes every run,
    every lookup misses, and the optimizer never reuses a statistic — while appearing
    to work. This asserts the digest is the content-addressed kind.
    """
    import subprocess
    import sys

    script = (
        "from batcher.io.formats.sql.dbapi import DBAPISource;"
        "print(DBAPISource('sqlite3', {'database': '/d.db'}, table='t').identity())"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(runs) == 1


def test_rotating_a_password_does_not_orphan_the_statistics() -> None:
    """Credentials are authentication, not identity — the relation is unchanged."""
    before = DBAPISource("sqlite3", {"database": "/d.db", "password": "old"}, table="t")
    after = DBAPISource("sqlite3", {"database": "/d.db", "password": "new"}, table="t")
    assert before.identity() == after.identity()


def test_identity_never_contains_a_credential() -> None:
    """`identity()` is logged and stored; a secret in it would outlive the process."""
    source = DBAPISource("sqlite3", {"database": "/d.db", "password": "hunter2"}, table="t")
    assert "hunter2" not in source.identity()


def test_a_direct_read_does_not_fan_out_into_range_queries(skewed_db) -> None:
    """Partitioning is a distribution concern; a local read should stay one query.

    Range partitioning exists to give N machines a query each. Run on one machine it is
    pure loss — the same rows arrive, having cost N round trips and N server-side planner
    invocations instead of one. So `read`/`iter_batches` skip it and `splits` applies it.

    ADBC and DB-API disagreed here: DB-API issued one query while ADBC iterated its
    splits, so the same partitioned source cost 1 query on one backend and N on the other.
    """
    partitioned = DBAPISource(
        "sqlite3",
        {"database": str(skewed_db)},
        table="t",
        partition_on="id",
        lower_bound=0,
        upper_bound=100,
        num_partitions=4,
    )
    assert len(partitioned.splits()) == 4

    adbc = ADBCSource(
        table="t",
        uri="postgresql://h/db",
        partition_on="id",
        lower_bound=0,
        upper_bound=100,
        num_partitions=4,
    )
    assert len(adbc.splits()) == 4
    # ...but the direct read path is a single query on both.
    assert len(adbc._direct_splits(None, None)) == 1


def test_a_direct_read_still_pushes_the_predicate_down(skewed_db) -> None:
    """Skipping the fan-out must not also skip the pushdown."""
    adbc = ADBCSource(
        table="t",
        uri="postgresql://h/db",
        partition_on="id",
        lower_bound=0,
        upper_bound=100,
        num_partitions=4,
    )
    sql = adbc._direct_splits(_COUNTRY_IS_US, ["id"])[0].sql
    assert "WHERE country = 'US'" in sql
    assert "SELECT id" in sql


def test_every_partitionable_connector_spells_it_the_same_way() -> None:
    """One vocabulary across backends, matching Spark's JDBC reader.

    ConnectorX used to call this `partition_num` while ADBC and DB-API called it
    `num_partitions` — the same concept under two names, which is exactly the
    inconsistency a user hits when moving a read from one database to another.
    """
    import inspect

    from batcher.io.formats.sql.connectorx import ConnectorXSource

    for source in (ADBCSource, DBAPISource, ConnectorXSource):
        params = inspect.signature(source.__init__).parameters
        assert "partition_on" in params, source.__name__
        assert "num_partitions" in params, source.__name__
        assert "partition_num" not in params, source.__name__
