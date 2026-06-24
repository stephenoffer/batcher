"""SQL / warehouse connector coverage — registration, query shaping, single-submit.

The connectors lazily import their optional drivers, so registration, identity,
and query rewriting work without any driver installed; tests that need a real
driver are gated with `pytest.importorskip` and skip cleanly otherwise. A real
ADBC SQLite round-trip runs when `adbc_driver_sqlite` is available.

The hard design rule — a SQL read issues exactly ONE query submission, with
schema and partitions coming from that single execute — is asserted with a cursor
spy that counts ``execute`` / ``adbc_execute_partitions`` calls.

Runs without the native engine — these exercise the Python IO layer only.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher._internal.errors import BackendError
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.formats.sql import (
    ADBCSink,
    ADBCSource,
    BigQuerySource,
    ClickHouseSource,
    ConnectorXSource,
    DatabricksSource,
    ODBCSource,
    SnowflakeSink,
    SnowflakeSource,
)
from batcher.io.formats.sql._common import (
    apply_predicate,
    apply_projection,
    require_module,
    wrap_subquery,
)


def _sorted_rows(table: pa.Table) -> list[dict]:
    return sorted(table.to_pylist(), key=lambda r: tuple(str(r[c]) for c in table.column_names))


# --- registration ------------------------------------------------------------


def test_connectors_registered() -> None:
    for name in ("adbc", "connectorx", "snowflake", "databricks", "bigquery", "clickhouse", "odbc"):
        assert name in SOURCES
    assert SOURCES.get("adbc") is ADBCSource
    assert SOURCES.get("connectorx") is ConnectorXSource
    assert SOURCES.get("snowflake") is SnowflakeSource
    assert SOURCES.get("databricks") is DatabricksSource
    assert SOURCES.get("bigquery") is BigQuerySource
    assert SOURCES.get("clickhouse") is ClickHouseSource
    assert SOURCES.get("odbc") is ODBCSource
    assert SINKS.get("adbc") is ADBCSink
    assert SINKS.get("snowflake") is SnowflakeSink


# --- query shaping (no driver, no rows) --------------------------------------


def test_wrap_subquery_table_and_query() -> None:
    assert "SELECT * FROM t" in wrap_subquery(None, table="t")  # type: ignore[arg-type]
    assert "AS _bc" in wrap_subquery("SELECT 1")


def test_apply_projection_rewrites_select_list() -> None:
    sql = apply_projection("SELECT * FROM t", ["a", "b"])
    assert sql.startswith("SELECT a, b FROM (")
    assert apply_projection("SELECT * FROM t", None).startswith("SELECT * FROM (")


def test_apply_predicate_appends_where() -> None:
    assert apply_predicate("SELECT * FROM t", None) == "SELECT * FROM t"
    assert "WHERE x > 1" in apply_predicate("SELECT * FROM t", "x > 1")


# --- identity / construction without a backend -------------------------------


def test_identity_does_not_require_backend() -> None:
    assert ADBCSource(driver="d", db_kwargs={}, table="t").identity() == "adbc:d:t"
    assert ConnectorXSource("SELECT 1", "mysql://h/db").identity() == "connectorx:SELECT 1"
    assert SnowflakeSource("SELECT 1", {"account": "a"}).identity() == "snowflake:SELECT 1"
    assert ClickHouseSource("SELECT 1", host="h").identity() == "clickhouse:h:SELECT 1"
    assert ODBCSource("SELECT 1", dsn="d").identity() == "odbc:d:SELECT 1"
    assert BigQuerySource(project="p", table="d.s.t").identity() == "bigquery:p:d.s.t"
    dbx = DatabricksSource(table="c.s.t", workspace="w", token="x")
    assert dbx.identity() == "databricks:c.s.t"


def test_construction_validation() -> None:
    with pytest.raises(BackendError):
        ADBCSource(driver="d", db_kwargs={})  # neither query nor table
    with pytest.raises(BackendError):
        BigQuerySource(project="p")  # neither query nor table
    with pytest.raises(BackendError):
        ODBCSource("SELECT 1")  # neither dsn nor connection_string
    with pytest.raises(BackendError):
        DatabricksSource()  # neither lakehouse nor warehouse config


# --- missing-driver errors are typed + actionable ----------------------------


def test_require_module_missing_raises_actionable() -> None:
    with pytest.raises(BackendError, match=r"\[sql\]"):
        require_module("definitely_not_a_real_module_xyz", extra="sql")


@pytest.mark.parametrize(
    ("make_source", "extra"),
    [
        (lambda: ConnectorXSource("SELECT 1", "mysql://h/db").schema(), "connectorx"),
        (lambda: SnowflakeSource("SELECT 1", {"account": "a"}).schema(), "snowflake"),
        (lambda: ClickHouseSource("SELECT 1", host="h").schema(), "clickhouse"),
        (lambda: ODBCSource("SELECT 1", dsn="d").schema(), "odbc"),
    ],
)
def test_missing_backend_raises_for_each_connector(make_source, extra, monkeypatch) -> None:
    import builtins

    blocked = {
        "connectorx": "connectorx",
        "snowflake": "snowflake",
        "clickhouse": "clickhouse_connect",
        "odbc": "turbodbc",
    }[extra]
    real_import = builtins.__import__

    def _block(name, *args, **kwargs):
        if name == blocked or name.startswith(blocked + "."):
            raise ImportError(f"no {blocked}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block)
    with pytest.raises(BackendError, match=rf"\[{extra}\]"):
        make_source()


# --- single-submission semantics (cursor spy) --------------------------------


class _SpyCursor:
    """A minimal ADBC-DBAPI cursor spy counting query submissions."""

    def __init__(self, log: dict[str, int]) -> None:
        self._log = log

    def execute(self, sql: str) -> None:
        self._log["execute"] = self._log.get("execute", 0) + 1

    def fetch_arrow_table(self) -> pa.Table:
        return pa.table({"id": [1, 2], "v": ["a", "b"]})

    def adbc_execute_partitions(self, sql: str):
        self._log["partitions"] = self._log.get("partitions", 0) + 1
        return ([b"desc-0", b"desc-1"], pa.schema([("id", pa.int64())]), -1)


class _SpyConn:
    def __init__(self, log: dict[str, int]) -> None:
        self._log = log

    def cursor(self) -> _SpyCursor:
        return _SpyCursor(self._log)

    def close(self) -> None:
        pass


def test_adbc_single_query_submission(monkeypatch) -> None:
    """A non-partitioned ADBC read submits the query exactly once per read."""
    log: dict[str, int] = {}

    def _fake_connect(driver, db_kwargs, conn_kwargs):
        return _SpyConn(log)

    monkeypatch.setattr("batcher.io.formats.sql.adbc._connect", _fake_connect)
    src = ADBCSource(driver="d", db_kwargs={}, query="SELECT * FROM t")
    rows = pa.Table.from_batches(src.read())
    assert rows.num_rows == 2
    # read() builds splits (no submission) then submits once on the single split.
    assert log.get("execute") == 1
    assert log.get("partitions", 0) == 0


def test_adbc_flightsql_partitions_single_submission(monkeypatch) -> None:
    """Partitioned ADBC issues ONE adbc_execute_partitions, one split per desc."""
    log: dict[str, int] = {}

    def _fake_connect(driver, db_kwargs, conn_kwargs):
        return _SpyConn(log)

    monkeypatch.setattr("batcher.io.formats.sql.adbc._connect", _fake_connect)
    src = ADBCSource(driver="d", db_kwargs={}, query="SELECT 1", partition=True)
    splits = src.splits()
    assert len(splits) == 2
    assert log.get("partitions") == 1  # exactly one submission for all partitions
    assert log.get("execute", 0) == 0


def test_adbc_partition_split_is_picklable() -> None:
    import pickle

    from batcher.io.formats.sql.adbc import _ADBCPartitionSplit

    split = _ADBCPartitionSplit("driver", {"uri": "x"}, None, b"desc", 0)
    restored = pickle.loads(pickle.dumps(split))
    assert restored.descriptor == b"desc"
    assert restored.identity() == "adbc-part:driver:0"


# --- real ADBC SQLite round-trip (gated) -------------------------------------


def test_adbc_sqlite_roundtrip(tmp_path) -> None:
    pytest.importorskip("adbc_driver_manager")
    sqlite = pytest.importorskip("adbc_driver_sqlite")

    db_path = str(tmp_path / "t.db")
    driver = sqlite.__name__
    db_kwargs = {"uri": db_path}

    table = pa.table({"id": [1, 2, 3], "v": ["a", "b", "c"]})
    ADBCSink(driver=driver, db_kwargs=db_kwargs, mode="create").write(table, "people")

    src = ADBCSource(driver=driver, db_kwargs=db_kwargs, table="people")
    out = pa.Table.from_batches(src.read())
    assert _sorted_rows(out) == _sorted_rows(table)

    # Projection rewrites the SELECT list and only that column comes back.
    proj = pa.Table.from_batches(src.read(projection=["id"]))
    assert proj.column_names == ["id"]
    assert proj.num_rows == 3


def test_adbc_sqlite_iter_batches(tmp_path) -> None:
    pytest.importorskip("adbc_driver_manager")
    sqlite = pytest.importorskip("adbc_driver_sqlite")

    db_path = str(tmp_path / "t2.db")
    driver = sqlite.__name__
    db_kwargs = {"uri": db_path}
    table = pa.table({"id": [1, 2]})
    ADBCSink(driver=driver, db_kwargs=db_kwargs, mode="create").write(table, "nums")

    src = ADBCSource(driver=driver, db_kwargs=db_kwargs, query="SELECT id FROM nums")
    batches = list(src.iter_batches())
    assert sum(b.num_rows for b in batches) == 2
