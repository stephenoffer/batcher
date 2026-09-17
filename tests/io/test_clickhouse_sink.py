"""`ds.write.clickhouse`, held to its contract with the driver through a recording fake.

No ClickHouse server is reachable from a test box, so this pins what the sink asks of
``clickhouse_connect``: one client per write with the connection details it was given, a
``TRUNCATE`` only for ``mode="overwrite"``, the rows handed over as one Arrow table (so types
and nulls arrive as Arrow, never as Python rows), and a secret reference resolved only when
the client is opened. The API calls are the ones the reader in the same module already makes
(`get_client`, `close`), plus ``insert_arrow`` and ``command``.
"""

from __future__ import annotations

import sys
import types

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import BackendError, FormatError, PlanError
from batcher.io.formats.base import SINKS
from batcher.io.formats.sql.clickhouse import ClickHouseSink

pytestmark = pytest.mark.io


@pytest.fixture
def driver(monkeypatch):
    calls: list[tuple] = []

    class Client:
        def command(self, sql):
            calls.append(("command", sql))

        def insert_arrow(self, table, arrow_table, database=None):
            calls.append(("insert_arrow", table, arrow_table, database))

        def close(self):
            calls.append(("close",))

    module = types.ModuleType("clickhouse_connect")

    def get_client(**params):
        calls.append(("get_client", params))
        return Client()

    module.get_client = get_client
    monkeypatch.setitem(sys.modules, "clickhouse_connect", module)
    return calls


ROWS = {"id": [1, 2, None], "name": ["a", None, "c" * 1_000_000], "tags": [["x"], [], None]}


def test_the_sink_is_registered():
    assert SINKS.get("clickhouse") is ClickHouseSink


def test_append_inserts_one_arrow_table_with_types_and_nulls(driver):
    ds = bt.from_pydict(ROWS)
    manifest = ds.write.clickhouse("orders", host="ch", port=8123, database="shop")
    assert [f.rows for f in manifest.files] == [3]
    assert [c[0] for c in driver] == ["get_client", "insert_arrow", "close"]
    assert driver[0][1] == {"host": "ch", "username": "default", "port": 8123, "database": "shop"}
    _, table_name, inserted, database = driver[1]
    assert (table_name, database) == ("orders", "shop")
    assert isinstance(inserted, pa.Table)
    assert inserted.schema == ds.schema
    assert inserted.to_pydict() == ROWS


def test_overwrite_truncates_the_quoted_table_first(driver):
    bt.from_pydict({"id": [1]}).write.clickhouse("db.orders", host="ch", mode="overwrite")
    assert [c[0] for c in driver] == ["get_client", "command", "insert_arrow", "close"]
    assert driver[1] == ("command", 'TRUNCATE TABLE "db"."orders"')


def test_an_empty_overwrite_truncates_and_inserts_nothing(driver):
    ds = bt.from_pydict({"id": [1]}).filter(bt.col("id") > 5)
    ds.write.clickhouse("orders", host="ch", mode="overwrite")
    assert [c[0] for c in driver] == ["get_client", "command", "close"]


def test_a_password_reference_is_resolved_only_when_the_client_opens(driver, monkeypatch):
    monkeypatch.setenv("CH_TEST_PASSWORD", "s3cret")
    sink = ClickHouseSink(host="ch", password="env:CH_TEST_PASSWORD")
    assert "s3cret" not in repr(sink) and "CH_TEST_PASSWORD" not in repr(sink)
    sink.write(pa.table({"id": [1]}), "orders")
    assert driver[0][1]["password"] == "s3cret"


def test_the_client_is_closed_when_the_insert_fails(monkeypatch):
    closed: list[bool] = []

    class Failing:
        def insert_arrow(self, *args, **kwargs):
            raise RuntimeError("server said no")

        def close(self):
            closed.append(True)

    module = types.ModuleType("clickhouse_connect")
    module.get_client = lambda **_: Failing()
    monkeypatch.setitem(sys.modules, "clickhouse_connect", module)
    with pytest.raises(RuntimeError, match="server said no"):
        ClickHouseSink(host="ch").write(pa.table({"id": [1]}), "orders")
    assert closed == [True]


def test_a_multi_shard_overwrite_is_refused(driver):
    sink = ClickHouseSink(host="ch", mode="overwrite")
    table = pa.table({"id": [1]})
    assert len(sink.write_partitioned(table, "orders", file_index=0)) == 1
    with pytest.raises(BackendError, match="distributed write"):
        sink.write_partitioned(table, "orders", file_index=1)


def test_modes_and_options_are_checked_before_connecting(driver):
    ds = bt.from_pydict({"id": [1]})
    with pytest.raises(PlanError, match="mode='append' or mode='overwrite'"):
        ds.write.clickhouse("orders", host="ch", mode="error")
    with pytest.raises(FormatError, match="Did you mean 'host'"):
        ds.write.clickhouse("orders", hots="ch")
    with pytest.raises(PlanError, match="partition_by"):
        ds.write.clickhouse("orders", host="ch", partition_by=["id"])
    assert driver == []


@pytest.mark.parametrize("fmt", ["snowflake", "dbapi", "adbc"])
@pytest.mark.parametrize("mode", ["error", "ignore"])
def test_a_protective_mode_on_any_table_sink_is_refused_not_overwritten(fmt, mode) -> None:
    """`error`/`ignore` cannot be checked against a table, so no table sink may treat them as
    overwrite; before this was refused, `write.snowflake(t, mode="error")` replaced the table."""
    ds = bt.from_pydict({"id": [1]})
    with pytest.raises(PlanError, match="cannot check whether its table exists"):
        ds.write(f"orders_{fmt}", fmt, mode=mode)
