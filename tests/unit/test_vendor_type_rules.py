"""Vendor type rules: declared conversions and precise refusals, against fake drivers.

Each rule is checked end to end through the source that applies it — `DBAPISource` with a
fake PEP 249 driver returning the Python objects the real driver documents, and
`ConnectorXSource` with a fake ``connectorx`` returning the Arrow it would — and the refusals
are checked for naming the column and the fix rather than an Arrow internal.
"""

from __future__ import annotations

import datetime
import decimal
import sys
import types

import pyarrow as pa
import pytest

from batcher._internal.errors import BackendError
from batcher.io.formats.sql.connectorx import ConnectorXSource
from batcher.io.formats.sql.dbapi import DBAPISource
from batcher.io.formats.sql.vendors.connect import prepare_connection
from batcher.io.formats.sql.vendors.types import conform, explain_failure

pytestmark = pytest.mark.unit


def _fake_driver(monkeypatch, name: str, rows: list[tuple], columns: list[str]) -> list:
    """Install a PEP 249 module `name` whose cursor returns `rows` once."""
    connections: list = []

    class _Cursor:
        description = None

        def execute(self, sql, params=None):
            self.description = [(c, None) for c in columns]
            self._rows = list(rows)

        def fetchmany(self, n):
            out, self._rows = self._rows[:n], self._rows[n:]
            return out

        def close(self):
            pass

    class _Conn:
        outputtypehandler = None

        def cursor(self):
            return _Cursor()

        def close(self):
            pass

    def connect(**kwargs):
        conn = _Conn()
        conn.kwargs = kwargs
        connections.append(conn)
        return conn

    module = types.ModuleType(name)
    module.connect = connect
    module.paramstyle = "pyformat"
    monkeypatch.setitem(sys.modules, name, module)
    return connections


def _read(source: DBAPISource) -> pa.Table:
    return pa.Table.from_batches(source.read())


# --- MySQL zero dates (PyMySQL returns the illegal date as the string it read) ---------


def test_a_zero_date_is_refused_naming_the_column_and_the_fix(monkeypatch):
    _fake_driver(monkeypatch, "fakemysql", [(datetime.date(2024, 1, 2),), ("0000-00-00",)], ["d"])
    source = DBAPISource(module="fakemysql", query="SELECT d FROM t")
    with pytest.raises(
        BackendError, match=r"column 'd' holds the MySQL zero date.*zero_dates='null'"
    ):
        _read(source)


def test_zero_dates_null_reads_it_as_null(monkeypatch):
    _fake_driver(
        monkeypatch, "fakemysql", [(datetime.date(2024, 1, 2),), ("0000-00-00 00:00:00",)], ["d"]
    )
    table = _read(DBAPISource(module="fakemysql", query="SELECT d FROM t", zero_dates="null"))
    assert table.column("d").to_pylist() == [datetime.date(2024, 1, 2), None]
    assert pa.types.is_date32(table.schema.field("d").type)


def test_an_unknown_policy_is_refused():
    with pytest.raises(BackendError, match="zero_dates='maybe'"):
        DBAPISource(module="sqlite3", query="SELECT 1", zero_dates="maybe")


# --- MySQL BIGINT UNSIGNED --------------------------------------------------------------


def test_unsigned_overflow_is_refused_naming_the_column():
    batch = pa.record_batch({"id": pa.array([1, 2**64 - 1], pa.uint64())})
    with pytest.raises(BackendError, match=r"column 'id'.*unsigned='decimal'"):
        conform(batch)


def test_unsigned_in_range_passes_unchanged():
    batch = pa.record_batch({"id": pa.array([1, 2], pa.uint64())})
    assert conform(batch) is not None
    assert conform(batch).schema == batch.schema


def test_unsigned_decimal_is_exact_over_the_whole_range():
    batch = pa.record_batch({"id": pa.array([0, 2**64 - 1, None], pa.uint64()), "x": [1, 2, 3]})
    out = conform(batch, unsigned="decimal")
    assert out.schema.field("id").type == pa.decimal128(20, 0)
    assert out.column("id").to_pylist() == [0, decimal.Decimal(2**64 - 1), None]
    assert out.column("x").to_pylist() == [1, 2, 3]


def _fake_connectorx(monkeypatch, table: pa.Table) -> None:
    cx = types.ModuleType("connectorx")
    cx.read_sql = lambda uri, query, **kw: table
    monkeypatch.setitem(sys.modules, "connectorx", cx)


def test_connectorx_applies_the_unsigned_rule_to_reads_and_the_schema_probe(monkeypatch):
    big = pa.table({"id": pa.array([2**64 - 1], pa.uint64())})
    _fake_connectorx(monkeypatch, big)
    refusing = ConnectorXSource("SELECT id FROM t", "mysql://u@h/db")
    with pytest.raises(BackendError, match="unsigned='decimal'"):
        refusing.read()
    exact = ConnectorXSource("SELECT id FROM t", "mysql://u@h/db", unsigned="decimal")
    assert pa.Table.from_batches(exact.read()).column("id").to_pylist() == [
        decimal.Decimal(2**64 - 1)
    ]
    _fake_connectorx(monkeypatch, big.slice(0, 0))  # the zero-row probe
    assert exact.schema().field("id").type == pa.decimal128(20, 0)


def test_a_missing_connectorx_names_the_vendor_routes(monkeypatch):
    monkeypatch.setitem(sys.modules, "connectorx", None)
    source = ConnectorXSource("SELECT 1", "mysql://u@h/db")
    with pytest.raises(ImportError, match="For MySQL / MariaDB: pip install connectorx"):
        source.read()


# --- PostgreSQL NUMERIC NaN / Infinity, driver handles ----------------------------------


def test_numeric_nan_is_refused_with_a_cast_hint(monkeypatch):
    _fake_driver(
        monkeypatch, "fakepg", [(decimal.Decimal("1.5"),), (decimal.Decimal("NaN"),)], ["price"]
    )
    with pytest.raises(BackendError, match=r"column 'price'.*NaN.*double precision"):
        _read(DBAPISource(module="fakepg", query="SELECT price FROM t"))


def test_postgres_arrays_and_decimals_convert(monkeypatch):
    rows = [([1, 2], decimal.Decimal("1.25")), ([3], decimal.Decimal("2.50"))]
    _fake_driver(monkeypatch, "fakepg", rows, ["tags", "price"])
    table = _read(DBAPISource(module="fakepg", query="SELECT * FROM t"))
    assert table.schema.field("tags").type == pa.list_(pa.int64())
    assert pa.types.is_decimal(table.schema.field("price").type)
    assert table.column("price").to_pylist() == [decimal.Decimal("1.25"), decimal.Decimal("2.50")]


def test_a_driver_handle_is_refused_naming_its_type():
    class LOB:
        def read(self):
            return "x"

    err = explain_failure("doc", [LOB()], pa.ArrowInvalid("cannot convert"))
    assert "column 'doc'" in str(err)
    assert "LOB" in str(err)


# --- Oracle NUMBER as exact Decimal -----------------------------------------------------


def test_oracle_numbers_decimal_installs_the_documented_handler(monkeypatch):
    oracledb = types.ModuleType("oracledb")
    oracledb.DB_TYPE_NUMBER = object()
    monkeypatch.setitem(sys.modules, "oracledb", oracledb)

    class _Conn:
        outputtypehandler = None

    conn = _Conn()
    prepare_connection(conn, "oracledb", oracle_numbers="decimal")
    handler = conn.outputtypehandler
    made = []

    class _Cursor:
        arraysize = 100

        def var(self, kind, arraysize):
            made.append((kind, arraysize))
            return "var"

    number = types.SimpleNamespace(type_code=oracledb.DB_TYPE_NUMBER)
    other = types.SimpleNamespace(type_code=object())
    assert handler(_Cursor(), number) == "var"
    assert handler(_Cursor(), other) is None
    assert made == [(decimal.Decimal, 100)]


def test_oracle_numbers_reaches_the_connection_a_read_opens(monkeypatch):
    oracledb_conns = _fake_driver(monkeypatch, "oracledb", [(1,)], ["n"])
    sys.modules["oracledb"].DB_TYPE_NUMBER = object()
    _read(DBAPISource(module="oracledb", query="SELECT n FROM t", oracle_numbers="decimal"))
    assert oracledb_conns and oracledb_conns[0].outputtypehandler is not None


def test_oracle_numbers_is_refused_for_another_driver():
    with pytest.raises(BackendError, match="python-oracledb only"):
        prepare_connection(object(), "psycopg", oracle_numbers="decimal")
