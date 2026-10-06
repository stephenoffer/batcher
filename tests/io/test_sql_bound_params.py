"""`bt.read.sql(..., params=...)` binds placeholders on every statement it runs (AP-442).

Covers the DB-API path through two real drivers (sqlite3, duckdb), the ADBC path through
an in-process stand-in for `adbc_driver_manager.dbapi`, the ``%`` escaping a
``pyformat`` driver needs once parameters are bound, the split identity, and the
ConnectorX refusal.
"""

from __future__ import annotations

import sqlite3
import sys
import types

import pytest
from tests._harness import assert_same

import batcher as bt
from batcher._internal.errors import BackendError
from batcher.io.formats.sql.dbapi.source import DBAPISource

pytestmark = pytest.mark.integration

_ROWS = [(1, "a%"), (2, "b"), (3, None), (3, "c%x"), (5, "a")]


@pytest.fixture
def sqlite_db(tmp_path):
    path = tmp_path / "t.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (k INTEGER, s TEXT)")
    con.executemany("INSERT INTO t VALUES (?, ?)", _ROWS)
    con.commit()
    con.close()
    return str(path)


@pytest.fixture
def duck_db(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    path = str(tmp_path / "t.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE t (k BIGINT, s VARCHAR)")
    con.executemany("INSERT INTO t VALUES (?, ?)", _ROWS)
    con.close()
    return path


def test_sqlite_uri_binds_positional_params(sqlite_db):
    ds = bt.read.sql("SELECT k, s FROM t WHERE k >= ?", uri=f"sqlite:///{sqlite_db}", params=[3])
    assert sorted(ds.to_pydict()["k"]) == [3, 3, 5]


def test_sqlite_named_params_and_a_pushed_filter(sqlite_db):
    con = sqlite3.connect(sqlite_db)
    ds = bt.read.sql("SELECT k, s FROM t WHERE k > :lo", connection=con, params={"lo": 1})
    got = ds.filter(bt.col("s").str.starts_with("c")).to_pydict()
    assert got == {"k": [3], "s": ["c%x"]}
    con.execute("SELECT 1")  # the borrowed connection is still usable


def test_duckdb_dbapi_matches_duckdb_on_the_same_bound_query(duck_db):
    duckdb = pytest.importorskip("duckdb")
    query = "SELECT k, s FROM t WHERE k BETWEEN ? AND ?"
    oracle = duckdb.connect(duck_db, read_only=True)
    expected = oracle.sql(
        "SELECT * FROM (SELECT k, s FROM t WHERE k BETWEEN 2 AND 5) WHERE s IS NOT NULL"
    )
    con = duckdb.connect(duck_db, read_only=True)
    got = bt.read.sql(query, connection=con, params=[2, 5]).filter(bt.col("s").is_not_null())
    assert_same(got.to_arrow(), expected)


def test_partitioned_read_binds_params_on_every_partition(sqlite_db):
    ds = bt.read.sql(
        "SELECT k, s FROM t WHERE k <> ?",
        uri=f"sqlite:///{sqlite_db}",
        params=[2],
        partition_on="k",
        lower_bound=1,
        upper_bound=5,
        num_partitions=3,
    )
    assert sorted(ds.to_pydict()["k"]) == [1, 3, 3, 5]


def test_an_empty_bound_result_keeps_its_columns(sqlite_db):
    ds = bt.read.sql("SELECT k FROM t WHERE k > ?", uri=f"sqlite:///{sqlite_db}", params=[99])
    assert ds.to_pydict() == {"k": []}


def test_params_are_part_of_the_identity(sqlite_db):
    uri = f"sqlite:///{sqlite_db}"
    q = "SELECT k FROM t WHERE k > ?"
    one, two = DBAPISource(query=q, uri=uri, params=[1]), DBAPISource(query=q, uri=uri, params=[2])
    assert one.identity() != two.identity()
    assert one.splits()[0].identity() != two.splits()[0].identity()
    assert "params=" not in DBAPISource(query=q, uri=uri).identity()


def test_params_are_refused_on_a_table_read(sqlite_db):
    with pytest.raises(BackendError, match="table= read has none"):
        bt.read.sql(table="t", uri=f"sqlite:///{sqlite_db}", params=[1])


def test_a_bare_string_is_refused(sqlite_db):
    with pytest.raises(BackendError, match="one placeholder per character"):
        bt.read.sql("SELECT k FROM t WHERE s = ?", uri=f"sqlite:///{sqlite_db}", params="a")


def _pyformat_driver(path: str) -> types.ModuleType:
    """A DB-API module with ``format`` paramstyle, over sqlite3.

    It does what psycopg and pymysql do with a statement executed *with* parameters:
    ``%s`` is a placeholder and ``%%`` is a literal ``%``. A lone ``%`` raises, which is
    exactly the failure an unescaped pushed ``LIKE`` literal would hit.
    """

    def translate(sql: str) -> str:
        out, i = [], 0
        while i < len(sql):
            if sql[i] == "%":
                nxt = sql[i + 1 : i + 2]
                if nxt == "s":
                    out.append("?")
                elif nxt == "%":
                    out.append("%")
                else:
                    raise ValueError(f"unsupported format character {nxt!r}")
                i += 2
            else:
                out.append(sql[i])
                i += 1
        return "".join(out)

    class Cursor:
        def __init__(self, cur):
            self._cur = cur

        def execute(self, sql, params=None):
            if params is None:
                return self._cur.execute(sql)
            return self._cur.execute(translate(sql), params)

        def __getattr__(self, name):
            return getattr(self._cur, name)

    class Connection:
        def __init__(self):
            self._con = sqlite3.connect(path)

        def cursor(self):
            return Cursor(self._con.cursor())

        def close(self):
            self._con.close()

    mod = types.ModuleType("fake_pyformat_driver")
    mod.paramstyle = "format"
    mod.connect = lambda **_: Connection()
    for name in ("STRING", "NUMBER", "DATETIME", "BINARY", "ROWID"):
        setattr(mod, name, getattr(sqlite3, name, object()))
    return mod


def test_a_pushed_percent_literal_is_escaped_under_a_pyformat_driver(sqlite_db, monkeypatch):
    monkeypatch.setitem(sys.modules, "fake_pyformat_driver", _pyformat_driver(sqlite_db))
    ds = bt.read.table(
        "dbapi", module="fake_pyformat_driver", query="SELECT k, s FROM t WHERE k > %s", params=[0]
    )
    # `ends_with("x")` is pushed as `LIKE '%x'`: that '%' must reach the driver doubled,
    # while the query's own `%s` placeholder reaches it untouched. Unescaped, the driver
    # reads `%x` as a format character and raises.
    filtered = ds.filter(bt.col("s").str.ends_with("x"))
    assert filtered.to_pydict() == {"k": [3], "s": ["c%x"]}


def test_the_adbc_path_binds_params(duck_db, monkeypatch):
    duckdb = pytest.importorskip("duckdb")
    fake = types.ModuleType("adbc_driver_manager.dbapi")
    fake.connect = lambda driver, db_kwargs, **_: duckdb.connect(db_kwargs["path"], read_only=True)
    pkg = types.ModuleType("adbc_driver_manager")
    pkg.dbapi = fake
    monkeypatch.setitem(sys.modules, "adbc_driver_manager", pkg)
    monkeypatch.setitem(sys.modules, "adbc_driver_manager.dbapi", fake)
    ds = bt.read.table(
        "adbc",
        driver="duckdb",
        db_kwargs={"path": duck_db},
        query="SELECT k FROM t WHERE k > ?",
        params=[2],
    )
    assert sorted(ds.to_pydict()["k"]) == [3, 3, 5]
    assert "adbc(duckdb)" in ds.explain()


def test_connectorx_refuses_params_and_names_the_dbapi_route(monkeypatch):
    monkeypatch.setattr("batcher.api.io_namespace.reader.read_backend", lambda *_: "connectorx")
    with pytest.raises(BackendError, match=r"no parameter binding.*module="):
        bt.read.sql("SELECT 1 WHERE 1 = ?", uri="mysql://h/db", params=[1])


def test_explain_names_the_dbapi_backend(sqlite_db):
    ds = bt.read.sql("SELECT k FROM t", uri=f"sqlite:///{sqlite_db}")
    assert "dbapi(sqlite3)" in ds.explain()
    # The control: a source with no backend to name renders as before.
    assert "dbapi" not in bt.from_pydict({"k": [1]}).explain()


def test_dbapi_fallback_is_logged(sqlite_db, caplog, monkeypatch):
    from batcher.io.formats.sql import routing

    monkeypatch.setattr(routing, "_installed", lambda module: False)
    with caplog.at_level("INFO", logger="batcher.io.sql"):
        assert routing.read_backend(f"sqlite:///{sqlite_db}", {}) == "dbapi"
    assert any("falls back to DB-API" in r.getMessage() for r in caplog.records)
    assert any("adbc-driver-sqlite" in r.getMessage() for r in caplog.records)
