"""Snowflake auth strategies, Databricks warehouse sessions and the Athena profile.

Fake client modules stand in for ``snowflake.connector``, ``databricks.sql`` and
``pyathena``; the tests pin what is sent to ``connect()``, that a worker connects with the
same declared strategy the driver validated, and what a failure reports.
"""

from __future__ import annotations

import pickle
import sys
import types
from contextlib import closing
from typing import Any

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import BackendError
from batcher.io.formats.sql._common import connection_fingerprint
from batcher.io.formats.sql.databricks import DatabricksSource
from batcher.io.formats.sql.vendors import athena_connect_kwargs, snowflake_options

pytestmark = pytest.mark.unit


# --- Snowflake ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (
            {"auth": "password", "user": "u", "password": "env:PW"},
            {"user": "u", "password": "env:PW"},
        ),
        (
            {"user": "u", "private_key_file": "/k.p8", "private_key_file_pwd": "env:K"},
            {
                "user": "u",
                "private_key_file": "/k.p8",
                "private_key_file_pwd": "env:K",
                "authenticator": "SNOWFLAKE_JWT",
            },
        ),
        ({"token": "env:T"}, {"token": "env:T", "authenticator": "oauth"}),
        (
            {"auth": "externalbrowser", "user": "u"},
            {"user": "u", "authenticator": "externalbrowser"},
        ),
    ],
)
def test_each_strategy_maps_to_the_connector_keywords(given, expected):
    out = snowflake_options({"account": "acme", "role": "R", "warehouse": "WH", **given})
    assert out["connection_kwargs"] == {
        "account": "acme",
        "role": "R",
        "warehouse": "WH",
        **expected,
    }


def test_options_without_auth_keywords_pass_through_unchanged():
    opts = {"connection_kwargs": {"account": "a"}, "governed_as": "t"}
    assert snowflake_options(dict(opts)) == opts


@pytest.mark.parametrize(
    ("given", "message"),
    [
        ({"auth": "saml"}, "not a Snowflake strategy"),
        ({"auth": "key_pair", "user": "u"}, "needs private_key_file"),
        ({"auth": "oauth", "token": "t", "password": "p"}, "does not use password"),
        ({"user": "u", "password": "p", "connection_kwargs": {"user": "v"}}, "pass it once"),
        ({"auth": "password", "password": "p", "user": "u", "account": None}, "needs account"),
    ],
)
def test_bad_combinations_are_refused_on_the_driver(given, message):
    with pytest.raises(BackendError, match=message):
        snowflake_options({"account": "acme", **given})


def test_the_key_passphrase_is_not_part_of_the_relation_identity():
    base = {"account": "a", "user": "u", "private_key_file": "/k.p8"}
    assert connection_fingerprint({**base, "private_key_file_pwd": "x"}) == connection_fingerprint(
        {**base, "private_key_file_pwd": "y"}
    )


def test_a_read_and_its_worker_connect_with_the_same_strategy(monkeypatch):
    seen: list[dict[str, Any]] = []
    connector = types.ModuleType("snowflake.connector")

    class _Batch:
        def to_arrow(self):
            return pa.table({"x": [1]})

    class _Cursor:
        def execute(self, sql):
            pass

        def get_result_batches(self):
            return [_Batch()]

    class _Conn:
        def cursor(self):
            return _Cursor()

        def close(self):
            pass

    def connect(**kwargs):
        seen.append(kwargs)
        return _Conn()

    connector.connect = connect
    monkeypatch.setitem(sys.modules, "snowflake", types.ModuleType("snowflake"))
    monkeypatch.setitem(sys.modules, "snowflake.connector", connector)
    monkeypatch.setenv("SF_KEY_PWD", "s3cret")
    ds = bt.read.snowflake(
        "SELECT x FROM t",
        account="acme",
        user="etl",
        private_key_file="/keys/etl.p8",
        private_key_file_pwd="env:SF_KEY_PWD",
        warehouse="WH",
    )
    source = ds._sources[0]
    # What a worker receives is the pickled source: the reference, not the secret.
    shipped = pickle.loads(pickle.dumps(source))
    assert shipped.connection_kwargs["private_key_file_pwd"] == "env:SF_KEY_PWD"
    assert "s3cret" not in repr(source)
    shipped.splits()
    assert seen[-1]["authenticator"] == "SNOWFLAKE_JWT"
    assert seen[-1]["private_key_file_pwd"] == "s3cret"  # resolved where it connects


# --- Databricks warehouse ----------------------------------------------------------------


class _FakeWarehouse:
    def __init__(self, fail: bool = False, rows: int = 3) -> None:
        self.connects: list[dict[str, Any]] = []
        self.cancelled = 0
        recorder = self

        class _Cursor:
            query_id = None

            def execute(self, sql):
                self.query_id = "01ef-query"
                if fail:
                    raise RuntimeError("TABLE_OR_VIEW_NOT_FOUND")

            def fetchmany_arrow(self, n):
                if not hasattr(self, "_left"):
                    self._left = rows
                if self._left <= 0:
                    return None
                self._left -= 1
                return pa.table({"x": [self._left]})

            def fetchall_arrow(self):
                return pa.table({"x": list(range(rows))})

            def cancel(self):
                recorder.cancelled += 1

        class _Conn:
            def cursor(self):
                return _Cursor()

            def close(self):
                pass

        def connect(**kwargs):
            recorder.connects.append(kwargs)
            return _Conn()

        self.module = types.ModuleType("databricks.sql")
        self.module.connect = connect


def _warehouse(**extra: Any) -> DatabricksSource:
    return DatabricksSource(
        query="SELECT x FROM orders",
        server_hostname="adb.example",
        http_path="/sql/1.0/warehouses/w",
        access_token="tok",
        **extra,
    )


def test_session_options_reach_connect(monkeypatch):
    fake = _FakeWarehouse()
    monkeypatch.setitem(sys.modules, "databricks.sql", fake.module)
    source = _warehouse(
        catalog="main",
        db_schema="sales",
        session_configuration={"ansi_mode": "true"},
        statement_timeout_s=60,
    )
    source.read()
    connect = fake.connects[-1]
    assert connect["catalog"] == "main"
    assert connect["schema"] == "sales"
    assert connect["session_configuration"] == {"ansi_mode": "true", "STATEMENT_TIMEOUT": "60"}
    # The split a worker rebuilds from carries the same session.
    (split,) = source.splits()
    assert pickle.loads(pickle.dumps(split)).connect_options == {
        "catalog": "main",
        "schema": "sales",
        "session_configuration": {"ansi_mode": "true", "STATEMENT_TIMEOUT": "60"},
    }


def test_an_abandoned_read_cancels_the_statement(monkeypatch):
    fake = _FakeWarehouse(rows=5)
    monkeypatch.setitem(sys.modules, "databricks.sql", fake.module)
    with closing(_warehouse().iter_batches()) as batches:
        next(batches)
    assert fake.cancelled == 1


def test_a_complete_read_does_not_cancel(monkeypatch):
    fake = _FakeWarehouse(rows=2)
    monkeypatch.setitem(sys.modules, "databricks.sql", fake.module)
    assert sum(b.num_rows for b in _warehouse().iter_batches()) == 2
    assert fake.cancelled == 0


def test_a_failure_names_the_warehouse_query_id(monkeypatch):
    fake = _FakeWarehouse(fail=True)
    monkeypatch.setitem(sys.modules, "databricks.sql", fake.module)
    with pytest.raises(BackendError, match=r"query id 01ef-query.*TABLE_OR_VIEW_NOT_FOUND"):
        _warehouse().read()


def test_session_defaults_separate_relations_but_absent_ones_keep_the_old_key():
    assert _warehouse().identity() != _warehouse(db_schema="sales").identity()
    assert _warehouse().identity() == _warehouse(statement_timeout_s=5).identity()
    fingerprint = connection_fingerprint(
        {"server_hostname": "adb.example", "http_path": "/sql/1.0/warehouses/w"}
    )
    assert _warehouse().identity() == f"databricks-wh:{fingerprint}:SELECT x FROM orders"


def test_read_databricks_maps_schema_and_takes_a_query(monkeypatch):
    monkeypatch.setitem(sys.modules, "databricks.sql", _FakeWarehouse().module)
    ds = bt.read.databricks(
        query="SELECT 1",
        server_hostname="h",
        http_path="p",
        access_token="t",
        schema="sales",
    )
    assert ds._sources[0].db_schema == "sales"


# --- Athena ------------------------------------------------------------------------------


def test_athena_profile_spells_pyathena_keywords():
    assert athena_connect_kwargs(
        region="eu-west-1",
        output_location="s3://bucket/results/",
        database="web",
        catalog="lake",
        profile_name="analyst",
    ) == {
        "region_name": "eu-west-1",
        "s3_staging_dir": "s3://bucket/results/",
        "schema_name": "web",
        "catalog_name": "lake",
        "profile_name": "analyst",
    }


def test_athena_needs_somewhere_to_write_results():
    with pytest.raises(BackendError, match="output_location"):
        athena_connect_kwargs(region="us-east-1")
    with pytest.raises(BackendError, match="s3://"):
        athena_connect_kwargs(region="us-east-1", output_location="/tmp/out")


def test_read_athena_runs_through_the_dbapi_source(monkeypatch):
    seen: list[dict[str, Any]] = []
    executed: list[str] = []

    class _Cursor:
        description = None

        def execute(self, sql, params=None):
            executed.append(sql)
            self.description = [("n", None)]
            self._rows = [(1,), (2,)]

        def fetchmany(self, n):
            out, self._rows = self._rows[:n], self._rows[n:]
            return out

        def close(self):
            pass

    class _Conn:
        def cursor(self):
            return _Cursor()

        def close(self):
            pass

    def connect(**kwargs):
        seen.append(kwargs)
        return _Conn()

    pyathena = types.ModuleType("pyathena")
    pyathena.connect = connect
    pyathena.paramstyle = "pyformat"
    monkeypatch.setitem(sys.modules, "pyathena", pyathena)
    ds = bt.read.athena("SELECT n FROM events", region="us-east-1", workgroup="analytics")
    assert ds.to_pydict() == {"n": [1, 2]}
    assert seen[-1] == {"region_name": "us-east-1", "work_group": "analytics"}
    assert any("SELECT n FROM events" in sql for sql in executed)


def test_read_athena_names_the_extra_when_pyathena_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyathena", None)
    with pytest.raises(ImportError, match=r"batcher-engine\[athena\]"):
        bt.read.athena("SELECT 1", region="us-east-1", workgroup="w")
