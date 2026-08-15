"""A `LIMIT` reaches the database, and only where the database can parse it.

``read.postgres(...).limit(100)`` issued an unbounded ``SELECT`` and pulled the whole
table across the network to keep a hundred rows, because the plan's source hand-off had
no row-cap channel at all. `PhysicalPlan.source_limits` is that channel and this is the
half that turns it into SQL.

The dialect gate is the load-bearing part, and the two failure directions are not
symmetric. Not capping a read costs the rows the server would have skipped, which is
exactly what every read did before. Emitting `LIMIT` at a server that spells the same
thing `TOP` or `FETCH FIRST` turns a *working* query into a syntax error. So the gate is
an allow-list, and anything unrecognized reads uncapped.

Pure control-plane string generation; no live database is required.
"""

from __future__ import annotations

import pytest

from batcher.io.formats.sql._common import push_down
from batcher.io.formats.sql.uri import supports_limit_clause

pytestmark = pytest.mark.unit


def test_the_cap_is_appended_outside_the_projection():
    # Outermost, unlike the predicate: it counts the rows the read *returns*, so it has
    # to sit above the filter rather than below it.
    sql = push_down("SELECT * FROM t", None, ["a"], limit=10)
    assert sql.endswith("LIMIT 10")
    assert sql.index("SELECT a") < sql.index("LIMIT 10")


def test_the_cap_sits_above_a_pushed_predicate():
    predicate = {
        "e": "binary",
        "op": "gt",
        "left": {"e": "col", "name": "a"},
        "right": {"e": "lit", "value": {"int": 5}},
    }
    sql = push_down("SELECT * FROM t", predicate, None, limit=3)
    assert sql.index("a > 5") < sql.index("LIMIT 3")


def test_no_cap_leaves_the_sql_exactly_as_it_was():
    assert push_down("SELECT * FROM t", None, ["a"]) == push_down(
        "SELECT * FROM t", None, ["a"], limit=None
    )


@pytest.mark.parametrize(
    "scheme",
    [
        "postgresql",
        "postgresql+psycopg2",
        "postgres",
        "mysql",
        "mariadb",
        "sqlite",
        "duckdb",
        "clickhouse",
        "snowflake",
        "bigquery",
        "trino",
        "redshift",
        "cockroachdb",
        "tidb",
        "singlestore",
    ],
)
def test_dialects_that_take_a_limit_clause(scheme):
    assert supports_limit_clause(scheme) is True


@pytest.mark.parametrize("scheme", ["mssql", "sqlserver", "oracle", "flightsql", "unknown"])
def test_dialects_that_must_not_be_capped(scheme):
    # SQL Server spells it `TOP`/`OFFSET…FETCH` and Oracle `FETCH FIRST`; emitting
    # `LIMIT` at either is a syntax error, not a slow query.
    assert supports_limit_clause(scheme) is False


def test_clickhouse_declares_support_as_a_class_variable_not_a_field():
    # With postponed annotations a `ClassVar` whose name the module never imported is
    # read as an ordinary dataclass field, which would put `supports_limit` in the
    # constructor and give every instance its own copy.
    import dataclasses

    from batcher.io.formats.sql.clickhouse import ClickHouseSource

    assert ClickHouseSource.supports_limit is True
    assert "supports_limit" not in {f.name for f in dataclasses.fields(ClickHouseSource)}


@pytest.mark.parametrize(
    ("uri", "capped"),
    [
        ("postgresql://u:p@h/db", True),
        ("mysql://h/db", True),
        ("mssql://u:p@h/db", False),
        ("oracle://u:p@h/db", False),
    ],
)
def test_connectorx_decides_per_connection_because_it_fronts_many_dialects(uri, capped):
    from batcher.io.formats.sql.connectorx import ConnectorXSource

    source = ConnectorXSource(query="SELECT * FROM t", conn_uri=uri)
    assert source.supports_limit is capped


def test_a_backend_that_declines_the_clause_emits_no_cap():
    from batcher.io.formats.sql.connectorx import ConnectorXSource

    source = ConnectorXSource(query="SELECT * FROM t", conn_uri="mssql://u:p@h/db")
    assert "LIMIT" not in source._split(None, None, 10).query


def test_a_backend_that_accepts_the_clause_emits_the_cap():
    from batcher.io.formats.sql.connectorx import ConnectorXSource

    source = ConnectorXSource(query="SELECT * FROM t", conn_uri="postgresql://u:p@h/db")
    assert source._split(None, None, 10).query.endswith("LIMIT 10")
