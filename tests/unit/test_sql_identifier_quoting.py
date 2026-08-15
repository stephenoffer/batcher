"""Column names reach the server delimited, in the delimiter that server understands.

Pushed projections and predicates emitted column names verbatim, and three ordinary names
break that way:

- a reserved word (``order``, ``user``, ``key``, ``value``, ``date``) is a syntax error;
- a name holding a space is *worse than* an error — ``SELECT my col`` parses as the column
  ``my`` aliased to ``col``, so the query succeeds and returns the wrong column under the
  right name;
- an unaliased aggregate in a user's own query yields a result column literally named
  ``count(*)``, which unquoted is re-parsed as a function call.

The delimiter cannot be one constant, and MySQL is why. ANSI double quotes only delimit an
identifier there under ``ANSI_QUOTES``, which is off by default, so ``"user"`` is a
*string literal* — quoting that way would silently select the constant ``'user'`` for
every row. Backticks on the MySQL family, brackets on SQL Server, double quotes elsewhere,
and nothing at all for a dialect Batcher cannot name.

Pure control-plane string generation; no live database is required.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.io.formats.sql._common import apply_projection, push_down
from batcher.io.formats.sql.uri import quote_identifier

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("scheme", "expected"),
    [
        ("postgresql", '"order"'),
        ("postgresql+psycopg2", '"order"'),
        ("snowflake", '"order"'),
        ("oracle", '"order"'),
        ("duckdb", '"order"'),
        ("mysql", "`order`"),
        ("mariadb", "`order`"),
        ("tidb", "`order`"),
        ("bigquery", "`order`"),
        ("mssql", "[order]"),
        ("sqlserver", "[order]"),
    ],
)
def test_each_dialect_gets_its_own_delimiter(scheme, expected):
    assert quote_identifier("order", scheme) == expected


def test_an_unknown_dialect_is_left_verbatim():
    # Quoting with the wrong delimiter is a new failure; not quoting is what every read
    # already did.
    assert quote_identifier("order", "some-unknown-database") == "order"
    assert quote_identifier("order", "") == "order"


@pytest.mark.parametrize(
    ("scheme", "name", "expected"),
    [
        ("postgresql", 'we"ird', '"we""ird"'),
        ("mysql", "we`ird", "`we``ird`"),
        ("mssql", "we]ird", "[we]]ird]"),
    ],
)
def test_a_delimiter_inside_the_name_is_doubled(scheme, name, expected):
    assert quote_identifier(name, scheme) == expected


def test_a_projection_is_delimited():
    quote = lambda name: quote_identifier(name, "postgresql")  # noqa: E731
    sql = apply_projection("SELECT * FROM t", ["order", "my col"], quote=quote)
    assert '"order", "my col"' in sql


def test_a_name_with_a_space_is_one_column_not_a_column_and_an_alias():
    # Unquoted, `SELECT my col` is `my AS col` — a successful query returning the wrong
    # column. This is the case that fails silently, so it is pinned on its own.
    plain = apply_projection("SELECT * FROM t", ["my col"])
    assert "my col" in plain and '"my col"' not in plain  # the defect, still the default
    quote = lambda name: quote_identifier(name, "postgresql")  # noqa: E731
    assert '"my col"' in apply_projection("SELECT * FROM t", ["my col"], quote=quote)


def test_a_pushed_predicate_is_delimited_too():
    quote = lambda name: quote_identifier(name, "postgresql")  # noqa: E731
    predicate = ((bt.col("order") == 1) & bt.col("user").is_in(["a"])).to_ir()
    sql = push_down("SELECT * FROM t", predicate, ["order"], quote=quote)
    assert '"order" = 1' in sql
    assert '"user" IN' in sql


def test_the_default_stays_verbatim_for_callers_that_cannot_name_their_dialect():
    # Every backend that does not declare a dialect keeps exactly the SQL it emitted
    # before, so this can never regress one of them.
    assert apply_projection("SELECT * FROM t", ["order"]) == (
        "SELECT order FROM (\nSELECT * FROM t\n) AS _bc"
    )


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("postgresql://h/db", '"order"'),
        ("mysql://h/db", "`order`"),
        ("mssql://h/db", "[order]"),
        ("weird://h/db", "order"),
    ],
)
def test_connectorx_delimits_per_connection_because_it_fronts_many_dialects(uri, expected):
    from batcher.io.formats.sql.connectorx import ConnectorXSource

    source = ConnectorXSource(query="SELECT * FROM t", conn_uri=uri)
    assert expected in source._split(None, ["order"]).query


def test_odbc_stays_verbatim_because_a_dsn_names_a_driver_not_a_dialect():
    from batcher.io.formats.sql.odbc import ODBCSource

    assert ODBCSource.sql_dialect == ""
