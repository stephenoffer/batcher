"""The SQL that pushdown generates is executed by real databases and returns the right rows.

The other pushdown tests assert on the *text* Batcher generates. Text can be wrong in two
ways text-comparison cannot see: a server may reject it, or accept it and return different
rows. Both are what pushdown is for, and neither shows up until a database is involved.

DuckDB and SQLite stand in for the ones that are not installable here. They are genuinely
independent implementations, and between them they cover the two properties every SQL
backend's pushdown depends on: that a delimited identifier means a *column* (so a column
named `order` or `my col` is readable at all) and that `LIMIT n` is accepted where Batcher
claims the dialect takes it. Both are in Batcher's own dialect tables, so this exercises
the same code path a Postgres or ClickHouse read would.

The property asserted is not "the SQL parses" but "the pushed query returns exactly the
rows the unpushed one would have, after the same filter" — which is the only thing that
makes pushing a predicate to a server safe.
"""

from __future__ import annotations

import sqlite3

import pytest

import batcher as bt
from batcher.io.formats.sql._common import count_query, push_down
from batcher.io.formats.sql.uri import quote_identifier

pytestmark = pytest.mark.unit

# Column names chosen to break an unquoted emitter: two reserved words and one with a
# space, which unquoted parses as a column plus an alias rather than as one column.
_DDL = 'CREATE TABLE "t" ("order" BIGINT, "user" VARCHAR, "my col" BIGINT)'
_ROWS = [(1, "US", 10), (2, "CA", 20), (3, "MX", 30), (None, "US", 40), (2, None, 50)]

# (name, batcher predicate, equivalent SQL predicate written by hand)
_CASES = [
    ("eq", bt.col("order") == 2, '"order" = 2'),
    ("in_list", bt.col("order").is_in([1, 3]), '"order" IN (1, 3)'),
    ("not_in", ~bt.col("order").is_in([1, 3]), 'NOT ("order" IN (1, 3))'),
    ("not_eq", ~(bt.col("order") == 2), 'NOT ("order" = 2)'),
    ("is_null", bt.col("order").is_null(), '"order" IS NULL'),
    ("is_not_null", bt.col("user").is_not_null(), '"user" IS NOT NULL'),
    ("starts_with", bt.col("user").str.starts_with("U"), "\"user\" LIKE 'U%'"),
    ("contains", bt.col("user").str.contains("S"), "\"user\" LIKE '%S%'"),
    (
        "and",
        bt.col("order").is_in([1, 2]) & (bt.col("my col") > 10),
        '"order" IN (1,2) AND "my col" > 10',
    ),
    (
        "or",
        (bt.col("order") == 1) | bt.col("user").str.starts_with("M"),
        '"order" = 1 OR "user" LIKE \'M%\'',
    ),
    ("gt_spaced_col", bt.col("my col") > 25, '"my col" > 25'),
]


def _duckdb():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute(_DDL)
    con.executemany('INSERT INTO "t" VALUES (?, ?, ?)', _ROWS)
    return con, lambda sql: con.execute(sql).fetchall()


def _sqlite():
    con = sqlite3.connect(":memory:")
    con.execute(_DDL.replace("BIGINT", "INTEGER").replace("VARCHAR", "TEXT"))
    con.executemany('INSERT INTO "t" VALUES (?, ?, ?)', _ROWS)
    return con, lambda sql: con.execute(sql).fetchall()


@pytest.fixture(params=["duckdb", "sqlite"])
def engine(request):
    """A live SQL engine plus the scheme naming its dialect."""
    con, run = _duckdb() if request.param == "duckdb" else _sqlite()
    yield request.param, run
    con.close()


@pytest.mark.parametrize(("name", "predicate", "reference"), _CASES, ids=[c[0] for c in _CASES])
def test_a_pushed_predicate_selects_the_rows_the_server_would(engine, name, predicate, reference):
    scheme, run = engine
    quote = lambda column: quote_identifier(column, scheme)  # noqa: E731
    pushed = push_down(None, predicate.to_ir(), None, table=quote("t"), quote=quote)
    expected = run(f'SELECT * FROM "t" WHERE {reference}')
    assert sorted(run(pushed), key=repr) == sorted(expected, key=repr)


def test_a_pushed_projection_reads_only_those_columns_including_awkward_names(engine):
    scheme, run = engine
    quote = lambda column: quote_identifier(column, scheme)  # noqa: E731
    pushed = push_down(None, None, ["order", "my col"], table=quote("t"), quote=quote)
    got = run(pushed)
    assert len(got[0]) == 2, "the projection did not reach the server"
    assert sorted(got, key=repr) == sorted(run('SELECT "order", "my col" FROM "t"'), key=repr)


@pytest.mark.parametrize("limit", [1, 3, 99])
def test_a_pushed_limit_caps_what_the_server_returns(engine, limit):
    scheme, run = engine
    quote = lambda column: quote_identifier(column, scheme)  # noqa: E731
    pushed = push_down(None, None, None, table=quote("t"), limit=limit, quote=quote)
    assert len(run(pushed)) == min(limit, len(_ROWS))


def test_a_limit_applies_after_the_filter_not_before_it(engine):
    # The cap counts the rows the read *returns*, so it must sit above the predicate. Below
    # it, `LIMIT 2` would take two rows and then filter them, yielding fewer than it should.
    scheme, run = engine
    quote = lambda column: quote_identifier(column, scheme)  # noqa: E731
    predicate = (bt.col("my col") > 20).to_ir()
    pushed = push_down(None, predicate, None, table=quote("t"), limit=2, quote=quote)
    assert len(run(pushed)) == 2


def test_the_count_query_returns_the_row_count(engine):
    scheme, run = engine
    quote = lambda column: quote_identifier(column, scheme)  # noqa: E731
    assert run(count_query(None, table=quote("t")))[0][0] == len(_ROWS)


def test_an_unquoted_reserved_word_would_not_have_worked(engine):
    """The control: the same read without delimiters is rejected or wrong.

    Without this, the quoting tests pass on a table whose columns never needed quoting
    and prove nothing.
    """
    _scheme, run = engine
    with pytest.raises(Exception):  # noqa: B017 - each driver raises its own type
        run('SELECT order, user FROM "t"')


# --- top-N: the pushed ORDER BY must select the rows the engine's own sort would --------
#
# This is the case a text assertion cannot check. Servers disagree about where a null
# sorts by default: on sqlite 3.52, `ORDER BY k LIMIT 2` over [3, NULL, 1, NULL, 2]
# returns [NULL, NULL], while DuckDB and Batcher return [1, 2]. Pushed without an explicit
# NULLS clause, a top-N therefore asks two servers for two different answers and neither
# is guaranteed to be the engine's. These run the pushed SQL and compare it against
# Batcher computing the same top-N itself.

_TOPN_ROWS = [(3, "c"), (None, "n1"), (1, "a"), (None, "n2"), (2, "b")]
_TOPN_DDL = 'CREATE TABLE "t" ("order" BIGINT, "v" VARCHAR)'


@pytest.fixture(params=["duckdb", "sqlite"])
def topn_engine(request):
    if request.param == "duckdb":
        duckdb = pytest.importorskip("duckdb")
        con = duckdb.connect()
        con.execute(_TOPN_DDL)
        run = lambda sql: con.execute(sql).fetchall()  # noqa: E731
    else:
        con = sqlite3.connect(":memory:")
        con.execute(_TOPN_DDL.replace("BIGINT", "INTEGER").replace("VARCHAR", "TEXT"))
        run = lambda sql: con.execute(sql).fetchall()  # noqa: E731
    con.executemany('INSERT INTO "t" VALUES (?, ?)', _TOPN_ROWS)
    yield request.param, run
    con.close()


@pytest.mark.parametrize(
    ("descending", "nulls_first"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_a_pushed_top_n_returns_what_batcher_would_have_sorted(
    topn_engine, descending, nulls_first
):
    scheme, run = topn_engine
    quote = lambda column: quote_identifier(column, scheme)  # noqa: E731
    pushed = push_down(
        None,
        None,
        ["v"],
        table=quote("t"),
        limit=2,
        order_by=(("order", descending, nulls_first),),
        quote=quote,
    )
    engine_answer = (
        bt.from_pydict({"order": [r[0] for r in _TOPN_ROWS], "v": [r[1] for r in _TOPN_ROWS]})
        .sort("order", descending=descending, nulls_first=nulls_first)
        .limit(2)
        .to_pydict()["v"]
    )
    assert [row[0] for row in run(pushed)] == engine_answer


def test_the_server_default_would_not_have_agreed_on_sqlite():
    """The control: without the explicit clause these engines genuinely differ.

    If both servers happened to agree by default, the test above would pass on a
    generated clause that did nothing, and prove nothing.
    """
    duckdb = pytest.importorskip("duckdb")
    rows = [(3,), (None,), (1,), (None,), (2,)]

    duck = duckdb.connect()
    duck.execute('CREATE TABLE "t" ("k" BIGINT)')
    duck.executemany('INSERT INTO "t" VALUES (?)', rows)
    lite = sqlite3.connect(":memory:")
    lite.execute('CREATE TABLE "t" ("k" INTEGER)')
    lite.executemany('INSERT INTO "t" VALUES (?)', rows)

    unqualified = 'SELECT "k" FROM "t" ORDER BY "k" LIMIT 2'
    assert duck.execute(unqualified).fetchall() != lite.execute(unqualified).fetchall()

    qualified = 'SELECT "k" FROM "t" ORDER BY "k" ASC NULLS LAST LIMIT 2'
    assert duck.execute(qualified).fetchall() == lite.execute(qualified).fetchall()
    duck.close()
    lite.close()
