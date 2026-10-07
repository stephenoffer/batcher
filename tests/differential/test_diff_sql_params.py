"""SQL parameter binding (``params=``) vs DuckDB's own parameter binding.

Every case runs the same query text with the same values through
``duckdb.sql(query, params=...)`` and ``bt.sql(query, params=...)``. DuckDB binds a value as
a typed parameter, never as text, so it is the right oracle for the property that matters
most here: a string holding a quote, a comment marker or a whole statement is only a string.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pyarrow as pa
import pytest
from tests._harness import assert_same, assert_same_ordered

import batcher as bt

pytestmark = pytest.mark.differential

_ROWS = {
    "id": [1, 2, 3, 4, 5, 6],
    "name": ["ann", "o'brien", "x'; DROP TABLE t; --", None, "ann", ""],
    "amount": [Decimal("1.50"), Decimal("2.25"), None, Decimal("-3.00"), Decimal("1.50"), None],
    "day": [dt.date(2024, 1, d) for d in (1, 2, 3, 4, 5, 6)],
    "score": [0.5, 1.5, 3.5, None, 2.5, -0.0],
}


@pytest.fixture
def t(duck):
    table = pa.table(
        _ROWS,
        schema=pa.schema(
            [
                ("id", pa.int64()),
                ("name", pa.string()),
                ("amount", pa.decimal128(5, 2)),
                ("day", pa.date32()),
                ("score", pa.float64()),
            ]
        ),
    )
    duck.register("t", table)
    return table


def _both(duck, t, query, params):
    return bt.sql(query, t=t, params=params).collect(), duck.sql(query, params=params)


@pytest.mark.parametrize(
    ("query", "params"),
    [
        ("SELECT id FROM t WHERE name = ?", ["o'brien"]),
        ("SELECT id FROM t WHERE name = ?", ["x'; DROP TABLE t; --"]),
        ("SELECT id FROM t WHERE name = ?", [""]),
        ("SELECT id FROM t WHERE name = ?", ["nobody"]),
        ("SELECT id FROM t WHERE name = ?", [None]),
        ("SELECT id FROM t WHERE id IN (?, ?, ?)", [1, 3, 99]),
        ("SELECT id FROM t WHERE id BETWEEN ? AND ?", [2, 4]),
        ("SELECT id FROM t WHERE amount = ?", [Decimal("1.50")]),
        ("SELECT id FROM t WHERE amount > ?", [Decimal("-3")]),
        ("SELECT id FROM t WHERE day >= ?", [dt.date(2024, 1, 4)]),
        ("SELECT id FROM t WHERE score > ?", [1.0]),
        ("SELECT id FROM t WHERE (id > ?) = ?", [3, True]),
        ("SELECT id, name FROM t WHERE id > $1 AND name = $2", [0, "ann"]),
        ("SELECT id FROM t WHERE id = $2 OR id = $1", [1, 2]),
        ("SELECT id FROM t WHERE id > $1 AND id < $1 + 3", [2]),
        ("SELECT name, count(*) AS n FROM t WHERE id > ? GROUP BY name", [1]),
    ],
)
def test_positional_and_numbered_parameters(duck, t, query, params):
    assert_same(*_both(duck, t, query, params))


@pytest.mark.parametrize(
    ("query", "params"),
    [
        ("SELECT id FROM t WHERE name = $who", {"who": "o'brien"}),
        ("SELECT id FROM t WHERE id >= $lo AND id <= $hi", {"lo": 2, "hi": 5}),
        ("SELECT id FROM t WHERE id = $k OR id = $k + 1", {"k": 3}),
        ("SELECT id FROM t WHERE name = $who", {"who": None}),
    ],
)
def test_named_parameters(duck, t, query, params):
    assert_same(*_both(duck, t, query, params))


@pytest.mark.parametrize(
    ("query", "params"),
    [
        ("SELECT id FROM t ORDER BY id LIMIT ?", [2]),
        ("SELECT id FROM t ORDER BY id LIMIT ? OFFSET ?", [2, 3]),
        ("SELECT id FROM t ORDER BY id DESC LIMIT ?", [0]),
    ],
)
def test_limit_takes_a_parameter(duck, t, query, params):
    assert_same_ordered(*_both(duck, t, query, params))


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        0,
        -7,
        2**40,
        2.5,
        -0.0,
        "it's",
        "",
        Decimal("12.345"),
        Decimal("-0.05"),
        Decimal("100"),
        dt.date(2024, 2, 29),
        dt.datetime(2024, 1, 2, 3, 4, 5, 123456),
        dt.time(3, 4, 5),
        b"abc",
    ],
)
def test_a_value_binds_to_the_type_duckdb_gives_it(duck, value):
    """A select of one bound value matches DuckDB's value; Batcher widens narrow types."""
    query = "SELECT ? AS v"
    assert_same(bt.sql(query, params=[value]).collect(), duck.sql(query, params=[value]))


@pytest.mark.parametrize(
    ("value", "arrow_type"),
    [
        (Decimal("12.345"), pa.decimal128(5, 3)),
        (Decimal("-0.05"), pa.decimal128(2, 2)),
        (dt.date(2024, 2, 29), pa.date32()),
        (dt.datetime(2024, 1, 2, 3, 4, 5), pa.timestamp("us")),
        (2.5, pa.float64()),
        (b"abc", pa.binary()),
    ],
)
def test_a_value_keeps_its_type(value, arrow_type):
    """A float is a DOUBLE, not the DECIMAL its digits spell; a Decimal keeps p and s."""
    got = bt.sql("SELECT ? AS v", params=[value]).collect()
    assert got.schema.field("v").type == arrow_type


def test_non_finite_floats_bind(duck):
    values = [float("nan"), float("inf"), float("-inf")]
    got = bt.sql("SELECT ? AS a, ? AS b, ? AS c", params=values).to_pydict()
    assert got["a"][0] != got["a"][0]  # NaN
    assert (got["b"], got["c"]) == ([float("inf")], [float("-inf")])


def test_an_aware_datetime_is_the_same_instant(duck):
    """A zone-aware datetime is a TIMESTAMPTZ; Batcher reads it as that instant in UTC."""
    value = dt.datetime(2024, 1, 2, 3, 4, 5, tzinfo=dt.timezone(dt.timedelta(hours=5)))
    got = bt.sql("SELECT ? AS v", params=[value]).to_pydict()["v"][0]
    assert got == dt.datetime(2024, 1, 1, 22, 4, 5)
    expected = duck.sql("SELECT ? AS v", params=[value]).fetchone()[0]
    assert expected.astimezone(dt.UTC).replace(tzinfo=None) == got


def test_the_same_query_rebinds_new_values(t):
    """The prepared-statement cache keys on the values, typed: 1 and True do not collide."""
    query = "SELECT count(*) AS n FROM t WHERE id > ?"
    assert bt.sql(query, t=t, params=[4]).to_pydict() == {"n": [2]}
    assert bt.sql(query, t=t, params=[1]).to_pydict() == {"n": [5]}
    typed = "SELECT ? AS v"
    assert bt.sql(typed, params=[1]).collect().schema.field("v").type == pa.int64()
    assert bt.sql(typed, params=[True]).collect().schema.field("v").type == pa.bool_()


def test_session_and_dataset_bind_the_same_way(duck, t):
    query = "SELECT id FROM t WHERE name = ?"
    expected = duck.sql(query, params=["ann"])
    session = bt.Session()
    session.register("t", bt.from_arrow(t))
    assert_same(session.sql(query, params=["ann"]).collect(), expected)
    assert_same(bt.from_arrow(t).sql(query, table_name="t", params=["ann"]).collect(), expected)


def test_parameters_reach_dml(t):
    session = bt.Session()
    session.register("t", bt.from_arrow(t))
    session.sql("INSERT INTO t (id, name) VALUES (?, ?)", params=[7, "it's"])
    got = session.sql("SELECT name FROM t WHERE id = 7").to_pydict()
    assert got == {"name": ["it's"]}


@pytest.mark.parametrize(
    ("query", "params", "match"),
    [
        ("SELECT ?", None, "no params= were given"),
        ("SELECT 1", [1], "no parameter placeholders"),
        ("SELECT ?, $x", [1], "mix parameter styles"),
        ("SELECT ?", [1, 2], "1 \\? placeholder"),
        ("SELECT $2", [1, 2], "each of \\$1..\\$2"),
        ("SELECT $x", {"y": 1}, "missing value"),
        ("SELECT $x", [1], "take a mapping"),
        ("SELECT ?", {"x": 1}, "take a sequence"),
        ("SELECT ?", "a", "take a sequence"),
        ("SELECT ?", [object()], "cannot bind a value of type object"),
        ("SELECT ?", [b"\xff\x00"], "valid UTF-8"),
        ("SELECT ?", [Decimal("NaN")], "non-finite Decimal"),
    ],
)
def test_a_binding_mismatch_is_a_plan_error(query, params, match):
    with pytest.raises(bt.PlanError, match=match):
        bt.sql(query, params=params)


def test_tables_with_reserved_names_bind_through_the_mapping(t):
    """``tables=`` reaches the names a keyword cannot: dialect, params, query, my-table."""
    ds = bt.from_arrow(t)
    names = {"dialect": ds, "params": ds, "query": ds, "my-table": ds}
    query = 'SELECT count(*) AS n FROM dialect, params, query, "my-table" WHERE dialect.id = 1'
    assert bt.sql(query, names).to_pydict() == {"n": [216]}
    assert bt.Session().sql(query, tables=names).to_pydict() == {"n": [216]}
