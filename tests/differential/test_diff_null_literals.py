"""A bare ``None`` and ``bt.lit(None)`` in an expression, against DuckDB's NULL (AP-184/196).

Before the fix, ``col == None``, ``eq_missing(None)``, ``coalesce(col, None)`` and
``fill_null(None)`` raised "a literal must be ... None" -- rejecting the value the message
listed as allowed -- and ``bt.lit(None)``, an Int64 NULL, failed in the engine against a
string or date column (``Utf8 == Int64``). SQL's answer is NULL whatever the other side's
type, so each shape is checked on an integer, a string and a date column, over a table with
nulls and duplicates, a single row, and an empty relation.

The nested and decimal literals of AP-185 are checked here too: ``lit([1, 2])``,
``lit({"a": 1})`` and an exact ``lit(Decimal, dtype="decimal(p,s)")``.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

TABLE = pa.table(
    {
        "i": pa.array([0, 1, 2, 3], pa.int64()),
        "n": pa.array([1, None, 1, 7], pa.int64()),
        "s": pa.array(["a", None, "a", "b"], pa.string()),
        "d": pa.array([dt.date(2024, 1, 1), None, dt.date(2024, 1, 1), dt.date(2024, 2, 1)]),
    }
)

_SHAPES = [
    ("eq_none", lambda c: c == None, "{c} = NULL"),  # noqa: E711
    ("ne_none", lambda c: c != None, "{c} <> NULL"),  # noqa: E711
    ("ge_none", lambda c: c >= None, "{c} >= NULL"),
    ("eq_lit_none", lambda c: c == bt.lit(None), "{c} = NULL"),
    ("eq_missing_none", lambda c: c.eq_missing(None), "{c} IS NOT DISTINCT FROM NULL"),
    ("eq_missing_lit_none", lambda c: c.eq_missing(bt.lit(None)), "{c} IS NOT DISTINCT FROM NULL"),
    ("coalesce_none", lambda c: bt.coalesce(c, None), "coalesce({c}, NULL)"),
    ("coalesce_lit_none", lambda c: bt.coalesce(c, bt.lit(None)), "coalesce({c}, NULL)"),
    ("fill_null_none", lambda c: c.fill_null(None), "coalesce({c}, NULL)"),
]


@pytest.fixture
def table(duck):
    duck.register("t", TABLE)
    return bt.from_arrow(TABLE)


@pytest.mark.parametrize("column", ["n", "s", "d"])
@pytest.mark.parametrize(("name", "build", "sql"), _SHAPES, ids=[s[0] for s in _SHAPES])
def test_null_operand_matches_duckdb(duck, table, column, name, build, sql):
    """Each NULL shape on an int, string and date column, with nulls and duplicates."""
    out = table.select("i", r=build(bt.col(column))).collect()
    assert_same(out, duck.sql(f"SELECT i, {sql.format(c=column)} AS r FROM t"))


@pytest.mark.parametrize(("name", "build", "sql"), _SHAPES, ids=[s[0] for s in _SHAPES])
def test_null_operand_on_one_row_and_on_empty(duck, table, name, build, sql):
    """A one-row null input and an empty relation give DuckDB's answer too."""
    for where in ("i = 1", "i > 100"):
        out = table.filter(bt.sql_expr(where)).select("i", r=build(bt.col("s"))).collect()
        assert_same(out, duck.sql(f"SELECT i, {sql.format(c='s')} AS r FROM t WHERE {where}"))


def test_a_comparison_with_none_keeps_the_column_name():
    """``select(col("s") == None)`` is named after `s`, as ``col("s") == "a"`` is."""
    assert bt.from_arrow(TABLE).select(bt.col("s") == None).columns == ["s"]  # noqa: E711


def test_bare_none_elsewhere_is_an_int64_null(duck, table):
    """Where no operator types it, ``None`` is the Int64 NULL ``lit(None)`` always was."""
    out = table.select("i", r=bt.col("n") + None).collect()
    assert out.schema.field("r").type == pa.int64()
    assert_same(out, duck.sql("SELECT i, n + NULL::BIGINT AS r FROM t"))


def test_unsupported_literal_raises_where_it_is_written():
    """The message no longer lists None as allowed while rejecting it, and it is eager."""
    with pytest.raises(bt.PlanError, match="cannot use bytes") as info:
        bt.lit(b"raw")
    assert "boolean, None" not in str(info.value)


@pytest.mark.parametrize(
    ("value", "sql"),
    [
        ([1, 2], "[1, 2]"),
        ([[1, 2], [3]], "[[1, 2], [3]]"),
        ({"a": 1, "b": "x"}, "{'a': 1, 'b': 'x'}"),
        ([1.5, None], "[1.5, NULL]"),
    ],
    ids=["list", "nested_list", "struct", "list_with_null"],
)
def test_nested_literals_match_duckdb(duck, table, value, sql):
    """A Python list or dict given to ``lit`` is a list or struct literal."""
    out = table.select("i", r=bt.lit(value)).collect()
    assert_same(out, duck.sql(f"SELECT i, {sql} AS r FROM t"))


def test_decimal_literal_is_exact(duck, table):
    """``lit(Decimal, dtype="decimal(p,s)")`` keeps the type and every digit."""
    value = Decimal("123456789012345678.25")
    out = table.select("i", r=bt.lit(value, dtype="decimal(38,2)")).collect()
    assert out.schema.field("r").type == pa.decimal128(38, 2)
    assert set(out.column("r").to_pylist()) == {value}
    assert_same(out, duck.sql("SELECT i, 123456789012345678.25::DECIMAL(38,2) AS r FROM t"))


def test_decimal_dtype_types_a_null_and_rejects_nonsense():
    assert bt.from_arrow(TABLE).select(r=bt.lit(None, dtype="decimal(10, 2)")).schema.field(
        "r"
    ).type == pa.decimal128(10, 2)
    with pytest.raises(bt.PlanError, match="decimal"):
        bt.lit(Decimal("1.5"), dtype="decimal(99,2)")
    with pytest.raises(bt.PlanError, match="scalar literal"):
        bt.lit([1], dtype="int64")
