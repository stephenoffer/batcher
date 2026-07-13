"""Common-subexpression elimination does not change the answer.

CSE binds a repeated subexpression to a synthetic column and rewrites the outputs to read
it. That is only sound if the bound expression is a pure function of the row — so these
check the result against DuckDB on exactly the inputs where an unsound binding would show:
NULLs (which must propagate through the binding identically), empty input (where the extra
projection must still produce the right schema), duplicates (a per-row value must not be
shared across rows), and a filter below the projection (the binding must see only the rows
that survived).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from conftest import assert_same

pytestmark = pytest.mark.differential


@pytest.fixture
def t(duck):
    table = pa.table(
        {
            "s": pa.array(["a-b-c", "x-y-z", None, "a-b-c", ""], type=pa.string()),
            "n": pa.array([1, 2, None, 1, 5], type=pa.int64()),
        }
    )
    duck.register("t", table)
    return bt.from_arrow(table)


@pytest.fixture
def empty(duck):
    table = pa.table({"s": pa.array([], type=pa.string()), "n": pa.array([], type=pa.int64())})
    duck.register("e", table)
    return bt.from_arrow(table)


def test_repeated_string_expression(t, duck):
    """The shape CSE targets: one expensive expression feeding three outputs."""
    e = col("s").str.regexp_replace("-", "+")
    got = t.select(a=e, b=e.str.upper(), c=e.str.len()).collect()
    assert_same(
        got,
        duck.sql(
            "SELECT regexp_replace(s, '-', '+') AS a, "
            "       upper(regexp_replace(s, '-', '+')) AS b, "
            "       length(regexp_replace(s, '-', '+')) AS c "
            "FROM t"
        ),
    )


def test_repeated_expression_over_nulls_and_empty_strings(t, duck):
    e = col("s").str.regexp_replace("a", "Z")
    got = t.select(x=e, y=e.str.len(), z=e.str.upper()).collect()
    assert_same(
        got,
        duck.sql(
            "SELECT regexp_replace(s, 'a', 'Z') AS x, "
            "       length(regexp_replace(s, 'a', 'Z')) AS y, "
            "       upper(regexp_replace(s, 'a', 'Z')) AS z "
            "FROM t"
        ),
    )


def test_nested_repeated_expressions(t, duck):
    """A binding that reads another binding — the let-chain."""
    inner = col("s").str.regexp_replace("-", "+")
    outer = inner.str.regexp_replace("a", "z")
    got = t.select(a=inner, b=inner.str.upper(), c=outer, d=outer.str.len()).collect()
    assert_same(
        got,
        duck.sql(
            "SELECT regexp_replace(s, '-', '+') AS a, "
            "       upper(regexp_replace(s, '-', '+')) AS b, "
            "       regexp_replace(regexp_replace(s, '-', '+'), 'a', 'z') AS c, "
            "       length(regexp_replace(regexp_replace(s, '-', '+'), 'a', 'z')) AS d "
            "FROM t"
        ),
    )


def test_empty_input_still_produces_the_right_schema(empty, duck):
    e = col("s").str.regexp_replace("-", "+")
    got = empty.select(a=e, b=e.str.upper()).collect()
    assert_same(
        got,
        duck.sql(
            "SELECT regexp_replace(s, '-', '+') AS a, "
            "       upper(regexp_replace(s, '-', '+')) AS b FROM e"
        ),
    )


def test_repeated_expression_under_a_filter(t, duck):
    """The binding sits above a filter, so it must see only the surviving rows."""
    e = col("s").str.regexp_replace("-", "+")
    got = t.filter(col("n") > 1).select(a=e, b=e.str.upper(), c=e.str.len()).collect()
    assert_same(
        got,
        duck.sql(
            "SELECT regexp_replace(s, '-', '+') AS a, "
            "       upper(regexp_replace(s, '-', '+')) AS b, "
            "       length(regexp_replace(s, '-', '+')) AS c "
            "FROM t WHERE n > 1"
        ),
    )
