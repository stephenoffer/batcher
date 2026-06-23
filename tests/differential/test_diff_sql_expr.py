"""SQL expression completeness: `||` concat, CASE-without-ELSE, NULL, FILTER."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "g": [1, 1, 2, 2, 2],
            "v": [10, 25, 30, 5, 40],
            "a": ["x", "y", "z", "p", "q"],
            "b": ["1", "2", "3", "4", "5"],
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.mark.parametrize(
    "q",
    [
        "SELECT a || b AS r FROM t",
        "SELECT a || '!' AS r FROM t",
        "SELECT a || b || a AS r FROM t",
        "SELECT a || v AS r FROM t",  # string || int → cast
        "SELECT a FROM t WHERE a || b = 'x1'",
    ],
)
def test_string_concat(duck, t, q):
    from conftest import assert_same

    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


def test_concat_dataframe_roundtrip(duck, t):
    """`Binary("concat", ...)` round-trips through the JSON IR to the engine."""
    from batcher import col
    from batcher.plan.expr_ir import Binary
    from conftest import assert_same

    out = bt.from_arrow(t).select("g", r=Binary("concat", col("a"), col("b"))).collect()
    assert_same(out, duck.sql("SELECT g, a || b AS r FROM t"))


@pytest.mark.parametrize(
    "q",
    [
        "SELECT g, CASE WHEN v > 20 THEN 1 END c FROM t",
        "SELECT g, CASE WHEN v > 20 THEN 'big' END c FROM t",
        "SELECT g, CASE WHEN v > 25 THEN 2 WHEN v > 15 THEN 1 END c FROM t",
        "SELECT g, CASE WHEN v > 20 THEN 1 ELSE 0 END c FROM t",  # ELSE still works
    ],
)
def test_case_without_else(duck, t, q):
    from conftest import assert_same

    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        "SELECT g, NULL AS n FROM t",
        "SELECT coalesce(NULL, v) AS r FROM t",
    ],
)
def test_null_literal(duck, t, q):
    from conftest import assert_same

    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        "SELECT COUNT(*) FILTER (WHERE v > 20) n FROM t",
        "SELECT SUM(v) FILTER (WHERE v > 20) s FROM t",
        "SELECT g, COUNT(*) FILTER (WHERE v > 20) n, SUM(v) s FROM t GROUP BY g",
        "SELECT COUNT(*) FILTER (WHERE v > 20) hi, COUNT(*) FILTER (WHERE v <= 20) lo FROM t",
    ],
)
def test_aggregate_filter(duck, t, q):
    from conftest import assert_same

    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))
