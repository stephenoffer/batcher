"""Value window functions (first_value/last_value/lag/lead) vs DuckDB."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "p": [1, 1, 1, 2, 2, 2],
            "v": [10, 20, 30, 40, 50, 60],
            "s": ["a", "b", "c", "d", "e", "f"],
        }
    )
    duck.register("t", tbl)
    return tbl


def test_first_value(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .window(partition_by=["p"], order_by=[("v", False)], functions={"f": ("first_value", "v")})
        .select("p", "v", "f")
        .collect()
    )
    assert_same(
        out, duck.sql("SELECT p, v, first_value(v) OVER (PARTITION BY p ORDER BY v) f FROM t")
    )


def test_expr_value_functions_match_dict_api_and_duckdb(duck, t):
    # The Expr-level constructors (lag/lead/first_value over .over(...)) lower to the
    # same Window operator as the dict API and match DuckDB.
    from batcher import col, first_value, lag, lead
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .with_columns(
            prev=lag(col("v"), 1).over(partition_by=["p"], order_by=["v"]),
            nxt=lead(col("v"), 2).over(partition_by=["p"], order_by=["v"]),
            fst=first_value(col("v")).over(partition_by=["p"], order_by=["v"]),
        )
        .select("p", "v", "prev", "nxt", "fst")
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT p, v, "
            "lag(v, 1) OVER (PARTITION BY p ORDER BY v) prev, "
            "lead(v, 2) OVER (PARTITION BY p ORDER BY v) nxt, "
            "first_value(v) OVER (PARTITION BY p ORDER BY v) fst FROM t"
        ),
    )


def test_last_value_whole_frame(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .window(partition_by=["p"], order_by=[("v", False)], functions={"l": ("last_value", "v")})
        .select("p", "v", "l")
        .collect()
    )
    # Batcher's last_value uses the whole-partition frame; match with explicit ROWS frame.
    assert_same(
        out,
        duck.sql(
            "SELECT p, v, last_value(v) OVER (PARTITION BY p ORDER BY v "
            "ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) l FROM t"
        ),
    )


@pytest.mark.parametrize("n", [1, 2, 3])
def test_lag(duck, t, n):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .window(partition_by=["p"], order_by=[("v", False)], functions={"lg": ("lag", "v", n)})
        .select("p", "v", "lg")
        .collect()
    )
    assert_same(
        out, duck.sql(f"SELECT p, v, lag(v, {n}) OVER (PARTITION BY p ORDER BY v) lg FROM t")
    )


@pytest.mark.parametrize("n", [1, 2])
def test_lead(duck, t, n):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .window(partition_by=["p"], order_by=[("v", False)], functions={"ld": ("lead", "v", n)})
        .select("p", "v", "ld")
        .collect()
    )
    assert_same(
        out, duck.sql(f"SELECT p, v, lead(v, {n}) OVER (PARTITION BY p ORDER BY v) ld FROM t")
    )


def test_lag_lead_on_string_column(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .window(
            partition_by=["p"],
            order_by=[("s", False)],
            functions={"prev": ("lag", "s"), "nxt": ("lead", "s")},
        )
        .select("p", "s", "prev", "nxt")
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT p, s, lag(s) OVER w prev, lead(s) OVER w nxt "
            "FROM t WINDOW w AS (PARTITION BY p ORDER BY s)"
        ),
    )


@pytest.mark.parametrize("fn", ["first_value", "lag", "lead"])
def test_sql_value_window(duck, t, fn):
    from conftest import assert_same

    q = f"SELECT p, v, {fn}(v) OVER (PARTITION BY p ORDER BY v) r FROM t"
    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))
