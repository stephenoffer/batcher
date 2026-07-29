"""`ds.rollup` / `ds.cube` / `ds.grouping_sets`, against DuckDB's SQL spelling.

The SQL front-end has had multi-level grouping all along; the DataFrame surface had no
spelling for it, so a subtotal report had to be written as a hand-rolled `union` of
`group_by`s. These tests assert the DataFrame form answers exactly what DuckDB's
`GROUP BY ROLLUP/CUBE/GROUPING SETS` answers, including the NULL that marks a subtotal.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same


@pytest.fixture
def sales(duck):
    t = pa.table(
        {
            "region": ["east", "east", "west", "west", "west"],
            "city": ["nyc", "bos", "sfo", "sfo", "lax"],
            "amount": [10, 20, 30, 40, 50],
        }
    )
    duck.register("sales", t)
    return t


def test_rollup_matches_duckdb(duck, sales):
    got = bt.from_arrow(sales).rollup("region", "city").agg(total=bt.col("amount").sum())
    want = duck.sql(
        "SELECT region, city, sum(amount) AS total FROM sales GROUP BY ROLLUP(region, city)"
    )
    assert_same(got.collect(), want)


def test_cube_matches_duckdb(duck, sales):
    got = bt.from_arrow(sales).cube("region", "city").agg(total=bt.col("amount").sum())
    want = duck.sql(
        "SELECT region, city, sum(amount) AS total FROM sales GROUP BY CUBE(region, city)"
    )
    assert_same(got.collect(), want)


def test_grouping_sets_matches_duckdb(duck, sales):
    got = (
        bt.from_arrow(sales)
        .grouping_sets(["region"], ["city"], [])
        .agg(total=bt.col("amount").sum())
    )
    want = duck.sql(
        "SELECT region, city, sum(amount) AS total FROM sales "
        "GROUP BY GROUPING SETS ((region), (city), ())"
    )
    assert_same(got.collect(), want)


def test_rollup_of_one_key_is_the_group_plus_the_total(duck, sales):
    got = bt.from_arrow(sales).rollup("region").agg(n=bt.col("amount").count())
    want = duck.sql("SELECT region, count(amount) AS n FROM sales GROUP BY ROLLUP(region)")
    assert_same(got.collect(), want)


def test_several_aggregates_per_level(duck, sales):
    got = (
        bt.from_arrow(sales)
        .rollup("region", "city")
        .agg(total=bt.col("amount").sum(), biggest=bt.col("amount").max())
    )
    want = duck.sql(
        "SELECT region, city, sum(amount) AS total, max(amount) AS biggest "
        "FROM sales GROUP BY ROLLUP(region, city)"
    )
    assert_same(got.collect(), want)


def test_the_dataframe_form_equals_the_sql_form(sales):
    # The two front-ends must agree: same levels, same nulls, same numbers.
    frame = bt.from_arrow(sales).rollup("region", "city").agg(total=bt.col("amount").sum())
    sql = bt.sql(
        "SELECT region, city, sum(amount) AS total FROM sales GROUP BY ROLLUP(region, city)",
        sales=sales,
    )
    assert frame.sort("region", "city").to_pydict() == sql.sort("region", "city").to_pydict()


def test_a_subtotal_row_is_distinguishable_by_its_null_key(sales):
    out = bt.from_arrow(sales).rollup("region", "city").agg(total=bt.col("amount").sum())
    rows = out.to_pydict()
    totals = [
        t for r, c, t in zip(rows["region"], rows["city"], rows["total"], strict=True) if r is None
    ]
    assert totals == [150]  # exactly one grand-total row, and it is the sum of everything


def test_an_unknown_key_is_rejected_by_name(sales):
    with pytest.raises(Exception, match="not a column"):
        bt.from_arrow(sales).rollup("regionn").agg(n=bt.col("amount").sum())


def test_rollup_without_an_aggregate_says_so(sales):
    with pytest.raises(Exception, match="at least one aggregate"):
        bt.from_arrow(sales).rollup("region").agg()
