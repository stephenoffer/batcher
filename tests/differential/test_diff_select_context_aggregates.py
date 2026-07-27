"""An aggregate outside `group_by().agg()`, against DuckDB's spelling of the same thing.

Polars and pandas both let an aggregate stand where a column is expected, and mean two
different things by it depending on the context. Batcher raised in every such context.
Now:

* a `select` whose items are *all* aggregates is the whole-frame aggregation — DuckDB's
  `SELECT sum(x) FROM t`;
* anywhere else (`with_columns`, a mixed `select`, a `filter` predicate) the aggregate is
  computed over the whole frame and broadcast to every row — DuckDB's `sum(x) OVER ()`
  and, for a predicate, its uncorrelated scalar subquery.

Each case below is asserted against the DuckDB query it desugars to, so the claim is
about the *answer*, not about the plan shape.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same


@pytest.fixture
def t(duck):
    table = pa.table({"g": ["a", "a", "b", "b"], "x": [1, 2, 3, 10], "y": [1.5, 2.5, 3.5, 4.5]})
    duck.register("t", table)
    return table


def test_all_aggregate_select_is_the_whole_frame_aggregation(duck, t):
    got = bt.from_arrow(t).select(
        total=bt.col("x").sum(), avg=bt.col("y").mean(), n=bt.col("x").count()
    )
    assert_same(
        got.collect(), duck.sql("SELECT sum(x) AS total, avg(y) AS avg, count(x) AS n FROM t")
    )


def test_an_expression_over_aggregates_collapses_the_same_way(duck, t):
    got = bt.from_arrow(t).select(ratio=bt.col("x").sum() / bt.col("x").count())
    assert_same(got.collect(), duck.sql("SELECT sum(x) / count(x) AS ratio FROM t"))


def test_with_columns_broadcasts_the_aggregate_to_every_row(duck, t):
    got = bt.from_arrow(t).with_columns(total=bt.col("x").sum())
    assert_same(got.collect(), duck.sql("SELECT g, x, y, sum(x) OVER () AS total FROM t"))


def test_a_share_of_total_is_the_idiom_this_enables(duck, t):
    got = bt.from_arrow(t).with_columns(share=bt.col("x") / bt.col("x").sum())
    assert_same(got.collect(), duck.sql("SELECT g, x, y, x / sum(x) OVER () AS share FROM t"))


def test_a_mixed_select_broadcasts_rather_than_collapsing(duck, t):
    got = bt.from_arrow(t).select("g", total=bt.col("x").sum())
    assert_same(got.collect(), duck.sql("SELECT g, sum(x) OVER () AS total FROM t"))


def test_an_aggregate_in_a_filter_is_the_uncorrelated_subquery(duck, t):
    got = bt.from_arrow(t).filter(bt.col("x") > bt.col("x").mean())
    assert_same(got.collect(), duck.sql("SELECT * FROM t WHERE x > (SELECT avg(x) FROM t)"))


def test_a_grouped_aggregate_still_groups(duck, t):
    # The new reading must not disturb the explicit one.
    got = bt.from_arrow(t).group_by("g").agg(total=bt.col("x").sum())
    assert_same(got.collect(), duck.sql("SELECT g, sum(x) AS total FROM t GROUP BY g"))


def test_the_broadcast_result_survives_repartitioning(t):
    # The broadcast is a window over the whole frame, so it must not become per-morsel.
    one = bt.from_arrow(t).with_columns(total=bt.col("x").sum()).sort("x").to_pydict()
    many = (
        bt.from_arrow(t).repartition(3).with_columns(total=bt.col("x").sum()).sort("x").to_pydict()
    )
    assert one == many
    assert many["total"] == [16, 16, 16, 16]


def test_str_join_is_the_string_aggregate(duck, t):
    # Polars spells `string_agg` as `str.join`; it is an aggregate, so it collapses in a
    # select and groups in a group_by, like every other one.
    got = bt.from_arrow(t).group_by("g").agg(r=bt.col("g").str.join("-"))
    assert_same(got.collect(), duck.sql("SELECT g, string_agg(g, '-') AS r FROM t GROUP BY g"))


def test_str_join_over_the_whole_frame(duck, t):
    got = bt.from_arrow(t).select(r=bt.col("g").str.join(","))
    assert_same(got.collect(), duck.sql("SELECT string_agg(g, ',') AS r FROM t"))


def test_neg_is_the_unary_minus(duck, t):
    got = bt.from_arrow(t).select(r=bt.col("x").neg())
    assert_same(got.collect(), duck.sql("SELECT -x AS r FROM t"))
