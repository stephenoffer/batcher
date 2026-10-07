"""Differential tests vs DuckDB for the semi-join build-side rewrites (`join_elim.semi_build`).

Both rewrites are sound only because a semi join reads its right side as a set of keys, so
every case here runs the whole optimizer, asserts on the plan that the rewrite did (or did
not) fire, and compares the executed answer with DuckDB computing the query as written. The
tables carry what would break a careless version: a NULL compared column (``<>`` against
NULL is not true), a NULL key (it joins nothing, yet a ``GROUP BY`` keeps it as a group),
keys held by one row, keys whose rows all share one value, and duplicate rows.

The shape is TPC-DS q95's: a CTE self-joining a table on a key with ``<>`` on another
column, read through ``IN (SELECT key FROM cte)`` and through ``IN`` over a join with a
second table.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher.io.source import source_statistics
from batcher.kyber.optimizer import Optimizer
from batcher.plan.logical import Aggregate, Join
from batcher.plan.visitor import walk

# Order 1: two warehouses. Order 2: one warehouse, repeated. Order 3: one row. Order 4: a
# warehouse and a NULL warehouse (no two non-null values differ). Order 5: three rows, two
# warehouses. NULL order: two different warehouses -- a group a GROUP BY keeps and a join
# never produces. Order 6: only NULL warehouses.
_SALES = {
    "ord": [1, 1, 2, 2, 3, 4, 4, 5, 5, 5, None, None, 6, 6],
    "wh": [10, 20, 30, 30, 40, 50, None, 60, 70, 60, 80, 90, None, None],
    "amt": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
}
_RETURNS = {"rord": [1, 1, 2, 5, 7, None]}

_CTE = """WITH ws_wh AS (
  SELECT a.ord AS ord, a.{col} AS w1, b.{col} AS w2
  FROM sales a, sales b
  WHERE a.ord = b.ord AND a.{col} <> b.{col})
"""


@pytest.fixture
def sess(duck):
    sales = pa.table(
        {
            **_SALES,
            "fwh": pa.array([None if w is None else float(w) for w in _SALES["wh"]]),
            "swh": pa.array([None if w is None else f"w{w}" for w in _SALES["wh"]]),
        }
    )
    returns = pa.table(_RETURNS)
    duck.register("sales", sales)
    duck.register("returns", returns)
    s = bt.Session()
    s.register("sales", sales)
    s.register("returns", returns)
    return s


def _plan(ds):
    stats = [source_statistics(s) for s in ds._sources]
    return Optimizer(sources=ds._sources, source_stats=stats).logical_rewrite(ds._plan)


def _rewritten(ds) -> bool:
    """Whether the self-join became the `min <> max` group-by (the rewrite fired)."""
    return any(
        isinstance(n, Aggregate) and any(s.alias == "__semi_min" for s in n.aggregates)
        for n in walk(_plan(ds))
    )


@pytest.mark.parametrize("column", ["wh", "swh"])
def test_in_self_join_neq_matches_duckdb(duck, sess, column):
    """`IN (SELECT ord FROM ws_wh)`: the orders with two different non-null values."""
    sql = _CTE.format(col=column) + (
        "SELECT ord, sum(amt) AS s, count(*) AS n FROM sales "
        "WHERE ord IN (SELECT ord FROM ws_wh) GROUP BY ord"
    )
    ds = sess.sql(sql)
    assert _rewritten(ds), "the self-join under the IN was not rewritten"
    assert_same(ds.collect(), duck.sql(sql))


def test_in_join_over_self_join_matches_duckdb(duck, sess):
    """q95's second consumer: `IN` over `returns JOIN ws_wh`, both rewrites in turn."""
    sql = _CTE.format(col="wh") + (
        "SELECT ord, count(*) AS n FROM sales WHERE ord IN "
        "(SELECT rord FROM returns, ws_wh WHERE rord = ws_wh.ord) GROUP BY ord"
    )
    ds = sess.sql(sql)
    assert _rewritten(ds)
    assert not any(isinstance(n, Join) and n.join_type == "inner" for n in walk(_plan(ds))), (
        "the inner join under the semi join's build side was not made a semi join"
    )
    assert_same(ds.collect(), duck.sql(sql))


def test_both_consumers_together_matches_duckdb(duck, sess):
    """The whole q95 shape: two `IN`s over the same CTE, plus a global aggregate."""
    sql = _CTE.format(col="wh") + (
        "SELECT count(DISTINCT ord) AS orders, sum(amt) AS total FROM sales "
        "WHERE ord IN (SELECT ord FROM ws_wh) "
        "AND ord IN (SELECT rord FROM returns, ws_wh WHERE rord = ws_wh.ord)"
    )
    assert_same(sess.sql(sql).collect(), duck.sql(sql))


def test_float_column_declines_and_still_matches(duck, sess):
    """A float `<>` is left alone (NaN and -0.0 make min/max disagree with it)."""
    sql = _CTE.format(col="fwh") + (
        "SELECT ord, count(*) AS n FROM sales WHERE ord IN (SELECT ord FROM ws_wh) GROUP BY ord"
    )
    ds = sess.sql(sql)
    assert not _rewritten(ds)
    assert_same(ds.collect(), duck.sql(sql))


def test_sides_that_differ_decline_and_still_match(duck, sess):
    """A filter on one side makes the two sides different relations: no rewrite."""
    sql = (
        "WITH ws_wh AS (SELECT a.ord AS ord FROM sales a, sales b "
        "WHERE a.ord = b.ord AND a.wh <> b.wh AND b.amt > 2.5) "
        "SELECT ord, count(*) AS n FROM sales WHERE ord IN (SELECT ord FROM ws_wh) GROUP BY ord"
    )
    ds = sess.sql(sql)
    assert not _rewritten(ds)
    assert_same(ds.collect(), duck.sql(sql))


def test_inner_join_in_build_side_matches_duckdb(duck, sess):
    """`IN` over an inner join with duplicate matches: the semi join must not fan out."""
    sql = (
        "SELECT ord, count(*) AS n FROM sales WHERE ord IN "
        "(SELECT s2.ord FROM sales s2, returns WHERE s2.ord = rord) GROUP BY ord"
    )
    ds = sess.sql(sql)
    assert not any(isinstance(n, Join) and n.join_type == "inner" for n in walk(_plan(ds)))
    assert_same(ds.collect(), duck.sql(sql))


def test_empty_self_join_matches_duckdb(duck, sess):
    """No key holds two values once filtered: an empty build side, so an empty answer."""
    sql = _CTE.format(col="wh").replace("a.ord = b.ord", "a.ord = b.ord AND a.ord = 2") + (
        "SELECT ord, count(*) AS n FROM sales WHERE ord IN (SELECT ord FROM ws_wh) GROUP BY ord"
    )
    assert_same(sess.sql(sql).collect(), duck.sql(sql))
