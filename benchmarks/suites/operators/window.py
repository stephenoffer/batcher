"""Operator-mix: window functions over TPC-H ``lineitem`` — Polars/DuckDB's turf.

Window functions (ranking, running aggregates, positional lag/lead) are a signature
DataFrame workload. Each case orders within ``l_orderkey`` partitions by a column that
is unique inside a partition (``l_linenumber``) or resolves ties deterministically
(``rank`` shares the min rank), so the per-row result is a single well-defined answer
the correctness gate compares as a multiset across engines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from registry import suite

from .base import sql_fanout

if TYPE_CHECKING:
    from context import Context

window = suite("ops-window", dataset="operators")


@window.case("op-window-rank")
def window_rank(ctx: Context):
    """Rank line items by price within each order (ranking — ties share min rank)."""
    return sql_fanout(
        ctx,
        "SELECT l_orderkey, rank() OVER "
        "(PARTITION BY l_orderkey ORDER BY l_extendedprice DESC) AS r FROM lineitem",
    )


@window.case("op-window-runsum")
def window_runsum(ctx: Context):
    """Running revenue within each order (cumulative aggregate over an ordered partition)."""
    return sql_fanout(
        ctx,
        "SELECT l_orderkey, sum(l_extendedprice) OVER "
        "(PARTITION BY l_orderkey ORDER BY l_linenumber) AS rs FROM lineitem",
    )


@window.case("op-window-lag")
def window_lag(ctx: Context):
    """Previous line's price within each order (positional lag over an ordered partition)."""
    return sql_fanout(
        ctx,
        "SELECT l_orderkey, lag(l_extendedprice) OVER "
        "(PARTITION BY l_orderkey ORDER BY l_linenumber) AS lg FROM lineitem",
    )


@window.case("op-window-sum-partition")
def window_sum_partition(ctx: Context):
    """Order total broadcast to every line (frameless whole-partition aggregate)."""
    return sql_fanout(
        ctx,
        "SELECT l_orderkey, sum(l_extendedprice) OVER (PARTITION BY l_orderkey) AS s FROM lineitem",
    )
