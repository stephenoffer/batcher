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

from .base import cannot_run, sql_fanout

if TYPE_CHECKING:
    from context import Context

window = suite("ops-window", dataset="operators")

# Daft cannot complete an *ordered* window over the 6M-row `lineitem` on a 30 GiB box.
# Measured here, each case alone in a fresh process (`rank` and `runsum` reach a peak RSS of
# 22.2 GB; `lag` exceeds the box entirely and is SIGKILLed, exit 137), against Batcher's
# ~90 ms and DuckDB's ~135 ms. Alone, `rank` and `runsum` just fit; inside the suite — which
# holds four other engines' 6M-row results at the same time — they do not, and the OOM killer
# takes the *runner*, not the query. That is what silently truncated the `operators` run.
#
# Capping the child's address space does not convert this into an error either: under a
# 12 GiB `RLIMIT_AS`, Daft thrashed for over ten minutes on a query the other engines answer
# in under a second. So there is no in-process way to observe this failure and continue, and
# the honest handling is to state it. `op-window-sum-partition` is *not* excluded: its
# frameless whole-partition aggregate peaks at 3.2 GB and Daft is timed on it normally.
_DAFT_ORDERED_WINDOW_OOM = (
    "OOM: daft needs ~22 GB for an ordered 6M-row window (lag exceeds 30 GB and is "
    "SIGKILLed); it cannot share the box with the other engines' results"
)


@window.case("op-window-rank")
def window_rank(ctx: Context):
    """Rank line items by price within each order (ranking — ties share min rank)."""
    return cannot_run(
        sql_fanout(
            ctx,
            "SELECT l_orderkey, rank() OVER "
            "(PARTITION BY l_orderkey ORDER BY l_extendedprice DESC) AS r FROM lineitem",
        ),
        "daft",
        _DAFT_ORDERED_WINDOW_OOM,
    )


@window.case("op-window-runsum")
def window_runsum(ctx: Context):
    """Running revenue within each order (cumulative aggregate over an ordered partition)."""
    return cannot_run(
        sql_fanout(
            ctx,
            "SELECT l_orderkey, sum(l_extendedprice) OVER "
            "(PARTITION BY l_orderkey ORDER BY l_linenumber) AS rs FROM lineitem",
        ),
        "daft",
        _DAFT_ORDERED_WINDOW_OOM,
    )


@window.case("op-window-lag")
def window_lag(ctx: Context):
    """Previous line's price within each order (positional lag over an ordered partition)."""
    return cannot_run(
        sql_fanout(
            ctx,
            "SELECT l_orderkey, lag(l_extendedprice) OVER "
            "(PARTITION BY l_orderkey ORDER BY l_linenumber) AS lg FROM lineitem",
        ),
        "daft",
        _DAFT_ORDERED_WINDOW_OOM,
    )


@window.case("op-window-sum-partition")
def window_sum_partition(ctx: Context):
    """Order total broadcast to every line (frameless whole-partition aggregate)."""
    return sql_fanout(
        ctx,
        "SELECT l_orderkey, sum(l_extendedprice) OVER (PARTITION BY l_orderkey) AS s FROM lineitem",
    )
