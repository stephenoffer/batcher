"""Operator-mix: the aggregates that are not a fold — distinct counts, order statistics, moments.

`SUM`/`COUNT`/`MIN`/`MAX` merge by adding or comparing two numbers, and `ops-aggregation`
covers them. The public API's other aggregates do not: `COUNT(DISTINCT)` must remember every
value it has seen, `MEDIAN` and `QUANTILE_CONT` must order a group's values, and a sample
variance carries three moments through the merge. Each reaches a different state shape in the
mergeable algebra, so each gets a case, and `HAVING` closes the set as the filter that runs
over an aggregate's output rather than its input.

Every case groups by a real low-cardinality `lineitem` column or reduces to one row, so the
correctness gate compares a handful of rows. Engines whose SQL dialect lacks a function report
it as a `PARTIAL` row, which is the result, not a gap in the harness.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from registry import suite

from .base import sql_fanout

if TYPE_CHECKING:
    from context import Context

stats = suite("ops-statistics", dataset="operators")


@stats.case("op-count-distinct")
def count_distinct(ctx: Context):
    """COUNT(DISTINCT l_partkey) — 200,000 distinct values out of 6M, one row out."""
    return sql_fanout(ctx, "SELECT COUNT(DISTINCT l_partkey) AS n FROM lineitem")


@stats.case("op-count-distinct-grouped")
def count_distinct_grouped(ctx: Context):
    """COUNT(DISTINCT l_suppkey) per return flag — a distinct set per group, merged per group."""
    return sql_fanout(
        ctx,
        "SELECT l_returnflag, COUNT(DISTINCT l_suppkey) AS n FROM lineitem GROUP BY l_returnflag",
    )


@stats.case("op-median-grouped")
def median_grouped(ctx: Context):
    """MEDIAN(l_extendedprice) per ship mode — an exact order statistic over ~860k rows a group."""
    return sql_fanout(
        ctx,
        "SELECT l_shipmode, MEDIAN(l_extendedprice) AS m FROM lineitem GROUP BY l_shipmode",
    )


@stats.case("op-quantile-grouped")
def quantile_grouped(ctx: Context):
    """QUANTILE_CONT(l_extendedprice, 0.9) per ship mode — the interpolated order statistic."""
    return sql_fanout(
        ctx,
        "SELECT l_shipmode, QUANTILE_CONT(l_extendedprice, 0.9) AS q "
        "FROM lineitem GROUP BY l_shipmode",
    )


@stats.case("op-stddev-grouped")
def stddev_grouped(ctx: Context):
    """STDDEV_SAMP and VAR_SAMP per ship mode — moments that merge as (n, sum, sum of squares)."""
    return sql_fanout(
        ctx,
        "SELECT l_shipmode, STDDEV_SAMP(l_extendedprice) AS s, VAR_SAMP(l_quantity) AS v "
        "FROM lineitem GROUP BY l_shipmode",
    )


@stats.case("op-having")
def having(ctx: Context):
    """GROUP BY l_suppkey HAVING SUM(l_quantity) > 15300 — keeps about half of 10,000 groups."""
    return sql_fanout(
        ctx,
        "SELECT l_suppkey, SUM(l_quantity) AS q FROM lineitem "
        "GROUP BY l_suppkey HAVING SUM(l_quantity) > 15300",
    )
