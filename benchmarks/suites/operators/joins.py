"""Operator-mix: hash join + aggregate over TPC-H ``lineitem`` ⋈ ``orders``.

A join followed by a small grouped aggregate (revenue by order priority) keeps the
result tiny, so the correctness gate compares a handful of rows rather than the
multi-million-row join output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from registry import suite

from .base import sql_fanout

if TYPE_CHECKING:
    from context import Context

joins = suite("ops-joins", dataset="operators")


@joins.case("op-join-agg")
def join_agg(ctx: Context):
    """Revenue by order priority — the canonical join-then-aggregate pipeline."""
    sql = (
        "SELECT o.o_orderpriority, "
        "SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue "
        "FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey "
        "GROUP BY o.o_orderpriority"
    )
    fns = sql_fanout(ctx, sql)

    if "pyarrow" in ctx.names():
        lineitem, orders = ctx.table("lineitem"), ctx.table("orders")

        def pyarrow() -> pa.Table:
            # Both sides projected, not just the right. PyArrow has no optimizer, so a
            # column it is handed is a column it carries: joining the full 16-column
            # `lineitem` made it gather 13 columns nothing downstream reads, while every
            # SQL engine's planner projected to three. That is a handicap the comparator
            # cannot remove for itself, and one that flatters Batcher.
            joined = lineitem.select(["l_orderkey", "l_extendedprice", "l_discount"]).join(
                orders.select(["o_orderkey", "o_orderpriority"]),
                keys="l_orderkey",
                right_keys="o_orderkey",
                join_type="inner",
            )
            revenue = pc.multiply(joined["l_extendedprice"], pc.subtract(1.0, joined["l_discount"]))
            joined = joined.append_column("revenue", revenue)
            a = joined.group_by("o_orderpriority").aggregate([("revenue", "sum")])
            return pa.table({"o_orderpriority": a["o_orderpriority"], "revenue": a["revenue_sum"]})

        fns["pyarrow"] = pyarrow

    if "ray" in ctx.names():
        line_h = ctx.handle("lineitem", "ray")
        orders_h = ctx.handle("orders", "ray")

        def ray() -> pa.Table:
            from ray.data.aggregate import Sum

            orders_pri = orders_h.map_batches(
                lambda b: pa.table(
                    {"o_orderkey": b["o_orderkey"], "o_orderpriority": b["o_orderpriority"]}
                ),
                batch_format="pyarrow",
            )
            joined = line_h.join(
                orders_pri,
                join_type="inner",
                num_partitions=64,
                on=("l_orderkey",),
                right_on=("o_orderkey",),
            )

            def revenue(b: pa.Table) -> pa.Table:
                rev = pc.multiply(b["l_extendedprice"], pc.subtract(1.0, b["l_discount"]))
                return pa.table({"o_orderpriority": b["o_orderpriority"], "revenue": rev})

            g = (
                joined.map_batches(revenue, batch_format="pyarrow")
                .groupby("o_orderpriority")
                .aggregate(Sum("revenue"))
            )
            df = g.to_pandas().rename(columns={"sum(revenue)": "revenue"})
            return pa.Table.from_pandas(df, preserve_index=False)

        fns["ray"] = ray

    return fns


@joins.case("op-join-left-outer")
def join_left_outer(ctx: Context):
    """orders LEFT JOIN lineitem -- the null-producing side the inner join never builds.

    Filtered to a narrow ship-date window on the right so the probe genuinely misses for
    most orders, which is what makes this an outer join rather than an inner one wearing
    the keyword.
    """
    sql = (
        "SELECT o.o_orderpriority, COUNT(l.l_orderkey) AS matched, COUNT(*) AS total "
        "FROM orders o LEFT JOIN ("
        "  SELECT l_orderkey FROM lineitem WHERE l_shipdate < DATE '1992-04-01'"
        ") l ON l.l_orderkey = o.o_orderkey "
        "GROUP BY o.o_orderpriority"
    )
    return sql_fanout(ctx, sql)


@joins.case("op-join-semi")
def join_semi(ctx: Context):
    """EXISTS -- a semi-join, which must stop at the first match rather than fan out."""
    sql = (
        "SELECT COUNT(*) AS n FROM orders o WHERE EXISTS ("
        "SELECT 1 FROM lineitem l WHERE l.l_orderkey = o.o_orderkey AND l.l_quantity > 45)"
    )
    return sql_fanout(ctx, sql)


@joins.case("op-join-anti")
def join_anti(ctx: Context):
    """NOT EXISTS -- an anti-join, the complement of the case above over the same inputs."""
    sql = (
        "SELECT COUNT(*) AS n FROM orders o WHERE NOT EXISTS ("
        "SELECT 1 FROM lineitem l WHERE l.l_orderkey = o.o_orderkey AND l.l_quantity > 45)"
    )
    return sql_fanout(ctx, sql)


@joins.case("op-join-multikey")
def join_multikey(ctx: Context):
    """lineitem join partsupp on (partkey, suppkey) -- a composite key the hash must pack."""
    sql = (
        "SELECT COUNT(*) AS n, SUM(ps.ps_supplycost) AS c "
        "FROM lineitem l JOIN partsupp ps "
        "ON l.l_partkey = ps.ps_partkey AND l.l_suppkey = ps.ps_suppkey"
    )
    return sql_fanout(ctx, sql)


@joins.case("op-join-range")
def join_range(ctx: Context):
    """An inequality join against a small bucket table -- the shape with no hash to build.

    The scorecard records this as a loss above about one million probe rows
    (``competitive_architecture.md`` ceiling 7), so the suite needs a case that sits there.
    The build side is six buckets, which keeps the output small while forcing every probe
    row through a range comparison rather than an equality lookup.
    """
    sql = (
        "SELECT b.lo, COUNT(*) AS n FROM lineitem l JOIN ("
        "  SELECT 0.00 AS lo, 0.02 AS hi UNION ALL SELECT 0.02, 0.04 "
        "  UNION ALL SELECT 0.04, 0.06 UNION ALL SELECT 0.06, 0.08 "
        "  UNION ALL SELECT 0.08, 0.10 UNION ALL SELECT 0.10, 0.12"
        ") b ON l.l_discount >= b.lo AND l.l_discount < b.hi "
        "GROUP BY b.lo"
    )
    return sql_fanout(ctx, sql)


@joins.case("op-join-build-large")
def join_build_large(ctx: Context):
    """lineitem joined to itself on orderkey -- a build side far larger than any dimension.

    Every other join case here builds on a dimension table that fits in cache. This one
    forces a multi-million-row hash table, which is where build-side partitioning, spill,
    and the choice of build side actually decide the time.
    """
    sql = (
        "SELECT COUNT(*) AS n FROM ("
        "  SELECT l_orderkey, SUM(l_quantity) AS q FROM lineitem GROUP BY l_orderkey"
        ") a JOIN ("
        "  SELECT l_orderkey, MAX(l_discount) AS d FROM lineitem GROUP BY l_orderkey"
        ") b ON a.l_orderkey = b.l_orderkey WHERE a.q > 20 AND b.d > 0.02"
    )
    return sql_fanout(ctx, sql)
