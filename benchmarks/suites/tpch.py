"""TPC-H subset benchmarks (Q1, Q3, Q6, Q10).

Date predicates follow the TPC-H spec. The dataset (``tpch``) casts decimals to
float64 and dates to timestamp[us] up front so all three engines read identical
inputs; see ``contexts.TpchContext``.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import polars as pl

from batcher import col, count, lit
from registry import suite

if TYPE_CHECKING:
    from contexts import TpchContext

tpch = suite("tpch", dataset="tpch")


@tpch.case("tpch-q1")
def q1_pricing_summary(ctx: TpchContext):
    """Q1: pricing summary report (grouped aggregate over lineitem)."""

    def batcher():
        line = ctx.bt_t["lineitem"].filter(col("l_shipdate") <= lit(dt.datetime(1998, 9, 2)))
        return (
            line.with_columns(
                disc_price=col("l_extendedprice") * (lit(1.0) - col("l_discount")),
                charge=col("l_extendedprice")
                * (lit(1.0) - col("l_discount"))
                * (lit(1.0) + col("l_tax")),
            )
            .group_by("l_returnflag", "l_linestatus")
            .agg(
                sum_qty=col("l_quantity").sum(),
                sum_base_price=col("l_extendedprice").sum(),
                sum_disc_price=col("disc_price").sum(),
                sum_charge=col("charge").sum(),
                avg_qty=col("l_quantity").mean(),
                avg_price=col("l_extendedprice").mean(),
                avg_disc=col("l_discount").mean(),
                count_order=count(),
            )
            .collect()
        )

    def duckdb():
        return ctx.con.sql(
            """SELECT l_returnflag, l_linestatus,
                      sum(l_quantity) AS sum_qty,
                      sum(l_extendedprice) AS sum_base_price,
                      sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
                      sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
                      avg(l_quantity) AS avg_qty,
                      avg(l_extendedprice) AS avg_price,
                      avg(l_discount) AS avg_disc,
                      count(*) AS count_order
               FROM lineitem
               WHERE l_shipdate <= TIMESTAMP '1998-09-02'
               GROUP BY l_returnflag, l_linestatus"""
        ).to_arrow_table()

    def polars():
        line = ctx.pl_t["lineitem"].lazy().filter(pl.col("l_shipdate") <= dt.datetime(1998, 9, 2))
        return (
            line.group_by("l_returnflag", "l_linestatus")
            .agg(
                pl.col("l_quantity").sum().alias("sum_qty"),
                pl.col("l_extendedprice").sum().alias("sum_base_price"),
                (pl.col("l_extendedprice") * (1 - pl.col("l_discount")))
                .sum()
                .alias("sum_disc_price"),
                (pl.col("l_extendedprice") * (1 - pl.col("l_discount")) * (1 + pl.col("l_tax")))
                .sum()
                .alias("sum_charge"),
                pl.col("l_quantity").mean().alias("avg_qty"),
                pl.col("l_extendedprice").mean().alias("avg_price"),
                pl.col("l_discount").mean().alias("avg_disc"),
                pl.len().alias("count_order"),
            )
            .collect()
            .to_arrow()
        )

    return {"batcher": batcher, "duckdb": duckdb, "polars": polars}


@tpch.case("tpch-q6")
def q6_forecasting_revenue(ctx: TpchContext):
    """Q6: forecasting revenue change (selective scalar aggregate)."""

    def batcher():
        return (
            ctx.bt_t["lineitem"]
            .filter(
                (col("l_shipdate") >= lit(dt.datetime(1994, 1, 1)))
                & (col("l_shipdate") < lit(dt.datetime(1995, 1, 1)))
                & col("l_discount").between(0.05, 0.07)
                & (col("l_quantity") < 24.0)
            )
            .with_columns(rev=col("l_extendedprice") * col("l_discount"))
            .group_by()
            .agg(revenue=col("rev").sum())
            .collect()
        )

    def duckdb():
        return ctx.con.sql(
            """SELECT sum(l_extendedprice * l_discount) AS revenue
               FROM lineitem
               WHERE l_shipdate >= TIMESTAMP '1994-01-01'
                 AND l_shipdate < TIMESTAMP '1995-01-01'
                 AND l_discount BETWEEN 0.05 AND 0.07
                 AND l_quantity < 24"""
        ).to_arrow_table()

    def polars():
        return (
            ctx.pl_t["lineitem"]
            .lazy()
            .filter(
                (pl.col("l_shipdate") >= dt.datetime(1994, 1, 1))
                & (pl.col("l_shipdate") < dt.datetime(1995, 1, 1))
                & pl.col("l_discount").is_between(0.05, 0.07)
                & (pl.col("l_quantity") < 24)
            )
            .select((pl.col("l_extendedprice") * pl.col("l_discount")).sum().alias("revenue"))
            .collect()
            .to_arrow()
        )

    return {"batcher": batcher, "duckdb": duckdb, "polars": polars}


@tpch.case("tpch-q3")
def q3_shipping_priority(ctx: TpchContext):
    """Q3: shipping priority (three-way join with grouped top-10)."""

    def batcher():
        cust = ctx.bt_t["customer"].filter(col("c_mktsegment") == lit("BUILDING"))
        orders = ctx.bt_t["orders"].filter(col("o_orderdate") < lit(dt.datetime(1995, 3, 15)))
        line = ctx.bt_t["lineitem"].filter(col("l_shipdate") > lit(dt.datetime(1995, 3, 15)))
        joined = (
            cust.join(orders, left_on="c_custkey", right_on="o_custkey", how="inner")
            .join(line, left_on="o_orderkey", right_on="l_orderkey", how="inner")
            .with_columns(rev=col("l_extendedprice") * (lit(1.0) - col("l_discount")))
        )
        return (
            joined.group_by("o_orderkey", "o_orderdate", "o_shippriority")
            .agg(revenue=col("rev").sum())
            .sort("revenue", "o_orderdate", descending=[True, False])
            .limit(10)
            .collect()
        )

    def duckdb():
        return ctx.con.sql(
            """SELECT l_orderkey AS o_orderkey, sum(l_extendedprice * (1 - l_discount)) AS revenue,
                      o_orderdate, o_shippriority
               FROM customer, orders, lineitem
               WHERE c_mktsegment = 'BUILDING'
                 AND c_custkey = o_custkey
                 AND l_orderkey = o_orderkey
                 AND o_orderdate < TIMESTAMP '1995-03-15'
                 AND l_shipdate > TIMESTAMP '1995-03-15'
               GROUP BY l_orderkey, o_orderdate, o_shippriority
               ORDER BY revenue DESC, o_orderdate
               LIMIT 10"""
        ).to_arrow_table()

    def polars():
        cust = ctx.pl_t["customer"].lazy().filter(pl.col("c_mktsegment") == "BUILDING")
        orders = ctx.pl_t["orders"].lazy().filter(pl.col("o_orderdate") < dt.datetime(1995, 3, 15))
        line = ctx.pl_t["lineitem"].lazy().filter(pl.col("l_shipdate") > dt.datetime(1995, 3, 15))
        return (
            cust.join(orders, left_on="c_custkey", right_on="o_custkey")
            .join(line, left_on="o_orderkey", right_on="l_orderkey")
            .group_by("o_orderkey", "o_orderdate", "o_shippriority")
            .agg((pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).sum().alias("revenue"))
            .sort(["revenue", "o_orderdate"], descending=[True, False])
            .limit(10)
            .collect()
            .to_arrow()
        )

    return {"batcher": batcher, "duckdb": duckdb, "polars": polars}


@tpch.case("tpch-q10")
def q10_returned_item_reporting(ctx: TpchContext):
    """Q10: returned item reporting (four-way join with grouped top-20)."""

    def batcher():
        cust = ctx.bt_t["customer"]
        orders = ctx.bt_t["orders"].filter(
            (col("o_orderdate") >= lit(dt.datetime(1993, 10, 1)))
            & (col("o_orderdate") < lit(dt.datetime(1994, 1, 1)))
        )
        line = ctx.bt_t["lineitem"].filter(col("l_returnflag") == lit("R"))
        nation = ctx.bt_t["nation"]
        joined = (
            cust.join(orders, left_on="c_custkey", right_on="o_custkey", how="inner")
            .join(line, left_on="o_orderkey", right_on="l_orderkey", how="inner")
            .join(nation, left_on="c_nationkey", right_on="n_nationkey", how="inner")
            .with_columns(rev=col("l_extendedprice") * (lit(1.0) - col("l_discount")))
        )
        return (
            joined.group_by("c_custkey", "c_name", "n_name")
            .agg(revenue=col("rev").sum())
            .sort("revenue", "c_custkey", descending=[True, False])
            .limit(20)
            .collect()
        )

    def duckdb():
        return ctx.con.sql(
            """SELECT c_custkey, c_name, n_name,
                      sum(l_extendedprice * (1 - l_discount)) AS revenue
               FROM customer, orders, lineitem, nation
               WHERE c_custkey = o_custkey
                 AND l_orderkey = o_orderkey
                 AND o_orderdate >= TIMESTAMP '1993-10-01'
                 AND o_orderdate < TIMESTAMP '1994-01-01'
                 AND l_returnflag = 'R'
                 AND c_nationkey = n_nationkey
               GROUP BY c_custkey, c_name, n_name
               ORDER BY revenue DESC, c_custkey
               LIMIT 20"""
        ).to_arrow_table()

    def polars():
        cust = ctx.pl_t["customer"].lazy()
        orders = (
            ctx.pl_t["orders"]
            .lazy()
            .filter(
                (pl.col("o_orderdate") >= dt.datetime(1993, 10, 1))
                & (pl.col("o_orderdate") < dt.datetime(1994, 1, 1))
            )
        )
        line = ctx.pl_t["lineitem"].lazy().filter(pl.col("l_returnflag") == "R")
        nation = ctx.pl_t["nation"].lazy()
        return (
            cust.join(orders, left_on="c_custkey", right_on="o_custkey")
            .join(line, left_on="o_orderkey", right_on="l_orderkey")
            .join(nation, left_on="c_nationkey", right_on="n_nationkey")
            .group_by("c_custkey", "c_name", "n_name")
            .agg((pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).sum().alias("revenue"))
            .sort(["revenue", "c_custkey"], descending=[True, False])
            .limit(20)
            .collect()
            .to_arrow()
        )

    return {"batcher": batcher, "duckdb": duckdb, "polars": polars}
