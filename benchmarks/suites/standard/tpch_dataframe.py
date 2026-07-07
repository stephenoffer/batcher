"""Batcher native DataFrame (``bt.Dataset``) pipelines for the TPC-H queries.

Batcher's SQL and DataFrame surfaces lower to the *same* plan IR, so these are a
parse-free, apples-to-apples counterpart to Ray Data's ``ray.data.Dataset``
pipelines (``tpch_ray.py``): every engine in the multi-node lineup is then measured
on the same DataFrame-style whole-query workload, not a SQL string one engine can't
parse. Each impl takes a ``{table -> bt.Dataset}`` handle map and returns the result
as a ``pyarrow.Table`` whose column names match the SQL reference's aliases exactly
(the harness compares engines by column name + row multiset).

Only the queries that map cleanly to the relational DataFrame API live here; the
subquery/correlated ones (q2/q11/q17/q20/q21/q22) stay SQL-only and show ``n/a`` for
the DataFrame engines, never a wrong answer (the harness gates on correctness vs the
SQL reference before timing).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

import pyarrow as pa

import batcher as bt
from batcher import col, lit, when

# A batcher DataFrame pipeline: table-name -> bt.Dataset handle map, returning a table.
BatcherImpl = Callable[[dict[str, Any]], pa.Table]
_IMPLS: dict[str, BatcherImpl] = {}


def batcher_impl(name: str) -> BatcherImpl | None:
    """The registered batcher DataFrame pipeline for `name`, or None if SQL-only."""
    return _IMPLS.get(name)


def _impl(name: str) -> Callable[[BatcherImpl], BatcherImpl]:
    def register(fn: BatcherImpl) -> BatcherImpl:
        _IMPLS[name] = fn
        return fn

    return register


def _disc_revenue() -> bt.Expr:
    """``l_extendedprice * (1 - l_discount)`` — the TPC-H revenue measure."""
    return col("l_extendedprice") * (lit(1.0) - col("l_discount"))


# --------------------------------------------------------------------------- #
# q1 — filter → two-key aggregate (the canonical no-join aggregate shape).
# --------------------------------------------------------------------------- #
@_impl("tpch-q1")
def _q1(h: dict[str, Any]) -> pa.Table:
    charge = _disc_revenue() * (lit(1.0) + col("l_tax"))
    return (
        h["lineitem"]
        .filter(col("l_shipdate") <= lit(date(1998, 9, 2)))
        .group_by("l_returnflag", "l_linestatus")
        .agg(
            sum_qty=col("l_quantity").sum(),
            sum_base_price=col("l_extendedprice").sum(),
            sum_disc_price=_disc_revenue().sum(),
            sum_charge=charge.sum(),
            avg_qty=col("l_quantity").mean(),
            avg_price=col("l_extendedprice").mean(),
            avg_disc=col("l_discount").mean(),
            count_order=bt.count(),
        )
        .sort("l_returnflag", "l_linestatus")
        .to_arrow()
    )


# --------------------------------------------------------------------------- #
# q3 — customer ⋈ orders ⋈ lineitem, group by order, top-10 by revenue.
# --------------------------------------------------------------------------- #
@_impl("tpch-q3")
def _q3(h: dict[str, Any]) -> pa.Table:
    cust = h["customer"].filter(col("c_mktsegment") == "BUILDING")
    orders = h["orders"].filter(col("o_orderdate") < lit(date(1995, 3, 15)))
    line = h["lineitem"].filter(col("l_shipdate") > lit(date(1995, 3, 15)))
    return (
        cust.join(orders, left_on="c_custkey", right_on="o_custkey")
        .join(line, left_on="o_orderkey", right_on="l_orderkey")
        .group_by("o_orderkey", "o_orderdate", "o_shippriority")
        .agg(revenue=_disc_revenue().sum())
        .select(
            col("o_orderkey").alias("l_orderkey"),
            "revenue",
            "o_orderdate",
            "o_shippriority",
        )
        .sort("revenue", "o_orderdate", descending=[True, False])
        .limit(10)
        .to_arrow()
    )


# --------------------------------------------------------------------------- #
# q5 — 6-table join (region→nation→supplier→lineitem, orders→customer joined
# back on both orderkey AND nationkey), revenue by nation.
# --------------------------------------------------------------------------- #
@_impl("tpch-q5")
def _q5(h: dict[str, Any]) -> pa.Table:
    region = h["region"].filter(col("r_name") == "ASIA")
    nation = h["nation"].join(region, left_on="n_regionkey", right_on="r_regionkey")
    supplier = h["supplier"].join(nation, left_on="s_nationkey", right_on="n_nationkey")
    line_sup = h["lineitem"].join(supplier, left_on="l_suppkey", right_on="s_suppkey")
    orders = h["orders"].filter(
        (col("o_orderdate") >= lit(date(1994, 1, 1))) & (col("o_orderdate") < lit(date(1995, 1, 1)))
    )
    ord_cust = orders.join(h["customer"], left_on="o_custkey", right_on="c_custkey")
    return (
        line_sup.join(
            ord_cust,
            left_on=["l_orderkey", "s_nationkey"],
            right_on=["o_orderkey", "c_nationkey"],
        )
        .group_by("n_name")
        .agg(revenue=_disc_revenue().sum())
        .sort("revenue", descending=True)
        .to_arrow()
    )


# --------------------------------------------------------------------------- #
# q6 — single-table filtered global sum (a pure streaming reduction).
# --------------------------------------------------------------------------- #
@_impl("tpch-q6")
def _q6(h: dict[str, Any]) -> pa.Table:
    return (
        h["lineitem"]
        .filter(
            (col("l_shipdate") >= lit(date(1994, 1, 1)))
            & (col("l_shipdate") < lit(date(1995, 1, 1)))
            & (col("l_discount") >= lit(0.05))
            & (col("l_discount") <= lit(0.07))
            & (col("l_quantity") < lit(24.0))
        )
        .agg(revenue=(col("l_extendedprice") * col("l_discount")).sum())
        .to_arrow()
    )


# --------------------------------------------------------------------------- #
# q10 — customer ⋈ orders ⋈ lineitem ⋈ nation, returned-line revenue, top-20.
# --------------------------------------------------------------------------- #
@_impl("tpch-q10")
def _q10(h: dict[str, Any]) -> pa.Table:
    orders = h["orders"].filter(
        (col("o_orderdate") >= lit(date(1993, 10, 1)))
        & (col("o_orderdate") < lit(date(1994, 1, 1)))
    )
    line = h["lineitem"].filter(col("l_returnflag") == "R")
    return (
        h["customer"]
        .join(orders, left_on="c_custkey", right_on="o_custkey")
        .join(line, left_on="o_orderkey", right_on="l_orderkey")
        .join(h["nation"], left_on="c_nationkey", right_on="n_nationkey")
        .group_by(
            "c_custkey",
            "c_name",
            "c_acctbal",
            "c_phone",
            "n_name",
            "c_address",
            "c_comment",
        )
        .agg(revenue=_disc_revenue().sum())
        .sort("revenue", descending=True)
        .limit(20)
        .to_arrow()
    )


# --------------------------------------------------------------------------- #
# q12 — orders ⋈ lineitem, ship-mode line counts split by order priority (CASE).
# --------------------------------------------------------------------------- #
@_impl("tpch-q12")
def _q12(h: dict[str, Any]) -> pa.Table:
    high = (col("o_orderpriority") == "1-URGENT") | (col("o_orderpriority") == "2-HIGH")
    line = h["lineitem"].filter(
        col("l_shipmode").is_in(["MAIL", "SHIP"])
        & (col("l_commitdate") < col("l_receiptdate"))
        & (col("l_shipdate") < col("l_commitdate"))
        & (col("l_receiptdate") >= lit(date(1994, 1, 1)))
        & (col("l_receiptdate") < lit(date(1995, 1, 1)))
    )
    return (
        h["orders"]
        .join(line, left_on="o_orderkey", right_on="l_orderkey")
        .group_by("l_shipmode")
        .agg(
            high_line_count=when(high).then(lit(1)).otherwise(lit(0)).sum(),
            low_line_count=when(high).then(lit(0)).otherwise(lit(1)).sum(),
        )
        .sort("l_shipmode")
        .to_arrow()
    )


# --------------------------------------------------------------------------- #
# q14 — lineitem ⋈ part, promo-revenue share (CASE over a join).
# --------------------------------------------------------------------------- #
@_impl("tpch-q14")
def _q14(h: dict[str, Any]) -> pa.Table:
    line = h["lineitem"].filter(
        (col("l_shipdate") >= lit(date(1995, 9, 1))) & (col("l_shipdate") < lit(date(1995, 10, 1)))
    )
    promo = when(col("p_type").str.starts_with("PROMO")).then(_disc_revenue()).otherwise(lit(0.0))
    # agg() values must each be a bare aggregate, so sum the promo and total revenue
    # first, then form the ratio in a follow-up projection (same as the SQL does).
    return (
        line.join(h["part"], left_on="l_partkey", right_on="p_partkey")
        .agg(_promo=promo.sum(), _total=_disc_revenue().sum())
        .select(promo_revenue=lit(100.0) * col("_promo") / col("_total"))
        .to_arrow()
    )


# --------------------------------------------------------------------------- #
# q19 — lineitem ⋈ part, three OR-ed brand/container/quantity bands, sum revenue.
# --------------------------------------------------------------------------- #
@_impl("tpch-q19")
def _q19(h: dict[str, Any]) -> pa.Table:
    shared = col("l_shipinstruct") == "DELIVER IN PERSON"
    air = col("l_shipmode").is_in(["AIR", "AIR REG"])

    def band(brand: str, containers: list[str], qlo: float, shi: int) -> bt.Expr:
        return (
            (col("p_brand") == brand)
            & col("p_container").is_in(containers)
            & (col("l_quantity") >= lit(qlo))
            & (col("l_quantity") <= lit(qlo + 10))
            & (col("p_size") >= lit(1))
            & (col("p_size") <= lit(shi))
        )

    band1 = band("Brand#12", ["SM CASE", "SM BOX", "SM PACK", "SM PKG"], 1.0, 5)
    band2 = band("Brand#23", ["MED BAG", "MED BOX", "MED PKG", "MED PACK"], 10.0, 10)
    band3 = band("Brand#34", ["LG CASE", "LG BOX", "LG PACK", "LG PKG"], 20.0, 15)
    return (
        h["lineitem"]
        .join(h["part"], left_on="l_partkey", right_on="p_partkey")
        .filter(air & shared & (band1 | band2 | band3))
        .agg(revenue=_disc_revenue().sum())
        .to_arrow()
    )
