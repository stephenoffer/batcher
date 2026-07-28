"""TPC-H q12-q22 on the Polars lazy DataFrame API.

The second half of the suite; see ``queries_a`` for the conventions these follow.
This half is where the SQL frontend gave up hardest — ``NOT IN``, ``NOT EXISTS``,
correlated per-group thresholds and the double-negation of q21 — so each one notes
how its subquery was flattened.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from .base import impl, revenue

# The q22 country codes, used both to pick the customers and to scope their average.
_CNTRYCODES = ["13", "31", "23", "29", "30", "18", "17"]


@impl("tpch-q12")
def q12(t: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Shipping modes and order priority: two conditional counts per ship mode."""
    urgent = pl.col("o_orderpriority").is_in(["1-URGENT", "2-HIGH"])
    return (
        t["lineitem"]
        .filter(
            pl.col("l_shipmode").is_in(["MAIL", "SHIP"])
            & (pl.col("l_commitdate") < pl.col("l_receiptdate"))
            & (pl.col("l_shipdate") < pl.col("l_commitdate"))
            & (pl.col("l_receiptdate") >= date(1994, 1, 1))
            & (pl.col("l_receiptdate") < date(1995, 1, 1))
        )
        .join(t["orders"], left_on="l_orderkey", right_on="o_orderkey")
        .group_by("l_shipmode")
        .agg(
            urgent.sum().alias("high_line_count"),
            (~urgent).sum().alias("low_line_count"),
        )
        .sort("l_shipmode")
    )


@impl("tpch-q13")
def q13(t: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Customer distribution: a left join whose ON-clause predicate filters the right side.

    ``ON c_custkey = o_custkey AND o_comment NOT LIKE ...`` is a left join against the
    *pre-filtered* orders — filtering after the join would drop customers instead of
    giving them a zero count. ``count`` skips nulls, so unmatched customers land at 0.
    """
    orders = t["orders"].filter(~pl.col("o_comment").str.contains("special.*requests"))
    return (
        t["customer"]
        .join(orders, left_on="c_custkey", right_on="o_custkey", how="left")
        .group_by("c_custkey")
        .agg(pl.col("o_orderkey").count().alias("c_count"))
        .group_by("c_count")
        .agg(pl.len().alias("custdist"))
        .select(pl.col("c_count").cast(pl.Int64), pl.col("custdist").cast(pl.Int64))
        .sort(["custdist", "c_count"], descending=[True, True])
    )


@impl("tpch-q14")
def q14(t: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Promotion effect: promotional revenue as a percentage of the month's total."""
    rev = pl.col("l_extendedprice") * (1.0 - pl.col("l_discount"))
    return (
        t["lineitem"]
        .filter(
            (pl.col("l_shipdate") >= date(1995, 9, 1)) & (pl.col("l_shipdate") < date(1995, 10, 1))
        )
        .join(t["part"], left_on="l_partkey", right_on="p_partkey")
        .select(
            (
                100.0
                * pl.when(pl.col("p_type").str.starts_with("PROMO")).then(rev).otherwise(0.0).sum()
                / rev.sum()
            ).alias("promo_revenue")
        )
    )


@impl("tpch-q15")
def q15(t: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Top supplier: the CTE is built once, then filtered to its own maximum.

    ``total_revenue = (SELECT max(total_revenue) FROM revenue)`` is a filter against an
    aggregate of the same frame, which Polars expresses directly — no self-join, and
    ties are all kept exactly as the SQL keeps them.
    """
    quarterly = (
        t["lineitem"]
        .filter(
            (pl.col("l_shipdate") >= date(1996, 1, 1)) & (pl.col("l_shipdate") < date(1996, 4, 1))
        )
        .group_by("l_suppkey")
        .agg(revenue("total_revenue"))
    )
    return (
        t["supplier"]
        .join(quarterly, left_on="s_suppkey", right_on="l_suppkey")
        .filter(pl.col("total_revenue") == pl.col("total_revenue").max())
        .select("s_suppkey", "s_name", "s_address", "s_phone", "total_revenue")
        .sort("s_suppkey")
    )


@impl("tpch-q16")
def q16(t: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Parts/supplier relationship: the ``NOT IN`` subquery as an anti-join.

    ``s_suppkey`` is non-nullable, so the anti-join is exactly equivalent to ``NOT IN``
    (they diverge only when the subquery can yield a null).
    """
    complaining = (
        t["supplier"]
        .filter(pl.col("s_comment").str.contains("Customer.*Complaints"))
        .select("s_suppkey")
    )
    return (
        t["part"]
        .filter(
            (pl.col("p_brand") != "Brand#45")
            & ~pl.col("p_type").str.starts_with("MEDIUM POLISHED")
            & pl.col("p_size").is_in([49, 14, 23, 45, 19, 3, 36, 9])
        )
        .join(t["partsupp"], left_on="p_partkey", right_on="ps_partkey")
        .join(complaining, left_on="ps_suppkey", right_on="s_suppkey", how="anti")
        .group_by("p_brand", "p_type", "p_size")
        .agg(pl.col("ps_suppkey").n_unique().alias("supplier_cnt"))
        .select("p_brand", "p_type", "p_size", pl.col("supplier_cnt").cast(pl.Int64))
        .sort(
            ["supplier_cnt", "p_brand", "p_type", "p_size"],
            descending=[True, False, False, False],
        )
    )


@impl("tpch-q17")
def q17(t: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Small-quantity-order revenue: the correlated average becomes a window.

    The subquery averages ``l_quantity`` per ``l_partkey``; because the outer query has
    already restricted to those same parts by an equi-join on partkey, the correlated
    scan is exactly a partitioned mean over the joined frame.
    """
    joined = (
        t["part"]
        .filter((pl.col("p_brand") == "Brand#23") & (pl.col("p_container") == "MED BOX"))
        .join(t["lineitem"], left_on="p_partkey", right_on="l_partkey")
    )
    return joined.filter(
        pl.col("l_quantity") < 0.2 * pl.col("l_quantity").mean().over("p_partkey")
    ).select((pl.col("l_extendedprice").sum() / 7.0).alias("avg_yearly"))


@impl("tpch-q18")
def q18(t: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Large volume customer: the ``IN (GROUP BY ... HAVING ...)`` subquery as a semi-join."""
    big_orders = (
        t["lineitem"]
        .group_by("l_orderkey")
        .agg(pl.col("l_quantity").sum().alias("order_qty"))
        .filter(pl.col("order_qty") > 300)
        .select("l_orderkey")
    )
    return (
        t["orders"]
        .join(big_orders, left_on="o_orderkey", right_on="l_orderkey", how="semi")
        .join(t["customer"], left_on="o_custkey", right_on="c_custkey")
        .join(t["lineitem"], left_on="o_orderkey", right_on="l_orderkey")
        .group_by("c_name", "o_custkey", "o_orderkey", "o_orderdate", "o_totalprice")
        # The SQL leaves this aggregate unaliased; DuckDB names it after the expression
        # and the harness compares on column names, so it is reproduced verbatim.
        .agg(pl.col("l_quantity").sum().alias("sum_qty"))
        .select(
            "c_name",
            pl.col("o_custkey").alias("c_custkey"),
            "o_orderkey",
            "o_orderdate",
            "o_totalprice",
            "sum_qty",
        )
        .sort(["o_totalprice", "o_orderdate"], descending=[True, False])
        .head(100)
    )


@impl("tpch-q19")
def q19(t: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Discounted revenue: three disjunctive brand/container/quantity bands.

    The three OR'd conjunctions share the same part-key equality and the same shipmode
    and shipinstruct predicates, so those are factored out of the disjunction.
    """
    shipped_by_air = pl.col("l_shipmode").is_in(["AIR", "AIR REG"]) & (
        pl.col("l_shipinstruct") == "DELIVER IN PERSON"
    )

    def band(brand: str, containers: list[str], qty_lo: int, size_hi: int) -> pl.Expr:
        return (
            (pl.col("p_brand") == brand)
            & pl.col("p_container").is_in(containers)
            & (pl.col("l_quantity") >= qty_lo)
            & (pl.col("l_quantity") <= qty_lo + 10)
            & (pl.col("p_size") >= 1)
            & (pl.col("p_size") <= size_hi)
        )

    bands = (
        band("Brand#12", ["SM CASE", "SM BOX", "SM PACK", "SM PKG"], 1, 5)
        | band("Brand#23", ["MED BAG", "MED BOX", "MED PKG", "MED PACK"], 10, 10)
        | band("Brand#34", ["LG CASE", "LG BOX", "LG PACK", "LG PKG"], 20, 15)
    )
    return (
        t["lineitem"]
        .join(t["part"], left_on="l_partkey", right_on="p_partkey")
        .filter(shipped_by_air & bands)
        .select(revenue())
    )


@impl("tpch-q20")
def q20(t: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Potential part promotion: two nested ``IN``s plus a correlated half-quantity threshold.

    The innermost correlated sum is grouped on exactly the ``(partkey, suppkey)`` pair
    it correlates on, so it precomputes into a frame the partsupp rows join against —
    an inner join, which also enforces the SQL's "no matching lineitems -> no row".
    """
    shipped_1994 = (
        t["lineitem"]
        .filter(
            (pl.col("l_shipdate") >= date(1994, 1, 1)) & (pl.col("l_shipdate") < date(1995, 1, 1))
        )
        .group_by("l_partkey", "l_suppkey")
        .agg((0.5 * pl.col("l_quantity").sum()).alias("half_qty"))
    )
    forest_parts = t["part"].filter(pl.col("p_name").str.starts_with("forest")).select("p_partkey")
    excess_suppliers = (
        t["partsupp"]
        .join(forest_parts, left_on="ps_partkey", right_on="p_partkey", how="semi")
        .join(
            shipped_1994,
            left_on=["ps_partkey", "ps_suppkey"],
            right_on=["l_partkey", "l_suppkey"],
        )
        .filter(pl.col("ps_availqty") > pl.col("half_qty"))
        .select("ps_suppkey")
        .unique()
    )
    return (
        t["supplier"]
        .join(
            t["nation"].filter(pl.col("n_name") == "CANADA"),
            left_on="s_nationkey",
            right_on="n_nationkey",
        )
        .join(excess_suppliers, left_on="s_suppkey", right_on="ps_suppkey", how="semi")
        .select("s_name", "s_address")
        .sort("s_name")
    )


@impl("tpch-q21")
def q21(t: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Suppliers who kept orders waiting: EXISTS and NOT EXISTS as per-order counts.

    ``EXISTS(another supplier on the order)`` and ``NOT EXISTS(another *late* supplier
    on the order)`` are both statements about how many distinct suppliers an order has.
    Counting distinct suppliers per order once — all of them, and the late ones — turns
    the pair of correlated subqueries into ``n_suppliers > 1 AND n_late_suppliers == 1``
    (the one late supplier necessarily being the row's own).
    """
    lines = t["lineitem"].select("l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate")
    late = pl.col("l_receiptdate") > pl.col("l_commitdate")
    per_order = lines.group_by("l_orderkey").agg(
        pl.col("l_suppkey").n_unique().alias("n_suppliers"),
        pl.col("l_suppkey").filter(late).n_unique().alias("n_late_suppliers"),
    )
    return (
        lines.filter(late)
        .join(per_order, on="l_orderkey")
        .filter((pl.col("n_suppliers") > 1) & (pl.col("n_late_suppliers") == 1))
        .join(
            t["orders"].filter(pl.col("o_orderstatus") == "F"),
            left_on="l_orderkey",
            right_on="o_orderkey",
        )
        .join(t["supplier"], left_on="l_suppkey", right_on="s_suppkey")
        .join(
            t["nation"].filter(pl.col("n_name") == "SAUDI ARABIA"),
            left_on="s_nationkey",
            right_on="n_nationkey",
        )
        .group_by("s_name")
        .agg(pl.len().alias("numwait"))
        .select("s_name", pl.col("numwait").cast(pl.Int64))
        .sort(["numwait", "s_name"], descending=[True, False])
        .head(100)
    )


@impl("tpch-q22")
def q22(t: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Global sales opportunity: a scalar average broadcast in, plus a ``NOT EXISTS``.

    The uncorrelated ``avg(c_acctbal)`` is a one-row frame cross-joined onto the
    candidates (Polars' way of broadcasting a scalar subquery), and ``NOT EXISTS
    (orders)`` is an anti-join on ``c_custkey``.
    """
    customers = t["customer"].with_columns(pl.col("c_phone").str.slice(0, 2).alias("cntrycode"))
    candidates = customers.filter(pl.col("cntrycode").is_in(_CNTRYCODES))
    avg_balance = candidates.filter(pl.col("c_acctbal") > 0.0).select(
        pl.col("c_acctbal").mean().alias("avg_acctbal")
    )
    return (
        candidates.join(avg_balance, how="cross")
        .filter(pl.col("c_acctbal") > pl.col("avg_acctbal"))
        .join(t["orders"], left_on="c_custkey", right_on="o_custkey", how="anti")
        .group_by("cntrycode")
        .agg(
            pl.len().alias("numcust"),
            pl.col("c_acctbal").sum().alias("totacctbal"),
        )
        .select("cntrycode", pl.col("numcust").cast(pl.Int64), "totacctbal")
        .sort("cntrycode")
    )
