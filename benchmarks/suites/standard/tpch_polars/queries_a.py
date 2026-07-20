"""TPC-H q1-q11 on the Polars lazy DataFrame API.

Every query is the same workload as the SQL text in ``suites/standard/tpch.py``,
re-expressed as a ``LazyFrame`` pipeline: implicit ``FROM a, b WHERE a.x = b.x``
joins become explicit ``join``s, ``EXISTS``/``IN`` become ``semi``/``anti`` joins,
and scalar subqueries become an aggregate joined or broadcast back in. Substitution
parameters are the TPC-H validation defaults, and every ``interval`` arithmetic is
pre-resolved to the literal date it denotes (``date '1993-07-01' + interval '3'
month`` -> ``date(1993, 10, 1)``).
"""

from __future__ import annotations

from datetime import date

import polars as pl

from .base import impl, revenue


@impl("tpch-q1")
def q1(t: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Pricing summary report: one filtered scan into a two-key aggregate."""
    disc_price = pl.col("l_extendedprice") * (1.0 - pl.col("l_discount"))
    return (
        t["lineitem"]
        .filter(pl.col("l_shipdate") <= date(1998, 9, 2))  # 1998-12-01 minus 90 days
        .group_by("l_returnflag", "l_linestatus")
        .agg(
            pl.col("l_quantity").sum().alias("sum_qty"),
            pl.col("l_extendedprice").sum().alias("sum_base_price"),
            disc_price.sum().alias("sum_disc_price"),
            (disc_price * (1.0 + pl.col("l_tax"))).sum().alias("sum_charge"),
            pl.col("l_quantity").mean().alias("avg_qty"),
            pl.col("l_extendedprice").mean().alias("avg_price"),
            pl.col("l_discount").mean().alias("avg_disc"),
            pl.len().alias("count_order"),
        )
        .sort("l_returnflag", "l_linestatus")
    )


@impl("tpch-q2")
def q2(t: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Minimum cost supplier: the correlated ``min(ps_supplycost)`` as a self-join.

    The subquery is correlated only on ``p_partkey`` and ranges over exactly the same
    European partsupp rows as the outer query, so it is a grouped minimum over the
    already-built join rather than a per-part re-scan.
    """
    europe = (
        t["region"]
        .filter(pl.col("r_name") == "EUROPE")
        .join(t["nation"], left_on="r_regionkey", right_on="n_regionkey")
        .join(t["supplier"], left_on="n_nationkey", right_on="s_nationkey")
        .join(t["partsupp"], left_on="s_suppkey", right_on="ps_suppkey")
    )
    parts = t["part"].filter((pl.col("p_size") == 15) & pl.col("p_type").str.ends_with("BRASS"))
    joined = europe.join(parts, left_on="ps_partkey", right_on="p_partkey")
    cheapest = joined.group_by("ps_partkey").agg(pl.col("ps_supplycost").min().alias("min_cost"))
    return (
        joined.join(cheapest, on="ps_partkey")
        .filter(pl.col("ps_supplycost") == pl.col("min_cost"))
        .select(
            "s_acctbal",
            "s_name",
            "n_name",
            pl.col("ps_partkey").alias("p_partkey"),
            "p_mfgr",
            "s_address",
            "s_phone",
            "s_comment",
        )
        .sort(
            ["s_acctbal", "n_name", "s_name", "p_partkey"],
            descending=[True, False, False, False],
        )
        .head(100)
    )


@impl("tpch-q3")
def q3(t: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Shipping priority: three-way join, grouped revenue, top 10."""
    cutoff = date(1995, 3, 15)
    return (
        t["customer"]
        .filter(pl.col("c_mktsegment") == "BUILDING")
        .join(t["orders"], left_on="c_custkey", right_on="o_custkey")
        .filter(pl.col("o_orderdate") < cutoff)
        .join(t["lineitem"], left_on="o_orderkey", right_on="l_orderkey")
        .filter(pl.col("l_shipdate") > cutoff)
        .group_by("o_orderkey", "o_orderdate", "o_shippriority")
        .agg(revenue())
        .select(
            pl.col("o_orderkey").alias("l_orderkey"),
            "revenue",
            "o_orderdate",
            "o_shippriority",
        )
        .sort(["revenue", "o_orderdate"], descending=[True, False])
        .head(10)
    )


@impl("tpch-q4")
def q4(t: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Order priority checking: the ``EXISTS`` subquery as a semi-join."""
    late_orders = (
        t["lineitem"]
        .filter(pl.col("l_commitdate") < pl.col("l_receiptdate"))
        .select("l_orderkey")
        .unique()
    )
    return (
        t["orders"]
        .filter(
            (pl.col("o_orderdate") >= date(1993, 7, 1))
            & (pl.col("o_orderdate") < date(1993, 10, 1))
        )
        .join(late_orders, left_on="o_orderkey", right_on="l_orderkey", how="semi")
        .group_by("o_orderpriority")
        .agg(pl.len().alias("order_count"))
        .sort("o_orderpriority")
    )


@impl("tpch-q5")
def q5(t: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Local supplier volume: six-way join where supplier and customer share a nation.

    ``c_nationkey = s_nationkey AND s_nationkey = n_nationkey`` is expressed by joining
    supplier on the *pair* (suppkey, nationkey), which is what forces the supplier to
    sit in the same nation the customer was reached through.
    """
    return (
        t["region"]
        .filter(pl.col("r_name") == "ASIA")
        .join(t["nation"], left_on="r_regionkey", right_on="n_regionkey")
        .join(t["customer"], left_on="n_nationkey", right_on="c_nationkey")
        .join(t["orders"], left_on="c_custkey", right_on="o_custkey")
        .filter(
            (pl.col("o_orderdate") >= date(1994, 1, 1)) & (pl.col("o_orderdate") < date(1995, 1, 1))
        )
        .join(t["lineitem"], left_on="o_orderkey", right_on="l_orderkey")
        .join(
            t["supplier"],
            left_on=["l_suppkey", "n_nationkey"],
            right_on=["s_suppkey", "s_nationkey"],
        )
        .group_by("n_name")
        .agg(revenue())
        .sort("revenue", descending=True)
    )


@impl("tpch-q6")
def q6(t: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Forecasting revenue change: one filtered scan into a global sum.

    The discount band is written as its resolved bounds ``[0.05, 0.07]`` rather than
    ``BETWEEN 0.06 - 0.01 AND 0.06 + 0.01``. That is the same band the TPC-H spec
    defines, and it sidesteps a real Polars-SQL defect: its parser folds the two
    decimal literals and converts the result to ``f64`` one ulp *below* ``0.07``, so
    every ``l_discount = 0.07`` row was silently dropped and the query returned
    75,207,768 against DuckDB's 123,141,078 (exactly the 11/18 of revenue the
    surviving 0.05/0.06 rows carry).
    """
    return (
        t["lineitem"]
        .filter(
            (pl.col("l_shipdate") >= date(1994, 1, 1))
            & (pl.col("l_shipdate") < date(1995, 1, 1))
            & (pl.col("l_discount") >= 0.05)
            & (pl.col("l_discount") <= 0.07)
            & (pl.col("l_quantity") < 24)
        )
        .select((pl.col("l_extendedprice") * pl.col("l_discount")).sum().alias("revenue"))
    )


@impl("tpch-q7")
def q7(t: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Volume shipping: nation joined twice, pre-renamed to stand in for ``n1``/``n2``."""
    supp_nation = t["nation"].select(
        pl.col("n_nationkey").alias("supp_nationkey"),
        pl.col("n_name").alias("supp_nation"),
    )
    cust_nation = t["nation"].select(
        pl.col("n_nationkey").alias("cust_nationkey"),
        pl.col("n_name").alias("cust_nation"),
    )
    return (
        t["lineitem"]
        .filter(
            (pl.col("l_shipdate") >= date(1995, 1, 1))
            & (pl.col("l_shipdate") <= date(1996, 12, 31))
        )
        .join(t["supplier"], left_on="l_suppkey", right_on="s_suppkey")
        .join(supp_nation, left_on="s_nationkey", right_on="supp_nationkey")
        .join(t["orders"], left_on="l_orderkey", right_on="o_orderkey")
        .join(t["customer"], left_on="o_custkey", right_on="c_custkey")
        .join(cust_nation, left_on="c_nationkey", right_on="cust_nationkey")
        .filter(
            ((pl.col("supp_nation") == "FRANCE") & (pl.col("cust_nation") == "GERMANY"))
            | ((pl.col("supp_nation") == "GERMANY") & (pl.col("cust_nation") == "FRANCE"))
        )
        .with_columns(pl.col("l_shipdate").dt.year().alias("l_year"))
        .group_by("supp_nation", "cust_nation", "l_year")
        .agg(revenue())
        .select("supp_nation", "cust_nation", pl.col("l_year").cast(pl.Int64), "revenue")
        .sort("supp_nation", "cust_nation", "l_year")
    )


@impl("tpch-q8")
def q8(t: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """National market share: the customer's region gates the row, the supplier's names it."""
    cust_nation = t["nation"].select(
        pl.col("n_nationkey").alias("cust_nationkey"), pl.col("n_regionkey")
    )
    supp_nation = t["nation"].select(
        pl.col("n_nationkey").alias("supp_nationkey"), pl.col("n_name").alias("nation")
    )
    volume = pl.col("l_extendedprice") * (1.0 - pl.col("l_discount"))
    return (
        t["part"]
        .filter(pl.col("p_type") == "ECONOMY ANODIZED STEEL")
        .join(t["lineitem"], left_on="p_partkey", right_on="l_partkey")
        .join(t["supplier"], left_on="l_suppkey", right_on="s_suppkey")
        .join(t["orders"], left_on="l_orderkey", right_on="o_orderkey")
        .filter(
            (pl.col("o_orderdate") >= date(1995, 1, 1))
            & (pl.col("o_orderdate") <= date(1996, 12, 31))
        )
        .join(t["customer"], left_on="o_custkey", right_on="c_custkey")
        .join(cust_nation, left_on="c_nationkey", right_on="cust_nationkey")
        .join(
            t["region"].filter(pl.col("r_name") == "AMERICA"),
            left_on="n_regionkey",
            right_on="r_regionkey",
        )
        .join(supp_nation, left_on="s_nationkey", right_on="supp_nationkey")
        .with_columns(
            pl.col("o_orderdate").dt.year().alias("o_year"),
            volume.alias("volume"),
        )
        .group_by("o_year")
        .agg(
            (
                pl.when(pl.col("nation") == "BRAZIL").then(pl.col("volume")).otherwise(0.0).sum()
                / pl.col("volume").sum()
            ).alias("mkt_share")
        )
        .select(pl.col("o_year").cast(pl.Int64), "mkt_share")
        .sort("o_year")
    )


@impl("tpch-q9")
def q9(t: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Product type profit measure: six-way join over the ``%green%`` parts."""
    amount = pl.col("l_extendedprice") * (1.0 - pl.col("l_discount")) - pl.col(
        "ps_supplycost"
    ) * pl.col("l_quantity")
    return (
        t["part"]
        .filter(pl.col("p_name").str.contains("green", literal=True))
        .join(t["lineitem"], left_on="p_partkey", right_on="l_partkey")
        .join(t["supplier"], left_on="l_suppkey", right_on="s_suppkey")
        .join(
            t["partsupp"],
            left_on=["p_partkey", "l_suppkey"],
            right_on=["ps_partkey", "ps_suppkey"],
        )
        .join(t["orders"], left_on="l_orderkey", right_on="o_orderkey")
        .join(t["nation"], left_on="s_nationkey", right_on="n_nationkey")
        .with_columns(
            pl.col("n_name").alias("nation"),
            pl.col("o_orderdate").dt.year().alias("o_year"),
            amount.alias("amount"),
        )
        .group_by("nation", "o_year")
        .agg(pl.col("amount").sum().alias("sum_profit"))
        .select("nation", pl.col("o_year").cast(pl.Int64), "sum_profit")
        .sort(["nation", "o_year"], descending=[False, True])
    )


@impl("tpch-q10")
def q10(t: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Returned item reporting: revenue by customer over one quarter's returns."""
    return (
        t["customer"]
        .join(t["orders"], left_on="c_custkey", right_on="o_custkey")
        .filter(
            (pl.col("o_orderdate") >= date(1993, 10, 1))
            & (pl.col("o_orderdate") < date(1994, 1, 1))
        )
        .join(t["lineitem"], left_on="o_orderkey", right_on="l_orderkey")
        .filter(pl.col("l_returnflag") == "R")
        .join(t["nation"], left_on="c_nationkey", right_on="n_nationkey")
        .group_by("c_custkey", "c_name", "c_acctbal", "c_phone", "n_name", "c_address", "c_comment")
        .agg(revenue())
        .select(
            "c_custkey",
            "c_name",
            "revenue",
            "c_acctbal",
            "n_name",
            "c_address",
            "c_phone",
            "c_comment",
        )
        .sort("revenue", descending=True)
        .head(20)
    )


@impl("tpch-q11")
def q11(t: dict[str, pl.LazyFrame]) -> pl.LazyFrame:
    """Important stock identification: a HAVING clause against a global scalar.

    The subquery totals the *same* German stock the outer query groups, so the
    threshold is a window-free ``sum`` over the grouped result rather than a re-scan.
    """
    german_stock = (
        t["partsupp"]
        .join(t["supplier"], left_on="ps_suppkey", right_on="s_suppkey")
        .join(
            t["nation"].filter(pl.col("n_name") == "GERMANY"),
            left_on="s_nationkey",
            right_on="n_nationkey",
        )
        .with_columns((pl.col("ps_supplycost") * pl.col("ps_availqty")).alias("stock_value"))
    )
    return (
        german_stock.group_by("ps_partkey")
        .agg(pl.col("stock_value").sum().alias("value"))
        .filter(pl.col("value") > pl.col("value").sum() * 0.0001)
        .select("ps_partkey", "value")
        .sort("value", descending=True)
    )
