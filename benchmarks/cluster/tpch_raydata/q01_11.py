"""TPC-H q1-q11 as Ray Data's release scripts state them."""

from __future__ import annotations

import batcher as bt
from batcher import col, lit

from .tables import d as _d
from .tables import f64 as _f64
from .tables import register as _q


@_q
def q1(h: dict[str, bt.Dataset], _sf: int) -> bt.Dataset:
    li = (
        h["lineitem"]
        .filter(col("l_shipdate") <= _d(1998, 9, 2))
        .with_columns(
            l_quantity_f=_f64("l_quantity"),
            l_extendedprice_f=_f64("l_extendedprice"),
            l_discount_f=_f64("l_discount"),
            l_tax_f=_f64("l_tax"),
        )
        .with_columns(disc_price=col("l_extendedprice_f") * (lit(1.0) - col("l_discount_f")))
        .with_columns(charge=col("disc_price") * (lit(1.0) + col("l_tax_f")))
    )
    return (
        li.group_by("l_returnflag", "l_linestatus")
        .agg(
            sum_qty=col("l_quantity_f").sum(),
            sum_base_price=col("l_extendedprice_f").sum(),
            sum_disc_price=col("disc_price").sum(),
            sum_charge=col("charge").sum(),
            avg_qty=col("l_quantity_f").mean(),
            avg_price=col("l_extendedprice_f").mean(),
            avg_disc=col("l_discount_f").mean(),
            count_order=bt.count(),
        )
        .sort("l_returnflag", "l_linestatus")
    )


@_q
def q2(h: dict[str, bt.Dataset], _sf: int) -> bt.Dataset:
    region = h["region"].select("r_regionkey", "r_name")
    nation = h["nation"].select("n_nationkey", "n_name", "n_regionkey")
    supplier = h["supplier"].select(
        "s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"
    )
    part = h["part"].select("p_partkey", "p_mfgr", "p_type", "p_size")
    partsupp = h["partsupp"].select("ps_partkey", "ps_suppkey", "ps_supplycost")

    nation_region = (
        region.filter(col("r_name") == lit("EUROPE"))
        .join(nation, left_on="r_regionkey", right_on="n_regionkey")
        .select("n_nationkey", "n_name")
    )
    regional_suppliers = nation_region.join(supplier, left_on="n_nationkey", right_on="s_nationkey")
    min_cost = (
        partsupp.join(
            regional_suppliers.select("s_suppkey"), left_on="ps_suppkey", right_on="s_suppkey"
        )
        .with_columns(ps_supplycost_f=_f64("ps_supplycost"))
        .group_by("ps_partkey")
        .agg(min_supplycost=col("ps_supplycost_f").min())
    )
    part_partsupp = (
        part.filter((col("p_size") == lit(15)) & col("p_type").str.ends_with("BRASS"))
        .join(partsupp, left_on="p_partkey", right_on="ps_partkey")
        .with_columns(ps_supplycost_f=_f64("ps_supplycost"))
        .select("p_partkey", "p_mfgr", "ps_suppkey", "ps_supplycost_f")
    )
    part_regional = part_partsupp.join(
        regional_suppliers.select(
            "s_suppkey", "s_name", "s_address", "s_phone", "s_acctbal", "s_comment", "n_name"
        ),
        left_on="ps_suppkey",
        right_on="s_suppkey",
    )
    return (
        part_regional.join(min_cost, left_on="p_partkey", right_on="ps_partkey")
        .filter(col("ps_supplycost_f") == col("min_supplycost"))
        .with_columns(s_acctbal=_f64("s_acctbal"))
        .select(
            "s_acctbal",
            "s_name",
            "n_name",
            "p_partkey",
            "p_mfgr",
            "s_address",
            "s_phone",
            "s_comment",
        )
        .sort("s_acctbal", "n_name", "s_name", "p_partkey", descending=[True, False, False, False])
        .limit(100)
    )


@_q
def q3(h: dict[str, bt.Dataset], _sf: int) -> bt.Dataset:
    date_ = _d(1995, 3, 15)
    customer = h["customer"].filter(col("c_mktsegment") == lit("BUILDING")).select("c_custkey")
    orders = h["orders"].select("o_orderkey", "o_custkey", "o_orderdate", "o_shippriority")
    orders_customer = customer.join(
        orders.filter(col("o_orderdate") < date_), left_on="c_custkey", right_on="o_custkey"
    ).select("o_orderkey", "o_orderdate", "o_shippriority")
    lineitem = (
        h["lineitem"]
        .select("l_orderkey", "l_shipdate", "l_extendedprice", "l_discount")
        .filter(col("l_shipdate") > date_)
    )
    return (
        orders_customer.join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .with_columns(revenue=_f64("l_extendedprice") * (lit(1.0) - _f64("l_discount")))
        .group_by("o_orderkey", "o_orderdate", "o_shippriority")
        .agg(revenue=col("revenue").sum())
        .sort("revenue", "o_orderdate", descending=[True, False])
    )


@_q
def q4(h: dict[str, bt.Dataset], _sf: int) -> bt.Dataset:
    orders = (
        h["orders"]
        .filter((col("o_orderdate") >= _d(1993, 7, 1)) & (col("o_orderdate") < _d(1993, 10, 1)))
        .select("o_orderkey", "o_orderpriority")
    )
    late = h["lineitem"].filter(col("l_commitdate") < col("l_receiptdate")).select("l_orderkey")
    return (
        orders.join(late, left_on="o_orderkey", right_on="l_orderkey", how="semi")
        .group_by("o_orderpriority")
        .agg(order_count=bt.count())
        .sort("o_orderpriority")
    )


@_q
def q5(h: dict[str, bt.Dataset], _sf: int) -> bt.Dataset:
    nation_region = (
        h["region"]
        .filter(col("r_name") == lit("ASIA"))
        .select("r_regionkey")
        .join(
            h["nation"].select("n_nationkey", "n_name", "n_regionkey"),
            left_on="r_regionkey",
            right_on="n_regionkey",
        )
        .select("n_nationkey", "n_name")
    )
    customer_nation = nation_region.join(
        h["customer"].select("c_custkey", "c_nationkey"),
        left_on="n_nationkey",
        right_on="c_nationkey",
    ).select("c_custkey", c_nationkey=col("n_nationkey"), n_name=col("n_name"))
    orders_customer = (
        h["orders"]
        .select("o_orderkey", "o_custkey", "o_orderdate")
        .filter((col("o_orderdate") >= _d(1994, 1, 1)) & (col("o_orderdate") < _d(1995, 1, 1)))
        .join(customer_nation, left_on="o_custkey", right_on="c_custkey")
        .select("o_orderkey", "c_nationkey", "n_name")
    )
    lineitem_orders = (
        h["lineitem"]
        .select("l_orderkey", "l_suppkey", "l_extendedprice", "l_discount")
        .join(orders_customer, left_on="l_orderkey", right_on="o_orderkey")
        .select("l_suppkey", "l_extendedprice", "l_discount", "c_nationkey", "n_name")
    )
    return (
        lineitem_orders.join(
            h["supplier"].select("s_suppkey", "s_nationkey"),
            left_on="l_suppkey",
            right_on="s_suppkey",
        )
        .filter(col("c_nationkey") == col("s_nationkey"))
        .with_columns(revenue=_f64("l_extendedprice") * (lit(1.0) - _f64("l_discount")))
        .group_by("n_name")
        .agg(revenue=col("revenue").sum())
        .sort("revenue", descending=True)
    )


@_q
def q6(h: dict[str, bt.Dataset], _sf: int) -> bt.Dataset:
    return (
        h["lineitem"]
        .filter(
            (col("l_shipdate") >= _d(1994, 1, 1))
            & (col("l_shipdate") < _d(1995, 1, 1))
            & (col("l_discount") >= lit(0.05))
            & (col("l_discount") <= lit(0.07))
            & (col("l_quantity") < lit(24))
        )
        .with_columns(revenue=_f64("l_extendedprice") * _f64("l_discount"))
        .agg(revenue=col("revenue").sum())
    )


@_q
def q7(h: dict[str, bt.Dataset], _sf: int) -> bt.Dataset:
    nations = (
        h["nation"]
        .select("n_nationkey", "n_name")
        .filter((col("n_name") == lit("FRANCE")) | (col("n_name") == lit("GERMANY")))
    )
    customer_nation = nations.join(
        h["customer"].select("c_custkey", "c_nationkey"),
        left_on="n_nationkey",
        right_on="c_nationkey",
    ).select("c_custkey", n_name_cust=col("n_name"))
    orders_customer = (
        h["orders"]
        .select("o_orderkey", "o_custkey")
        .join(customer_nation, left_on="o_custkey", right_on="c_custkey")
        .select("o_orderkey", "n_name_cust")
    )
    lineitem_orders = (
        h["lineitem"]
        .select("l_orderkey", "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount")
        .filter((col("l_shipdate") >= _d(1995, 1, 1)) & (col("l_shipdate") < _d(1997, 1, 1)))
        .join(orders_customer, left_on="l_orderkey", right_on="o_orderkey")
        .select("l_suppkey", "l_shipdate", "l_extendedprice", "l_discount", "n_name_cust")
    )
    lineitem_supplier = lineitem_orders.join(
        h["supplier"].select("s_suppkey", "s_nationkey"), left_on="l_suppkey", right_on="s_suppkey"
    ).select("l_shipdate", "l_extendedprice", "l_discount", "n_name_cust", "s_nationkey")
    nation_supp = nations.select(supp_nationkey=col("n_nationkey"), n_name_supp=col("n_name"))
    return (
        lineitem_supplier.join(nation_supp, left_on="s_nationkey", right_on="supp_nationkey")
        .filter(col("n_name_supp") != col("n_name_cust"))
        .with_columns(
            revenue=_f64("l_extendedprice") * (lit(1.0) - _f64("l_discount")),
            l_year=col("l_shipdate").dt.year(),
        )
        .group_by("n_name_supp", "n_name_cust", "l_year")
        .agg(revenue=col("revenue").sum())
        .sort("n_name_supp", "n_name_cust", "l_year")
    )


@_q
def q8(h: dict[str, bt.Dataset], _sf: int) -> bt.Dataset:
    nation = h["nation"].select("n_nationkey", "n_name", "n_regionkey")
    nation_region = (
        h["region"]
        .select("r_regionkey", "r_name")
        .filter(col("r_name") == lit("AMERICA"))
        .join(nation, left_on="r_regionkey", right_on="n_regionkey")
    )
    customer_nation = nation_region.join(
        h["customer"].select("c_custkey", "c_nationkey"),
        left_on="n_nationkey",
        right_on="c_nationkey",
    ).select("c_custkey")
    orders_customer = (
        h["orders"]
        .select("o_orderkey", "o_custkey", "o_orderdate")
        .filter((col("o_orderdate") >= _d(1995, 1, 1)) & (col("o_orderdate") < _d(1997, 1, 1)))
        .join(customer_nation, left_on="o_custkey", right_on="c_custkey")
        .select("o_orderkey", "o_orderdate")
    )
    lineitem_orders = (
        h["lineitem"]
        .select("l_orderkey", "l_partkey", "l_suppkey", "l_extendedprice", "l_discount")
        .join(orders_customer, left_on="l_orderkey", right_on="o_orderkey")
        .select(
            "l_orderkey", "l_partkey", "l_suppkey", "l_extendedprice", "l_discount", "o_orderdate"
        )
    )
    part = (
        h["part"]
        .select("p_partkey", "p_type")
        .filter(col("p_type") == lit("ECONOMY ANODIZED STEEL"))
    )
    lineitem_part = lineitem_orders.join(part, left_on="l_partkey", right_on="p_partkey").select(
        "l_suppkey", "l_extendedprice", "l_discount", "o_orderdate"
    )
    lineitem_supplier = lineitem_part.join(
        h["supplier"].select("s_suppkey", "s_nationkey"), left_on="l_suppkey", right_on="s_suppkey"
    ).select("l_extendedprice", "l_discount", "o_orderdate", "s_nationkey")
    nation_supp = nation.select(supp_nationkey=col("n_nationkey"), n_name_supp=col("n_name"))
    return (
        lineitem_supplier.join(nation_supp, left_on="s_nationkey", right_on="supp_nationkey")
        .with_columns(
            volume=_f64("l_extendedprice") * (lit(1.0) - _f64("l_discount")),
            o_year=col("o_orderdate").dt.year(),
            is_nation=(col("n_name_supp") == lit("BRAZIL")).cast("float64"),
        )
        .with_columns(nation_volume=col("is_nation") * col("volume"))
        .group_by("o_year")
        .agg(total_volume=col("volume").sum(), nation_volume=col("nation_volume").sum())
        .with_columns(mkt_share=col("nation_volume") / col("total_volume"))
        .select("o_year", "mkt_share")
        .sort("o_year")
    )


@_q
def q9(h: dict[str, bt.Dataset], _sf: int) -> bt.Dataset:
    lineitem = h["lineitem"].select(
        "l_orderkey", "l_partkey", "l_suppkey", "l_quantity", "l_extendedprice", "l_discount"
    )
    lineitem_part = (
        lineitem.join(
            h["part"].select("p_partkey", "p_name"), left_on="l_partkey", right_on="p_partkey"
        )
        .filter(col("p_name").str.contains("green"))
        .select(
            "l_orderkey", "l_partkey", "l_suppkey", "l_quantity", "l_extendedprice", "l_discount"
        )
    )
    lineitem_partsupp = lineitem_part.join(
        h["partsupp"].select("ps_partkey", "ps_suppkey", "ps_supplycost"),
        left_on=["l_partkey", "l_suppkey"],
        right_on=["ps_partkey", "ps_suppkey"],
    ).select(
        "l_orderkey", "l_quantity", "l_extendedprice", "l_discount", "l_suppkey", "ps_supplycost"
    )
    lineitem_supplier = lineitem_partsupp.join(
        h["supplier"].select("s_suppkey", "s_nationkey"), left_on="l_suppkey", right_on="s_suppkey"
    ).select(
        "l_orderkey", "l_quantity", "l_extendedprice", "l_discount", "ps_supplycost", "s_nationkey"
    )
    lineitem_nation = lineitem_supplier.join(
        h["nation"].select("n_nationkey", "n_name"), left_on="s_nationkey", right_on="n_nationkey"
    ).select("l_orderkey", "l_quantity", "l_extendedprice", "l_discount", "ps_supplycost", "n_name")
    return (
        lineitem_nation.join(
            h["orders"].select("o_orderkey", "o_orderdate"),
            left_on="l_orderkey",
            right_on="o_orderkey",
        )
        .with_columns(
            profit=_f64("l_extendedprice") * (lit(1.0) - _f64("l_discount"))
            - _f64("ps_supplycost") * _f64("l_quantity"),
            o_year=col("o_orderdate").dt.year(),
        )
        .group_by("n_name", "o_year")
        .agg(profit=col("profit").sum())
        .sort("n_name", "o_year", descending=[False, True])
    )


@_q
def q10(h: dict[str, bt.Dataset], _sf: int) -> bt.Dataset:
    customer = h["customer"].select(
        "c_custkey", "c_name", "c_nationkey", "c_acctbal", "c_address", "c_phone", "c_comment"
    )
    orders = (
        h["orders"]
        .select("o_orderkey", "o_custkey", "o_orderdate")
        .filter((col("o_orderdate") >= _d(1993, 10, 1)) & (col("o_orderdate") < _d(1994, 1, 1)))
    )
    lineitem = (
        h["lineitem"]
        .select("l_orderkey", "l_extendedprice", "l_discount", "l_returnflag")
        .filter(col("l_returnflag") == lit("R"))
    )
    ocn = (
        orders.join(customer, left_on="o_custkey", right_on="c_custkey")
        .join(
            h["nation"].select("n_nationkey", "n_name"),
            left_on="c_nationkey",
            right_on="n_nationkey",
        )
        .select(
            "o_orderkey",
            "o_custkey",
            "c_name",
            "c_acctbal",
            "n_name",
            "c_address",
            "c_phone",
            "c_comment",
        )
    )
    return (
        ocn.join(lineitem, left_on="o_orderkey", right_on="l_orderkey")
        .with_columns(revenue=_f64("l_extendedprice") * (lit(1.0) - _f64("l_discount")))
        .group_by("o_custkey", "c_name", "c_acctbal", "n_name", "c_address", "c_phone", "c_comment")
        .agg(revenue=col("revenue").sum())
        .sort("revenue", descending=True)
    )


@_q
def q11(h: dict[str, bt.Dataset], sf: int) -> bt.Dataset:
    nation_supplier = (
        h["nation"]
        .select("n_nationkey", "n_name")
        .filter(col("n_name") == lit("GERMANY"))
        .join(
            h["supplier"].select("s_suppkey", "s_nationkey"),
            left_on="n_nationkey",
            right_on="s_nationkey",
        )
        .select("s_suppkey")
    )
    germany = (
        h["partsupp"]
        .select("ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost")
        .join(nation_supplier, left_on="ps_suppkey", right_on="s_suppkey")
        .with_columns(value=_f64("ps_supplycost") * _f64("ps_availqty"))
        .select("ps_partkey", "value")
    )
    threshold = germany.agg(threshold=col("value").sum() * lit(0.0001 / sf))
    return (
        germany.group_by("ps_partkey")
        .agg(value=col("value").sum())
        .join(threshold, how="cross")
        .filter(col("value") > col("threshold"))
        .select("ps_partkey", "value")
        .sort("value", descending=True)
    )
