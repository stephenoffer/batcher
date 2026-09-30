"""TPC-H q12-q22 as Ray Data's release scripts state them."""

from __future__ import annotations

import batcher as bt
from batcher import col, lit

from .tables import d as _d
from .tables import f64 as _f64
from .tables import register as _q


@_q
def q12(h: dict[str, bt.Dataset], _sf: int) -> bt.Dataset:
    lineitem = (
        h["lineitem"]
        .select("l_orderkey", "l_shipmode", "l_commitdate", "l_shipdate", "l_receiptdate")
        .filter(
            col("l_shipmode").is_in(["MAIL", "SHIP"])
            & (col("l_commitdate") < col("l_receiptdate"))
            & (col("l_shipdate") < col("l_commitdate"))
            & (col("l_receiptdate") >= _d(1994, 1, 1))
            & (col("l_receiptdate") < _d(1995, 1, 1))
        )
        .select("l_orderkey", "l_shipmode")
    )
    high = col("o_orderpriority").is_in(["1-URGENT", "2-HIGH"])
    return (
        lineitem.join(
            h["orders"].select("o_orderkey", "o_orderpriority"),
            left_on="l_orderkey",
            right_on="o_orderkey",
        )
        .with_columns(high_line_count=high.cast("int64"), low_line_count=(~high).cast("int64"))
        .group_by("l_shipmode")
        .agg(
            high_line_count=col("high_line_count").sum(), low_line_count=col("low_line_count").sum()
        )
        .sort("l_shipmode")
    )


@_q
def q13(h: dict[str, bt.Dataset], _sf: int) -> bt.Dataset:
    orders = (
        h["orders"]
        .select("o_orderkey", "o_custkey", "o_comment")
        .filter(~col("o_comment").str.contains("special.*requests", literal=False))
        .select("o_orderkey", "o_custkey")
    )
    return (
        h["customer"]
        .select("c_custkey")
        .join(orders, left_on="c_custkey", right_on="o_custkey", how="left")
        .group_by("c_custkey")
        .agg(c_count=col("o_orderkey").count())
        .group_by("c_count")
        .agg(custdist=bt.count())
        .sort("custdist", "c_count", descending=[True, True])
    )


@_q
def q14(h: dict[str, bt.Dataset], _sf: int) -> bt.Dataset:
    lineitem = (
        h["lineitem"]
        .select("l_partkey", "l_extendedprice", "l_discount", "l_shipdate")
        .filter((col("l_shipdate") >= _d(1995, 9, 1)) & (col("l_shipdate") < _d(1995, 10, 1)))
    )
    return (
        lineitem.join(
            h["part"].select("p_partkey", "p_type"), left_on="l_partkey", right_on="p_partkey"
        )
        .with_columns(revenue=_f64("l_extendedprice") * (lit(1.0) - _f64("l_discount")))
        .with_columns(
            promo_revenue=col("revenue") * col("p_type").str.starts_with("PROMO").cast("float64")
        )
        .agg(sum_promo_revenue=col("promo_revenue").sum(), sum_revenue=col("revenue").sum())
    )


@_q
def q15(h: dict[str, bt.Dataset], _sf: int) -> bt.Dataset:
    revenue = (
        h["lineitem"]
        .select("l_suppkey", "l_extendedprice", "l_discount", "l_shipdate")
        .filter((col("l_shipdate") >= _d(1996, 1, 1)) & (col("l_shipdate") < _d(1996, 4, 1)))
        .with_columns(rev=_f64("l_extendedprice") * (lit(1.0) - _f64("l_discount")))
        .group_by("l_suppkey")
        .agg(total_revenue=col("rev").sum())
    )
    top = revenue.join(revenue.agg(max_rev=col("total_revenue").max()), how="cross").filter(
        col("total_revenue") == col("max_rev")
    )
    return (
        h["supplier"]
        .select("s_suppkey", "s_name", "s_address", "s_phone")
        .join(top, left_on="s_suppkey", right_on="l_suppkey")
        .select("s_suppkey", "s_name", "s_address", "s_phone", "total_revenue")
        .sort("s_suppkey")
    )


@_q
def q16(h: dict[str, bt.Dataset], _sf: int) -> bt.Dataset:
    part = (
        h["part"]
        .select("p_partkey", "p_brand", "p_type", "p_size")
        .filter(
            (col("p_brand") != lit("Brand#45"))
            & col("p_size").is_in([49, 14, 23, 45, 19, 3, 36, 9])
            & ~col("p_type").str.starts_with("MEDIUM POLISHED")
        )
    )
    complainers = (
        h["supplier"]
        .select("s_suppkey", "s_comment")
        .filter(col("s_comment").str.contains("Customer.*Complaints", literal=False))
        .select("s_suppkey")
    )
    joined = (
        h["partsupp"]
        .select("ps_partkey", "ps_suppkey")
        .join(complainers, left_on="ps_suppkey", right_on="s_suppkey", how="anti")
        .join(part, left_on="ps_partkey", right_on="p_partkey")
        .select("p_brand", "p_type", "p_size", "ps_suppkey")
    )
    return (
        joined.group_by("p_brand", "p_type", "p_size", "ps_suppkey")
        .agg(_dedupe=bt.count())
        .group_by("p_brand", "p_type", "p_size")
        .agg(supplier_cnt=bt.count())
        .sort("supplier_cnt", "p_brand", "p_type", "p_size", descending=[True, False, False, False])
    )


@_q
def q17(h: dict[str, bt.Dataset], _sf: int) -> bt.Dataset:
    part = (
        h["part"]
        .select("p_partkey", "p_brand", "p_container")
        .filter((col("p_brand") == lit("Brand#23")) & (col("p_container") == lit("MED BOX")))
        .select("p_partkey")
    )
    joined = part.join(
        h["lineitem"].select("l_partkey", "l_quantity", "l_extendedprice"),
        left_on="p_partkey",
        right_on="l_partkey",
    ).select("p_partkey", "l_quantity", "l_extendedprice")
    avg_qty = joined.group_by("p_partkey").agg(avg_quantity=col("l_quantity").mean())
    return (
        joined.join(avg_qty, on="p_partkey")
        .filter(_f64("l_quantity") < lit(0.2) * _f64("avg_quantity"))
        .agg(avg_yearly=_f64("l_extendedprice").sum())
    )


@_q
def q18(h: dict[str, bt.Dataset], _sf: int) -> bt.Dataset:
    lineitem = h["lineitem"].select("l_orderkey", "l_quantity")
    large = (
        lineitem.group_by("l_orderkey")
        .agg(total_quantity=col("l_quantity").sum())
        .filter(col("total_quantity") > lit(312))
    )
    orders_customer = (
        h["orders"]
        .select("o_orderkey", "o_custkey", "o_orderdate", "o_totalprice")
        .join(
            h["customer"].select("c_custkey", "c_name"), left_on="o_custkey", right_on="c_custkey"
        )
        .select("o_orderkey", "o_custkey", "o_orderdate", "o_totalprice", "c_name")
    )
    return (
        lineitem.join(large, on="l_orderkey")
        .select("l_orderkey", "l_quantity", "total_quantity")
        .join(orders_customer, left_on="l_orderkey", right_on="o_orderkey")
        .group_by("c_name", "o_custkey", "l_orderkey", "o_orderdate", "o_totalprice")
        .agg(sum_quantity=col("l_quantity").sum())
        .sort("o_totalprice", "o_orderdate", descending=[True, False])
    )


@_q
def q19(h: dict[str, bt.Dataset], _sf: int) -> bt.Dataset:
    clauses = [
        ("Brand#12", ["SM CASE", "SM BOX", "SM PACK", "SM PKG"], 1, 11, 5),
        ("Brand#23", ["MED BAG", "MED BOX", "MED PKG", "MED PACK"], 10, 20, 10),
        ("Brand#34", ["LG CASE", "LG BOX", "LG PACK", "LG PKG"], 20, 30, 15),
    ]
    lineitem = (
        h["lineitem"]
        .select(
            "l_partkey",
            "l_quantity",
            "l_extendedprice",
            "l_discount",
            "l_shipinstruct",
            "l_shipmode",
        )
        .filter(
            col("l_shipmode").is_in(["AIR", "AIR REG"])
            & (col("l_shipinstruct") == lit("DELIVER IN PERSON"))
        )
    )
    disjunction = None
    for brand, containers, lo, hi, size_hi in clauses:
        clause = (
            (col("p_brand") == lit(brand))
            & col("p_container").is_in(containers)
            & (col("l_quantity") >= lit(lo))
            & (col("l_quantity") <= lit(hi))
            & (col("p_size") >= lit(1))
            & (col("p_size") <= lit(size_hi))
        )
        disjunction = clause if disjunction is None else (disjunction | clause)
    return (
        lineitem.join(
            h["part"].select("p_partkey", "p_brand", "p_size", "p_container"),
            left_on="l_partkey",
            right_on="p_partkey",
        )
        .filter(disjunction)
        .with_columns(revenue=_f64("l_extendedprice") * (lit(1.0) - _f64("l_discount")))
        .agg(revenue=col("revenue").sum())
    )


@_q
def q20(h: dict[str, bt.Dataset], _sf: int) -> bt.Dataset:
    forest = (
        h["part"]
        .select("p_partkey", "p_name")
        .filter(col("p_name").str.starts_with("forest"))
        .select("p_partkey")
    )
    ps_forest = (
        h["partsupp"]
        .select("ps_partkey", "ps_suppkey", "ps_availqty")
        .join(forest, left_on="ps_partkey", right_on="p_partkey", how="semi")
        .with_columns(ps_availqty_f=_f64("ps_availqty"))
    )
    li_agg = (
        h["lineitem"]
        .select("l_partkey", "l_suppkey", "l_quantity", "l_shipdate")
        .filter((col("l_shipdate") >= _d(1994, 1, 1)) & (col("l_shipdate") < _d(1995, 1, 1)))
        .group_by("l_partkey", "l_suppkey")
        .agg(sum_qty=col("l_quantity").sum())
        .with_columns(sum_qty_f=_f64("sum_qty"))
        .select("l_partkey", "l_suppkey", "sum_qty_f")
    )
    qualified = (
        ps_forest.join(
            li_agg, left_on=["ps_partkey", "ps_suppkey"], right_on=["l_partkey", "l_suppkey"]
        )
        .filter(col("ps_availqty_f") > lit(0.5) * col("sum_qty_f"))
        .select("ps_suppkey")
    )
    canadian = (
        h["supplier"]
        .select("s_suppkey", "s_name", "s_address", "s_nationkey")
        .join(
            h["nation"].select("n_nationkey", "n_name").filter(col("n_name") == lit("CANADA")),
            left_on="s_nationkey",
            right_on="n_nationkey",
        )
        .select("s_suppkey", "s_name", "s_address")
    )
    return (
        canadian.join(qualified, left_on="s_suppkey", right_on="ps_suppkey", how="semi")
        .select("s_name", "s_address")
        .sort("s_name")
    )


@_q
def q21(h: dict[str, bt.Dataset], _sf: int) -> bt.Dataset:
    lineitem = h["lineitem"].select("l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate")
    multi = (
        lineitem.select("l_orderkey", "l_suppkey")
        .group_by("l_orderkey")
        .agg(num_suppliers=bt.count_distinct("l_suppkey"))
        .filter(col("num_suppliers") > lit(1))
    )
    late = lineitem.filter(col("l_receiptdate") > col("l_commitdate")).select(
        "l_orderkey", "l_suppkey"
    )
    single_late = (
        late.group_by("l_orderkey")
        .agg(num_late_suppliers=bt.count_distinct("l_suppkey"))
        .filter(col("num_late_suppliers") == lit(1))
    )
    saudi = (
        h["supplier"]
        .select("s_suppkey", "s_name", "s_nationkey")
        .join(
            h["nation"]
            .select("n_nationkey", "n_name")
            .filter(col("n_name") == lit("SAUDI ARABIA")),
            left_on="s_nationkey",
            right_on="n_nationkey",
        )
        .select("s_suppkey", "s_name")
    )
    failed = (
        h["orders"]
        .select("o_orderkey", "o_orderstatus")
        .filter(col("o_orderstatus") == lit("F"))
        .select("o_orderkey")
    )
    return (
        late.join(failed, left_on="l_orderkey", right_on="o_orderkey", how="semi")
        .join(saudi, left_on="l_suppkey", right_on="s_suppkey")
        .join(multi, on="l_orderkey")
        .join(single_late, on="l_orderkey")
        .group_by("s_name")
        .agg(numwait=bt.count())
        .sort("numwait", "s_name", descending=[True, False])
        .limit(100)
    )


@_q
def q22(h: dict[str, bt.Dataset], _sf: int) -> bt.Dataset:
    customer = (
        h["customer"]
        .select("c_custkey", "c_phone", "c_acctbal")
        .with_columns(cntrycode=col("c_phone").str.slice(0, 2), c_acctbal_f=_f64("c_acctbal"))
        .filter(col("cntrycode").is_in(["13", "31", "23", "29", "30", "18", "17"]))
    )
    avg_bal = customer.filter(col("c_acctbal_f") > lit(0.0)).agg(
        avg_acctbal=col("c_acctbal_f").mean()
    )
    return (
        customer.join(avg_bal, how="cross")
        .filter(col("c_acctbal_f") > col("avg_acctbal"))
        .join(
            h["orders"].select("o_custkey"), left_on="c_custkey", right_on="o_custkey", how="anti"
        )
        .group_by("cntrycode")
        .agg(numcust=bt.count(), totacctbal=col("c_acctbal_f").sum())
        .sort("cntrycode")
    )
