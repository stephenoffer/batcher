"""Ray Data pipelines for TPC-H q9-q16.

Each function takes the ``{table name -> ray.data.Dataset}`` handle map and returns
the result as a ``pyarrow.Table`` whose column names match the SQL reference's
aliases exactly (the harness compares engines by column name + row multiset).
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from .base import impl, join, mb, revenue, take, year_of


def _d(value: date) -> pa.Scalar:
    return pa.scalar(value, pa.date32())


# --------------------------------------------------------------------------- #
# q9 - six-way join including a composite (partkey, suppkey) key into partsupp.
# --------------------------------------------------------------------------- #
@impl("tpch-q9")
def q9(h: dict[str, Any]) -> pa.Table:
    from ray.data.aggregate import Sum

    parts = mb(
        h["part"],
        lambda b: pa.table(
            {"p_partkey": b.filter(pc.match_substring(b["p_name"], pattern="green"))["p_partkey"]}
        ),
    )
    line = mb(
        h["lineitem"],
        lambda b: pa.table(
            {
                "l_orderkey": b["l_orderkey"],
                "l_partkey": b["l_partkey"],
                "l_suppkey": b["l_suppkey"],
                "l_quantity": b["l_quantity"],
                "rev": revenue(b),
            }
        ),
    )
    lp = join(line, parts, "l_partkey", "p_partkey")
    ps = mb(
        h["partsupp"],
        lambda b: pa.table(
            {
                "ps_partkey": b["ps_partkey"],
                "ps_suppkey": b["ps_suppkey"],
                "ps_supplycost": b["ps_supplycost"],
            }
        ),
    )
    lps = join(lp, ps, ("l_partkey", "l_suppkey"), ("ps_partkey", "ps_suppkey"))
    ords = mb(
        h["orders"],
        lambda b: pa.table({"o_orderkey": b["o_orderkey"], "o_year": year_of(b["o_orderdate"])}),
    )
    lpso = join(lps, ords, "l_orderkey", "o_orderkey")
    supp_nation = mb(
        join(
            mb(
                h["supplier"],
                lambda b: pa.table({"s_suppkey": b["s_suppkey"], "s_nationkey": b["s_nationkey"]}),
            ),
            mb(
                h["nation"],
                lambda b: pa.table({"n_nationkey": b["n_nationkey"], "n_name": b["n_name"]}),
            ),
            "s_nationkey",
            "n_nationkey",
        ),
        lambda b: pa.table({"s_suppkey": b["s_suppkey"], "nation": b["n_name"]}),
    )
    full = join(lpso, supp_nation, "l_suppkey", "s_suppkey")

    def amount(b: pa.Table) -> pa.Table:
        cost = pc.multiply(b["ps_supplycost"], b["l_quantity"])
        return pa.table(
            {
                "nation": b["nation"],
                "o_year": b["o_year"],
                "amount": pc.subtract(b["rev"], cost),
            }
        )

    g = mb(full, amount).groupby(["nation", "o_year"]).aggregate(Sum("amount"))
    rows = take(g)
    rows.sort(key=lambda r: (r["nation"], -r["o_year"]))
    return pa.table(
        {
            "nation": [r["nation"] for r in rows],
            "o_year": [r["o_year"] for r in rows],
            "sum_profit": [r["sum(amount)"] for r in rows],
        }
    )


# --------------------------------------------------------------------------- #
# q10 - returned-item revenue by customer, top 20.
# --------------------------------------------------------------------------- #
@impl("tpch-q10")
def q10(h: dict[str, Any]) -> pa.Table:
    from ray.data.aggregate import Sum

    lo, hi = _d(date(1993, 10, 1)), _d(date(1994, 1, 1))

    def ords(b: pa.Table) -> pa.Table:
        b = b.filter(pc.and_(pc.greater_equal(b["o_orderdate"], lo), pc.less(b["o_orderdate"], hi)))
        return pa.table({"o_orderkey": b["o_orderkey"], "o_custkey": b["o_custkey"]})

    def line(b: pa.Table) -> pa.Table:
        b = b.filter(pc.equal(b["l_returnflag"], "R"))
        return pa.table({"l_orderkey": b["l_orderkey"], "rev": revenue(b)})

    lo_j = join(mb(h["lineitem"], line), mb(h["orders"], ords), "l_orderkey", "o_orderkey")
    cust = join(
        h["customer"],
        mb(
            h["nation"],
            lambda b: pa.table({"n_nationkey": b["n_nationkey"], "n_name": b["n_name"]}),
        ),
        "c_nationkey",
        "n_nationkey",
    )
    full = join(lo_j, cust, "o_custkey", "c_custkey")
    g = full.groupby(
        ["o_custkey", "c_name", "c_acctbal", "c_phone", "n_name", "c_address", "c_comment"]
    ).aggregate(Sum("rev"))

    rows = take(g)
    rows.sort(key=lambda r: -r["sum(rev)"])
    rows = rows[:20]
    return pa.table(
        {
            "c_custkey": [r["o_custkey"] for r in rows],
            "c_name": [r["c_name"] for r in rows],
            "revenue": [r["sum(rev)"] for r in rows],
            "c_acctbal": [r["c_acctbal"] for r in rows],
            "n_name": [r["n_name"] for r in rows],
            "c_address": [r["c_address"] for r in rows],
            "c_phone": [r["c_phone"] for r in rows],
            "c_comment": [r["c_comment"] for r in rows],
        }
    )


# --------------------------------------------------------------------------- #
# q11 - grouped stock value with a HAVING against a global fraction of the total.
# --------------------------------------------------------------------------- #
@impl("tpch-q11")
def q11(h: dict[str, Any]) -> pa.Table:
    from ray.data.aggregate import Sum

    germany = mb(
        h["nation"],
        lambda b: pa.table(
            {"n_nationkey": b.filter(pc.equal(b["n_name"], "GERMANY"))["n_nationkey"]}
        ),
    )
    supp = join(
        mb(
            h["supplier"],
            lambda b: pa.table({"s_suppkey": b["s_suppkey"], "s_nationkey": b["s_nationkey"]}),
        ),
        germany,
        "s_nationkey",
        "n_nationkey",
    )
    ps = mb(
        h["partsupp"],
        lambda b: pa.table(
            {
                "ps_partkey": b["ps_partkey"],
                "ps_suppkey": b["ps_suppkey"],
                "value": pc.multiply(b["ps_supplycost"], b["ps_availqty"]),
            }
        ),
    )
    joined = join(ps, supp, "ps_suppkey", "s_suppkey").materialize()
    threshold = joined.sum("value") * 0.0001
    rows = take(joined.groupby("ps_partkey").aggregate(Sum("value")))
    rows = [r for r in rows if r["sum(value)"] > threshold]
    rows.sort(key=lambda r: -r["sum(value)"])
    return pa.table(
        {
            "ps_partkey": [r["ps_partkey"] for r in rows],
            "value": [r["sum(value)"] for r in rows],
        }
    )


# --------------------------------------------------------------------------- #
# q12 - join(orders, lineitem) -> filter -> conditional two-bucket aggregate.
# --------------------------------------------------------------------------- #
@impl("tpch-q12")
def q12(h: dict[str, Any]) -> pa.Table:
    from ray.data.aggregate import Sum

    lo, hi = _d(date(1994, 1, 1)), _d(date(1995, 1, 1))
    ship_modes = pa.array(["MAIL", "SHIP"])

    def prep_line(b: pa.Table) -> pa.Table:
        mask = pc.and_(
            pc.and_(
                pc.is_in(b["l_shipmode"], value_set=ship_modes),
                pc.less(b["l_commitdate"], b["l_receiptdate"]),
            ),
            pc.and_(
                pc.less(b["l_shipdate"], b["l_commitdate"]),
                pc.and_(
                    pc.greater_equal(b["l_receiptdate"], lo),
                    pc.less(b["l_receiptdate"], hi),
                ),
            ),
        )
        b = b.filter(mask)
        return pa.table({"l_orderkey": b["l_orderkey"], "l_shipmode": b["l_shipmode"]})

    orders = mb(
        h["orders"],
        lambda b: pa.table(
            {"o_orderkey": b["o_orderkey"], "o_orderpriority": b["o_orderpriority"]}
        ),
    )
    joined = join(mb(h["lineitem"], prep_line), orders, "l_orderkey", "o_orderkey")

    def buckets(b: pa.Table) -> pa.Table:
        urgent = pc.is_in(b["o_orderpriority"], value_set=pa.array(["1-URGENT", "2-HIGH"]))
        return pa.table(
            {
                "l_shipmode": b["l_shipmode"],
                "high_line_count": pc.if_else(urgent, 1, 0),
                "low_line_count": pc.if_else(urgent, 0, 1),
            }
        )

    g = (
        mb(joined, buckets)
        .groupby("l_shipmode")
        .aggregate(Sum("high_line_count"), Sum("low_line_count"))
    )
    rows = sorted(take(g), key=lambda r: r["l_shipmode"])
    return pa.table(
        {
            "l_shipmode": [r["l_shipmode"] for r in rows],
            "high_line_count": [r["sum(high_line_count)"] for r in rows],
            "low_line_count": [r["sum(low_line_count)"] for r in rows],
        }
    )


# --------------------------------------------------------------------------- #
# q13 - LEFT OUTER JOIN, then a distribution over the per-customer order count.
# `count(o_orderkey)` ignores NULLs, but Ray's `Count(col)` counts rows, so the
# non-null test is done explicitly before summing.
# --------------------------------------------------------------------------- #
@impl("tpch-q13")
def q13(h: dict[str, Any]) -> pa.Table:
    from ray.data.aggregate import Count, Sum

    def ords(b: pa.Table) -> pa.Table:
        special = pc.match_substring_regex(b["o_comment"], pattern="special.*requests")
        b = b.filter(pc.invert(special))
        return pa.table({"o_orderkey": b["o_orderkey"], "o_custkey": b["o_custkey"]})

    cust = mb(h["customer"], lambda b: pa.table({"c_custkey": b["c_custkey"]}))
    outer = join(cust, mb(h["orders"], ords), "c_custkey", "o_custkey", how="left_outer")
    flagged = mb(
        outer,
        lambda b: pa.table(
            {
                "c_custkey": b["c_custkey"],
                "has_order": pc.cast(pc.is_valid(b["o_orderkey"]), pa.int64()),
            }
        ),
    )
    per_cust = flagged.groupby("c_custkey").aggregate(Sum("has_order"))
    counted = mb(per_cust, lambda b: pa.table({"c_count": b["sum(has_order)"]}))
    rows = take(counted.groupby("c_count").aggregate(Count()))
    rows.sort(key=lambda r: (-r["count()"], -r["c_count"]))
    return pa.table(
        {
            "c_count": [r["c_count"] for r in rows],
            "custdist": [r["count()"] for r in rows],
        }
    )


# --------------------------------------------------------------------------- #
# q14 - join(lineitem, part) -> filter -> ratio of promo revenue to total.
# --------------------------------------------------------------------------- #
@impl("tpch-q14")
def q14(h: dict[str, Any]) -> pa.Table:
    lo, hi = _d(date(1995, 9, 1)), _d(date(1995, 10, 1))

    def prep_line(b: pa.Table) -> pa.Table:
        b = b.filter(pc.and_(pc.greater_equal(b["l_shipdate"], lo), pc.less(b["l_shipdate"], hi)))
        return pa.table({"l_partkey": b["l_partkey"], "rev": revenue(b)})

    part = mb(h["part"], lambda b: pa.table({"p_partkey": b["p_partkey"], "p_type": b["p_type"]}))
    joined = join(mb(h["lineitem"], prep_line), part, "l_partkey", "p_partkey")

    def contrib(b: pa.Table) -> pa.Table:
        promo = pc.starts_with(b["p_type"], pattern="PROMO")
        return pa.table({"promo_rev": pc.if_else(promo, b["rev"], 0.0), "total_rev": b["rev"]})

    c = mb(joined, contrib).materialize()
    return pa.table({"promo_revenue": [100.0 * c.sum("promo_rev") / c.sum("total_rev")]})


# --------------------------------------------------------------------------- #
# q15 - the top-revenue supplier(s) of a quarter, via a grouped sum re-joined
# against its own max.
# --------------------------------------------------------------------------- #
@impl("tpch-q15")
def q15(h: dict[str, Any]) -> pa.Table:
    from ray.data.aggregate import Sum

    lo, hi = _d(date(1996, 1, 1)), _d(date(1996, 4, 1))

    def line(b: pa.Table) -> pa.Table:
        b = b.filter(pc.and_(pc.greater_equal(b["l_shipdate"], lo), pc.less(b["l_shipdate"], hi)))
        return pa.table({"l_suppkey": b["l_suppkey"], "rev": revenue(b)})

    per_supp = mb(h["lineitem"], line).groupby("l_suppkey").aggregate(Sum("rev")).materialize()
    top = per_supp.max("sum(rev)")
    best = mb(per_supp, lambda b: b.filter(pc.equal(b["sum(rev)"], top)))
    supp = mb(
        h["supplier"],
        lambda b: pa.table(
            {
                "s_suppkey": b["s_suppkey"],
                "s_name": b["s_name"],
                "s_address": b["s_address"],
                "s_phone": b["s_phone"],
            }
        ),
    )
    joined = join(best, supp, "l_suppkey", "s_suppkey")
    rows = take(joined)
    rows.sort(key=lambda r: r["l_suppkey"])
    return pa.table(
        {
            "s_suppkey": [r["l_suppkey"] for r in rows],
            "s_name": [r["s_name"] for r in rows],
            "s_address": [r["s_address"] for r in rows],
            "s_phone": [r["s_phone"] for r in rows],
            "total_revenue": [r["sum(rev)"] for r in rows],
        }
    )


# --------------------------------------------------------------------------- #
# q16 - NOT IN over complaint suppliers, which is a left-anti join, then a
# count(DISTINCT) per part group.
# --------------------------------------------------------------------------- #
@impl("tpch-q16")
def q16(h: dict[str, Any]) -> pa.Table:
    from ray.data.aggregate import CountDistinct

    sizes = pa.array([49, 14, 23, 45, 19, 3, 36, 9])

    def complaints(b: pa.Table) -> pa.Table:
        bad = pc.match_substring_regex(b["s_comment"], pattern="Customer.*Complaints")
        return pa.table({"s_suppkey": b.filter(bad)["s_suppkey"]})

    ps = mb(
        h["partsupp"],
        lambda b: pa.table({"ps_partkey": b["ps_partkey"], "ps_suppkey": b["ps_suppkey"]}),
    )
    clean = join(ps, mb(h["supplier"], complaints), "ps_suppkey", "s_suppkey", how="left_anti")

    def parts(b: pa.Table) -> pa.Table:
        keep = pc.and_(
            pc.and_(
                pc.not_equal(b["p_brand"], "Brand#45"),
                pc.invert(pc.starts_with(b["p_type"], pattern="MEDIUM POLISHED")),
            ),
            pc.is_in(b["p_size"], value_set=sizes),
        )
        b = b.filter(keep)
        return pa.table(
            {
                "p_partkey": b["p_partkey"],
                "p_brand": b["p_brand"],
                "p_type": b["p_type"],
                "p_size": b["p_size"],
            }
        )

    joined = join(clean, mb(h["part"], parts), "ps_partkey", "p_partkey")
    g = joined.groupby(["p_brand", "p_type", "p_size"]).aggregate(CountDistinct("ps_suppkey"))
    rows = take(g)
    rows.sort(
        key=lambda r: (
            -r["count_distinct(ps_suppkey)"],
            r["p_brand"],
            r["p_type"],
            r["p_size"],
        )
    )
    return pa.table(
        {
            "p_brand": [r["p_brand"] for r in rows],
            "p_type": [r["p_type"] for r in rows],
            "p_size": [r["p_size"] for r in rows],
            "supplier_cnt": [r["count_distinct(ps_suppkey)"] for r in rows],
        }
    )
