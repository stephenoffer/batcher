"""Ray Data pipelines for TPC-H q1-q8.

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
# q1 - scan -> filter -> two-key aggregate (sum/avg/count). The canonical
# aggregate workload; no joins, the shape Ray Data's groupby targets.
# --------------------------------------------------------------------------- #
@impl("tpch-q1")
def q1(h: dict[str, Any]) -> pa.Table:
    from ray.data.aggregate import Count, Mean, Sum

    cutoff = _d(date(1998, 9, 2))  # 1998-12-01 minus 90 days

    def prep(b: pa.Table) -> pa.Table:
        b = b.filter(pc.less_equal(b["l_shipdate"], cutoff))
        disc = revenue(b)
        return pa.table(
            {
                "l_returnflag": b["l_returnflag"],
                "l_linestatus": b["l_linestatus"],
                "l_quantity": b["l_quantity"],
                "l_extendedprice": b["l_extendedprice"],
                "disc": disc,
                "charge": pc.multiply(disc, pc.add(1.0, b["l_tax"])),
                "l_discount": b["l_discount"],
            }
        )

    g = (
        mb(h["lineitem"], prep)
        .groupby(["l_returnflag", "l_linestatus"])
        .aggregate(
            Sum("l_quantity"),
            Sum("l_extendedprice"),
            Sum("disc"),
            Sum("charge"),
            Mean("l_quantity"),
            Mean("l_extendedprice"),
            Mean("l_discount"),
            Count(),
        )
    )
    rows = take(g)
    return pa.table(
        {
            "l_returnflag": [r["l_returnflag"] for r in rows],
            "l_linestatus": [r["l_linestatus"] for r in rows],
            "sum_qty": [r["sum(l_quantity)"] for r in rows],
            "sum_base_price": [r["sum(l_extendedprice)"] for r in rows],
            "sum_disc_price": [r["sum(disc)"] for r in rows],
            "sum_charge": [r["sum(charge)"] for r in rows],
            "avg_qty": [r["mean(l_quantity)"] for r in rows],
            "avg_price": [r["mean(l_extendedprice)"] for r in rows],
            "avg_disc": [r["mean(l_discount)"] for r in rows],
            "count_order": [r["count()"] for r in rows],
        }
    )


def _region_nations(h: dict[str, Any], region_name: str) -> Any:
    """nation |x| region, restricted to one region name. Shared by q2/q5/q8."""
    reg = mb(
        h["region"],
        lambda b: pa.table(
            {"r_regionkey": b.filter(pc.equal(b["r_name"], region_name))["r_regionkey"]}
        ),
    )
    nat = mb(
        h["nation"],
        lambda b: pa.table(
            {
                "n_regionkey": b["n_regionkey"],
                "n_nationkey": b["n_nationkey"],
                "n_name": b["n_name"],
            }
        ),
    )
    return join(nat, reg, "n_regionkey", "r_regionkey")


# --------------------------------------------------------------------------- #
# q2 - five-way join with a correlated min(ps_supplycost) per part, expressed as
# a grouped min re-joined onto the candidate rows.
# --------------------------------------------------------------------------- #
@impl("tpch-q2")
def q2(h: dict[str, Any]) -> pa.Table:
    from ray.data.aggregate import Min

    nations = _region_nations(h, "EUROPE")
    supp = join(h["supplier"], nations, "s_nationkey", "n_nationkey")
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
    # Consumed twice below (the grouped min, and the candidate join). A Ray Dataset is
    # lazy and re-executes per consumer, so without this the whole five-way join runs
    # three times over -- which is what made this query take 20 minutes.
    ps_eu = join(ps, supp, "ps_suppkey", "s_suppkey").materialize()

    cheapest = mb(
        ps_eu.groupby("ps_partkey").aggregate(Min("ps_supplycost")),
        lambda b: pa.table({"ps_partkey": b["ps_partkey"], "min_cost": b["min(ps_supplycost)"]}),
    )

    def wanted(b: pa.Table) -> pa.Table:
        keep = pc.and_(pc.equal(b["p_size"], 15), pc.ends_with(b["p_type"], pattern="BRASS"))
        b = b.filter(keep)
        return pa.table({"p_partkey": b["p_partkey"], "p_mfgr": b["p_mfgr"]})

    cand = join(ps_eu, mb(h["part"], wanted), "ps_partkey", "p_partkey")
    matched = join(cand, cheapest, "ps_partkey")
    best = mb(matched, lambda b: b.filter(pc.equal(b["ps_supplycost"], b["min_cost"])))

    rows = take(best)
    rows.sort(key=lambda r: (-r["s_acctbal"], r["n_name"], r["s_name"], r["ps_partkey"]))
    rows = rows[:100]
    return pa.table(
        {
            "s_acctbal": [r["s_acctbal"] for r in rows],
            "s_name": [r["s_name"] for r in rows],
            "n_name": [r["n_name"] for r in rows],
            "p_partkey": [r["ps_partkey"] for r in rows],
            "p_mfgr": [r["p_mfgr"] for r in rows],
            "s_address": [r["s_address"] for r in rows],
            "s_phone": [r["s_phone"] for r in rows],
            "s_comment": [r["s_comment"] for r in rows],
        }
    )


# --------------------------------------------------------------------------- #
# q3 - customer |x| orders |x| lineitem, grouped revenue, top 10.
# --------------------------------------------------------------------------- #
@impl("tpch-q3")
def q3(h: dict[str, Any]) -> pa.Table:
    from ray.data.aggregate import Sum

    cutoff = _d(date(1995, 3, 15))

    def cust(b: pa.Table) -> pa.Table:
        b = b.filter(pc.equal(b["c_mktsegment"], "BUILDING"))
        return pa.table({"c_custkey": b["c_custkey"]})

    def ords(b: pa.Table) -> pa.Table:
        b = b.filter(pc.less(b["o_orderdate"], cutoff))
        return pa.table(
            {
                "o_orderkey": b["o_orderkey"],
                "o_custkey": b["o_custkey"],
                "o_orderdate": b["o_orderdate"],
                "o_shippriority": b["o_shippriority"],
            }
        )

    def line(b: pa.Table) -> pa.Table:
        b = b.filter(pc.greater(b["l_shipdate"], cutoff))
        return pa.table({"l_orderkey": b["l_orderkey"], "rev": revenue(b)})

    co = join(mb(h["orders"], ords), mb(h["customer"], cust), "o_custkey", "c_custkey")
    joined = join(mb(h["lineitem"], line), co, "l_orderkey", "o_orderkey")
    g = joined.groupby(["l_orderkey", "o_orderdate", "o_shippriority"]).aggregate(Sum("rev"))

    rows = take(g)
    rows.sort(key=lambda r: (-r["sum(rev)"], r["o_orderdate"]))
    rows = rows[:10]
    return pa.table(
        {
            "l_orderkey": [r["l_orderkey"] for r in rows],
            "revenue": [r["sum(rev)"] for r in rows],
            "o_orderdate": [r["o_orderdate"] for r in rows],
            "o_shippriority": [r["o_shippriority"] for r in rows],
        }
    )


# --------------------------------------------------------------------------- #
# q4 - EXISTS over lineitem, which is a left-semi join (Ray 2.56+).
# --------------------------------------------------------------------------- #
@impl("tpch-q4")
def q4(h: dict[str, Any]) -> pa.Table:
    from ray.data.aggregate import Count

    lo, hi = _d(date(1993, 7, 1)), _d(date(1993, 10, 1))

    def ords(b: pa.Table) -> pa.Table:
        b = b.filter(pc.and_(pc.greater_equal(b["o_orderdate"], lo), pc.less(b["o_orderdate"], hi)))
        return pa.table({"o_orderkey": b["o_orderkey"], "o_orderpriority": b["o_orderpriority"]})

    def late(b: pa.Table) -> pa.Table:
        b = b.filter(pc.less(b["l_commitdate"], b["l_receiptdate"]))
        return pa.table({"l_orderkey": b["l_orderkey"]})

    semi = join(
        mb(h["orders"], ords), mb(h["lineitem"], late), "o_orderkey", "l_orderkey", how="left_semi"
    )
    rows = take(semi.groupby("o_orderpriority").aggregate(Count()))
    rows.sort(key=lambda r: r["o_orderpriority"])
    return pa.table(
        {
            "o_orderpriority": [r["o_orderpriority"] for r in rows],
            "order_count": [r["count()"] for r in rows],
        }
    )


# --------------------------------------------------------------------------- #
# q5 - six-way join; customer and supplier must sit in the same nation.
# --------------------------------------------------------------------------- #
@impl("tpch-q5")
def q5(h: dict[str, Any]) -> pa.Table:
    from ray.data.aggregate import Sum

    lo, hi = _d(date(1994, 1, 1)), _d(date(1995, 1, 1))
    nations = _region_nations(h, "ASIA")
    supp = join(
        mb(
            h["supplier"],
            lambda b: pa.table({"s_suppkey": b["s_suppkey"], "s_nationkey": b["s_nationkey"]}),
        ),
        nations,
        "s_nationkey",
        "n_nationkey",
    )

    def ords(b: pa.Table) -> pa.Table:
        b = b.filter(pc.and_(pc.greater_equal(b["o_orderdate"], lo), pc.less(b["o_orderdate"], hi)))
        return pa.table({"o_orderkey": b["o_orderkey"], "o_custkey": b["o_custkey"]})

    cust = mb(
        h["customer"],
        lambda b: pa.table({"c_custkey": b["c_custkey"], "c_nationkey": b["c_nationkey"]}),
    )
    co = join(mb(h["orders"], ords), cust, "o_custkey", "c_custkey")
    line = mb(
        h["lineitem"],
        lambda b: pa.table(
            {"l_orderkey": b["l_orderkey"], "l_suppkey": b["l_suppkey"], "rev": revenue(b)}
        ),
    )
    lo_j = join(line, co, "l_orderkey", "o_orderkey")
    full = join(lo_j, supp, "l_suppkey", "s_suppkey")
    same = mb(full, lambda b: b.filter(pc.equal(b["c_nationkey"], b["s_nationkey"])))

    rows = take(same.groupby("n_name").aggregate(Sum("rev")))
    rows.sort(key=lambda r: -r["sum(rev)"])
    return pa.table(
        {"n_name": [r["n_name"] for r in rows], "revenue": [r["sum(rev)"] for r in rows]}
    )


# --------------------------------------------------------------------------- #
# q6 - scan -> filter -> single global sum. A pure streaming reduction.
# --------------------------------------------------------------------------- #
@impl("tpch-q6")
def q6(h: dict[str, Any]) -> pa.Table:
    lo, hi = _d(date(1994, 1, 1)), _d(date(1995, 1, 1))

    def rev(b: pa.Table) -> pa.Table:
        mask = pc.and_(
            pc.and_(pc.greater_equal(b["l_shipdate"], lo), pc.less(b["l_shipdate"], hi)),
            pc.and_(
                pc.and_(
                    pc.greater_equal(b["l_discount"], 0.05),
                    pc.less_equal(b["l_discount"], 0.07),
                ),
                pc.less(b["l_quantity"], 24),
            ),
        )
        b = b.filter(mask)
        return pa.table({"r": pc.multiply(b["l_extendedprice"], b["l_discount"])})

    return pa.table({"revenue": [mb(h["lineitem"], rev).sum("r")]})


# --------------------------------------------------------------------------- #
# q7 - nation joined twice (supplier side and customer side) over a shipping window.
# --------------------------------------------------------------------------- #
@impl("tpch-q7")
def q7(h: dict[str, Any]) -> pa.Table:
    from ray.data.aggregate import Sum

    lo, hi = _d(date(1995, 1, 1)), _d(date(1996, 12, 31))
    pair = pa.array(["FRANCE", "GERMANY"])

    # Joined on both the supplier and the customer side, so materialize once.
    two = mb(
        h["nation"],
        lambda b: pa.table({"n_nationkey": b["n_nationkey"], "n_name": b["n_name"]}).filter(
            pc.is_in(b["n_name"], value_set=pair)
        ),
    ).materialize()
    supp = mb(
        join(
            mb(
                h["supplier"],
                lambda b: pa.table({"s_suppkey": b["s_suppkey"], "s_nationkey": b["s_nationkey"]}),
            ),
            two,
            "s_nationkey",
            "n_nationkey",
        ),
        lambda b: pa.table({"s_suppkey": b["s_suppkey"], "supp_nation": b["n_name"]}),
    )
    cust = mb(
        join(
            mb(
                h["customer"],
                lambda b: pa.table({"c_custkey": b["c_custkey"], "c_nationkey": b["c_nationkey"]}),
            ),
            two,
            "c_nationkey",
            "n_nationkey",
        ),
        lambda b: pa.table({"c_custkey": b["c_custkey"], "cust_nation": b["n_name"]}),
    )
    oc = join(
        mb(
            h["orders"],
            lambda b: pa.table({"o_orderkey": b["o_orderkey"], "o_custkey": b["o_custkey"]}),
        ),
        cust,
        "o_custkey",
        "c_custkey",
    )

    def line(b: pa.Table) -> pa.Table:
        b = b.filter(
            pc.and_(pc.greater_equal(b["l_shipdate"], lo), pc.less_equal(b["l_shipdate"], hi))
        )
        return pa.table(
            {
                "l_orderkey": b["l_orderkey"],
                "l_suppkey": b["l_suppkey"],
                "volume": revenue(b),
                "l_year": year_of(b["l_shipdate"]),
            }
        )

    joined = join(
        join(mb(h["lineitem"], line), oc, "l_orderkey", "o_orderkey"),
        supp,
        "l_suppkey",
        "s_suppkey",
    )

    def cross_pair(b: pa.Table) -> pa.Table:
        fr_de = pc.and_(pc.equal(b["supp_nation"], "FRANCE"), pc.equal(b["cust_nation"], "GERMANY"))
        de_fr = pc.and_(pc.equal(b["supp_nation"], "GERMANY"), pc.equal(b["cust_nation"], "FRANCE"))
        return b.filter(pc.or_(fr_de, de_fr))

    g = (
        mb(joined, cross_pair)
        .groupby(["supp_nation", "cust_nation", "l_year"])
        .aggregate(Sum("volume"))
    )
    rows = take(g)
    rows.sort(key=lambda r: (r["supp_nation"], r["cust_nation"], r["l_year"]))
    return pa.table(
        {
            "supp_nation": [r["supp_nation"] for r in rows],
            "cust_nation": [r["cust_nation"] for r in rows],
            "l_year": [r["l_year"] for r in rows],
            "revenue": [r["sum(volume)"] for r in rows],
        }
    )


# --------------------------------------------------------------------------- #
# q8 - eight-way join reduced to a per-year market-share ratio.
# --------------------------------------------------------------------------- #
@impl("tpch-q8")
def q8(h: dict[str, Any]) -> pa.Table:
    from ray.data.aggregate import Sum

    lo, hi = _d(date(1995, 1, 1)), _d(date(1996, 12, 31))

    parts = mb(
        h["part"],
        lambda b: pa.table(
            {"p_partkey": b.filter(pc.equal(b["p_type"], "ECONOMY ANODIZED STEEL"))["p_partkey"]}
        ),
    )
    line = mb(
        h["lineitem"],
        lambda b: pa.table(
            {
                "l_orderkey": b["l_orderkey"],
                "l_partkey": b["l_partkey"],
                "l_suppkey": b["l_suppkey"],
                "volume": revenue(b),
            }
        ),
    )
    lp = join(line, parts, "l_partkey", "p_partkey")

    def ords(b: pa.Table) -> pa.Table:
        b = b.filter(
            pc.and_(pc.greater_equal(b["o_orderdate"], lo), pc.less_equal(b["o_orderdate"], hi))
        )
        return pa.table(
            {
                "o_orderkey": b["o_orderkey"],
                "o_custkey": b["o_custkey"],
                "o_year": year_of(b["o_orderdate"]),
            }
        )

    lpo = join(lp, mb(h["orders"], ords), "l_orderkey", "o_orderkey")
    america = _region_nations(h, "AMERICA")
    cust = mb(
        join(
            mb(
                h["customer"],
                lambda b: pa.table({"c_custkey": b["c_custkey"], "c_nationkey": b["c_nationkey"]}),
            ),
            america,
            "c_nationkey",
            "n_nationkey",
        ),
        lambda b: pa.table({"c_custkey": b["c_custkey"]}),
    )
    lpoc = join(lpo, cust, "o_custkey", "c_custkey")
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
    full = join(lpoc, supp_nation, "l_suppkey", "s_suppkey")

    def split(b: pa.Table) -> pa.Table:
        brazil = pc.equal(b["nation"], "BRAZIL")
        return pa.table(
            {
                "o_year": b["o_year"],
                "brazil_volume": pc.if_else(brazil, b["volume"], 0.0),
                "volume": b["volume"],
            }
        )

    g = mb(full, split).groupby("o_year").aggregate(Sum("brazil_volume"), Sum("volume"))
    rows = take(g)
    rows.sort(key=lambda r: r["o_year"])
    return pa.table(
        {
            "o_year": [r["o_year"] for r in rows],
            "mkt_share": [r["sum(brazil_volume)"] / r["sum(volume)"] for r in rows],
        }
    )
