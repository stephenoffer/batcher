"""Ray Data pipelines for TPC-H q17-q22.

These are the correlated-subquery queries. Each correlation is rewritten as a grouped
aggregate re-joined onto the rows it correlates with, and each ``EXISTS`` / ``NOT IN``
as a semi / anti join -- the standard decorrelation, done explicitly because Ray Data
has no optimizer to do it.

Each function takes the ``{table name -> ray.data.Dataset}`` handle map and returns
the result as a ``pyarrow.Table`` whose column names match the SQL reference's
aliases exactly (the harness compares engines by column name + row multiset).
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from .base import impl, join, mb, revenue, take


def _d(value: date) -> pa.Scalar:
    return pa.scalar(value, pa.date32())


# --------------------------------------------------------------------------- #
# q17 - correlated avg(l_quantity) per part, as a grouped mean re-joined back.
# --------------------------------------------------------------------------- #
@impl("tpch-q17")
def q17(h: dict[str, Any]) -> pa.Table:
    from ray.data.aggregate import Mean

    def parts(b: pa.Table) -> pa.Table:
        keep = pc.and_(pc.equal(b["p_brand"], "Brand#23"), pc.equal(b["p_container"], "MED BOX"))
        return pa.table({"p_partkey": b.filter(keep)["p_partkey"]})

    line = mb(
        h["lineitem"],
        lambda b: pa.table(
            {
                "l_partkey": b["l_partkey"],
                "l_quantity": b["l_quantity"],
                "l_extendedprice": b["l_extendedprice"],
            }
        ),
    )
    lp = join(line, mb(h["part"], parts), "l_partkey", "p_partkey").materialize()
    avg_qty = mb(
        lp.groupby("l_partkey").aggregate(Mean("l_quantity")),
        lambda b: pa.table(
            {"l_partkey": b["l_partkey"], "avg_qty": pc.multiply(b["mean(l_quantity)"], 0.2)}
        ),
    )
    joined = join(lp, avg_qty, "l_partkey")
    small = mb(joined, lambda b: b.filter(pc.less(b["l_quantity"], b["avg_qty"])))
    return pa.table({"avg_yearly": [small.sum("l_extendedprice") / 7.0]})


# --------------------------------------------------------------------------- #
# q18 - orders whose total quantity exceeds 300, joined back for their detail.
# --------------------------------------------------------------------------- #
@impl("tpch-q18")
def q18(h: dict[str, Any]) -> pa.Table:
    from ray.data.aggregate import Sum

    # Feeds both the >300 group filter and the final detail join, so materialize once.
    qty = mb(
        h["lineitem"],
        lambda b: pa.table({"l_orderkey": b["l_orderkey"], "l_quantity": b["l_quantity"]}),
    ).materialize()
    big = mb(
        qty.groupby("l_orderkey").aggregate(Sum("l_quantity")),
        lambda b: pa.table(
            {"l_orderkey": b.filter(pc.greater(b["sum(l_quantity)"], 300))["l_orderkey"]}
        ),
    )
    ords = mb(
        h["orders"],
        lambda b: pa.table(
            {
                "o_orderkey": b["o_orderkey"],
                "o_custkey": b["o_custkey"],
                "o_orderdate": b["o_orderdate"],
                "o_totalprice": b["o_totalprice"],
            }
        ),
    )
    wanted = join(ords, big, "o_orderkey", "l_orderkey", how="left_semi")
    cust = mb(
        h["customer"],
        lambda b: pa.table({"c_custkey": b["c_custkey"], "c_name": b["c_name"]}),
    )
    oc = join(wanted, cust, "o_custkey", "c_custkey")
    full = join(qty, oc, "l_orderkey", "o_orderkey")
    g = full.groupby(
        ["l_orderkey", "c_name", "o_custkey", "o_orderdate", "o_totalprice"]
    ).aggregate(Sum("l_quantity"))

    rows = take(g)
    rows.sort(key=lambda r: (-r["o_totalprice"], r["o_orderdate"]))
    rows = rows[:100]
    return pa.table(
        {
            "c_name": [r["c_name"] for r in rows],
            "c_custkey": [r["o_custkey"] for r in rows],
            "o_orderkey": [r["l_orderkey"] for r in rows],
            "o_orderdate": [r["o_orderdate"] for r in rows],
            "o_totalprice": [r["o_totalprice"] for r in rows],
            "sum_qty": [r["sum(l_quantity)"] for r in rows],
        }
    )


# --------------------------------------------------------------------------- #
# q19 - one join under a three-branch disjunctive predicate.
# --------------------------------------------------------------------------- #
@impl("tpch-q19")
def q19(h: dict[str, Any]) -> pa.Table:
    modes = pa.array(["AIR", "AIR REG"])

    def line(b: pa.Table) -> pa.Table:
        keep = pc.and_(
            pc.is_in(b["l_shipmode"], value_set=modes),
            pc.equal(b["l_shipinstruct"], "DELIVER IN PERSON"),
        )
        b = b.filter(keep)
        return pa.table(
            {"l_partkey": b["l_partkey"], "l_quantity": b["l_quantity"], "rev": revenue(b)}
        )

    part = mb(
        h["part"],
        lambda b: pa.table(
            {
                "p_partkey": b["p_partkey"],
                "p_brand": b["p_brand"],
                "p_container": b["p_container"],
                "p_size": b["p_size"],
            }
        ),
    )
    joined = join(mb(h["lineitem"], line), part, "l_partkey", "p_partkey")

    def branch(b: pa.Table, brand: str, containers: list[str], qlo: int, psize: int) -> pa.Array:
        return pc.and_(
            pc.and_(
                pc.equal(b["p_brand"], brand),
                pc.is_in(b["p_container"], value_set=pa.array(containers)),
            ),
            pc.and_(
                pc.and_(
                    pc.greater_equal(b["l_quantity"], qlo),
                    pc.less_equal(b["l_quantity"], qlo + 10),
                ),
                pc.and_(pc.greater_equal(b["p_size"], 1), pc.less_equal(b["p_size"], psize)),
            ),
        )

    def keep(b: pa.Table) -> pa.Table:
        one = branch(b, "Brand#12", ["SM CASE", "SM BOX", "SM PACK", "SM PKG"], 1, 5)
        two = branch(b, "Brand#23", ["MED BAG", "MED BOX", "MED PKG", "MED PACK"], 10, 10)
        three = branch(b, "Brand#34", ["LG CASE", "LG BOX", "LG PACK", "LG PKG"], 20, 15)
        return b.filter(pc.or_(one, pc.or_(two, three)))

    return pa.table({"revenue": [mb(joined, keep).sum("rev")]})


# --------------------------------------------------------------------------- #
# q20 - nested IN subqueries: an availability threshold per (part, supplier),
# then the suppliers that clear it.
# --------------------------------------------------------------------------- #
@impl("tpch-q20")
def q20(h: dict[str, Any]) -> pa.Table:
    from ray.data.aggregate import Sum

    lo, hi = _d(date(1994, 1, 1)), _d(date(1995, 1, 1))

    forest = mb(
        h["part"],
        lambda b: pa.table(
            {"p_partkey": b.filter(pc.starts_with(b["p_name"], pattern="forest"))["p_partkey"]}
        ),
    )
    ps = mb(
        h["partsupp"],
        lambda b: pa.table(
            {
                "ps_partkey": b["ps_partkey"],
                "ps_suppkey": b["ps_suppkey"],
                "ps_availqty": b["ps_availqty"],
            }
        ),
    )
    ps_forest = join(ps, forest, "ps_partkey", "p_partkey")

    def line(b: pa.Table) -> pa.Table:
        b = b.filter(pc.and_(pc.greater_equal(b["l_shipdate"], lo), pc.less(b["l_shipdate"], hi)))
        return pa.table(
            {
                "l_partkey": b["l_partkey"],
                "l_suppkey": b["l_suppkey"],
                "l_quantity": b["l_quantity"],
            }
        )

    shipped = mb(
        mb(h["lineitem"], line).groupby(["l_partkey", "l_suppkey"]).aggregate(Sum("l_quantity")),
        lambda b: pa.table(
            {
                "l_partkey": b["l_partkey"],
                "l_suppkey": b["l_suppkey"],
                "half_qty": pc.multiply(b["sum(l_quantity)"], 0.5),
            }
        ),
    )
    matched = join(ps_forest, shipped, ("ps_partkey", "ps_suppkey"), ("l_partkey", "l_suppkey"))
    excess = mb(
        matched,
        lambda b: pa.table(
            {"ps_suppkey": b.filter(pc.greater(b["ps_availqty"], b["half_qty"]))["ps_suppkey"]}
        ),
    )

    canada = mb(
        h["nation"],
        lambda b: pa.table(
            {"n_nationkey": b.filter(pc.equal(b["n_name"], "CANADA"))["n_nationkey"]}
        ),
    )
    supp = join(
        mb(
            h["supplier"],
            lambda b: pa.table(
                {
                    "s_suppkey": b["s_suppkey"],
                    "s_name": b["s_name"],
                    "s_address": b["s_address"],
                    "s_nationkey": b["s_nationkey"],
                }
            ),
        ),
        canada,
        "s_nationkey",
        "n_nationkey",
    )
    result = join(supp, excess, "s_suppkey", "ps_suppkey", how="left_semi")
    rows = take(result)
    rows.sort(key=lambda r: r["s_name"])
    return pa.table(
        {
            "s_name": [r["s_name"] for r in rows],
            "s_address": [r["s_address"] for r in rows],
        }
    )


# --------------------------------------------------------------------------- #
# q21 - EXISTS plus NOT EXISTS over the same order. A line qualifies when its
# order has more than one supplier but exactly one *late* supplier -- which,
# since the line itself is late, must be its own.
# --------------------------------------------------------------------------- #
@impl("tpch-q21")
def q21(h: dict[str, Any]) -> pa.Table:
    from ray.data.aggregate import Count, CountDistinct

    keys = mb(
        h["lineitem"],
        lambda b: pa.table({"l_orderkey": b["l_orderkey"], "l_suppkey": b["l_suppkey"]}),
    )
    suppliers_per_order = mb(
        keys.groupby("l_orderkey").aggregate(CountDistinct("l_suppkey")),
        lambda b: pa.table(
            {"l_orderkey": b["l_orderkey"], "n_supp": b["count_distinct(l_suppkey)"]}
        ),
    )

    def late(b: pa.Table) -> pa.Table:
        b = b.filter(pc.greater(b["l_receiptdate"], b["l_commitdate"]))
        return pa.table({"l_orderkey": b["l_orderkey"], "l_suppkey": b["l_suppkey"]})

    late_lines = mb(h["lineitem"], late).materialize()
    late_per_order = mb(
        late_lines.groupby("l_orderkey").aggregate(CountDistinct("l_suppkey")),
        lambda b: pa.table(
            {"l_orderkey": b["l_orderkey"], "n_late": b["count_distinct(l_suppkey)"]}
        ),
    )

    ords = mb(
        h["orders"],
        lambda b: pa.table(
            {"o_orderkey": b.filter(pc.equal(b["o_orderstatus"], "F"))["o_orderkey"]}
        ),
    )
    cand = join(late_lines, ords, "l_orderkey", "o_orderkey", how="left_semi")
    cand = join(cand, suppliers_per_order, "l_orderkey")
    cand = join(cand, late_per_order, "l_orderkey")
    only_late = mb(
        cand,
        lambda b: b.filter(pc.and_(pc.greater(b["n_supp"], 1), pc.equal(b["n_late"], 1))),
    )

    saudi = mb(
        h["nation"],
        lambda b: pa.table(
            {"n_nationkey": b.filter(pc.equal(b["n_name"], "SAUDI ARABIA"))["n_nationkey"]}
        ),
    )
    supp = mb(
        join(
            mb(
                h["supplier"],
                lambda b: pa.table(
                    {
                        "s_suppkey": b["s_suppkey"],
                        "s_name": b["s_name"],
                        "s_nationkey": b["s_nationkey"],
                    }
                ),
            ),
            saudi,
            "s_nationkey",
            "n_nationkey",
        ),
        lambda b: pa.table({"s_suppkey": b["s_suppkey"], "s_name": b["s_name"]}),
    )
    joined = join(only_late, supp, "l_suppkey", "s_suppkey")
    rows = take(joined.groupby("s_name").aggregate(Count()))
    rows.sort(key=lambda r: (-r["count()"], r["s_name"]))
    rows = rows[:100]
    return pa.table(
        {"s_name": [r["s_name"] for r in rows], "numwait": [r["count()"] for r in rows]}
    )


# --------------------------------------------------------------------------- #
# q22 - country-code cohort above the positive-balance average, with no orders.
# --------------------------------------------------------------------------- #
@impl("tpch-q22")
def q22(h: dict[str, Any]) -> pa.Table:
    from ray.data.aggregate import Count, Sum

    codes = pa.array(["13", "31", "23", "29", "30", "18", "17"])

    def cohort(b: pa.Table) -> pa.Table:
        code = pc.utf8_slice_codeunits(b["c_phone"], 0, 2)
        b = b.append_column("cntrycode", code)
        b = b.filter(pc.is_in(code, value_set=codes))
        return pa.table(
            {
                "c_custkey": b["c_custkey"],
                "c_acctbal": b["c_acctbal"],
                "cntrycode": b["cntrycode"],
            }
        )

    cust = mb(h["customer"], cohort).materialize()
    positive = mb(cust, lambda b: b.filter(pc.greater(b["c_acctbal"], 0.0))).materialize()
    avg_bal = positive.sum("c_acctbal") / positive.count()

    rich = mb(cust, lambda b: b.filter(pc.greater(b["c_acctbal"], avg_bal)))
    ords = mb(h["orders"], lambda b: pa.table({"o_custkey": b["o_custkey"]}))
    no_orders = join(rich, ords, "c_custkey", "o_custkey", how="left_anti")

    rows = take(no_orders.groupby("cntrycode").aggregate(Count(), Sum("c_acctbal")))
    rows.sort(key=lambda r: r["cntrycode"])
    return pa.table(
        {
            "cntrycode": [r["cntrycode"] for r in rows],
            "numcust": [r["count()"] for r in rows],
            "totacctbal": [r["sum(c_acctbal)"] for r in rows],
        }
    )
