"""Ray Data native pipelines for the TPC-H queries it can express.

Ray Data has no SQL surface, so the standard suite otherwise leaves it ``n/a`` on
all 22 TPC-H queries. This module supplies hand-written ``ray.data.Dataset``
pipelines for the queries that map cleanly to Ray Data's API — scan/filter/aggregate
and straightforward equi-joins — so Ray Data is measured on whole-query workloads,
not just the operator-mix. Queries that need correlated subqueries or deep multi-join
nests are intentionally absent (Ray Data cannot express them) and stay ``n/a`` — never
a wrong answer, because the harness gates each engine on correctness vs the SQL
reference before timing.

Each pipeline does its arithmetic and filtering in a ``map_batches`` over PyArrow
(the format Ray Data's blocks already use) and its grouping via Ray Data's native
``groupby().aggregate`` / ``join``, then renames the aggregate outputs to the SQL
aliases the harness compares on (column names must match the reference exactly).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc

from registry import EngineQueries, sql_case

if TYPE_CHECKING:
    from context import Context

# A Ray pipeline: a mapping of table name -> Ray Dataset handle, returning the result.
RayImpl = Callable[[dict[str, Any]], pa.Table]
_IMPLS: dict[str, RayImpl] = {}

# Tables a Ray pipeline may ask for (handles built lazily, only when ray is active).
_TPCH_TABLES = (
    "lineitem",
    "orders",
    "customer",
    "part",
    "supplier",
    "partsupp",
    "nation",
    "region",
)


# Ray Data's hash-join fans each side into this many partitions joined independently.
_JOIN_PARTITIONS = 64


def _impl(name: str) -> Callable[[RayImpl], RayImpl]:
    def register(fn: RayImpl) -> RayImpl:
        _IMPLS[name] = fn
        return fn

    return register


def case_with_ray(name: str, query: str) -> Callable[[Context], EngineQueries]:
    """Build a TPC-H case: the SQL fanout, plus a Ray pipeline when one exists.

    Mirrors the operator-mix ``with_native`` mechanism but threads *all* TPC-H table
    handles (a query joins several), so Ray Data competes on the full query.
    """
    sql_build = sql_case(query)

    def build(ctx: Context) -> EngineQueries:
        fns = sql_build(ctx)
        impl = _IMPLS.get(name)
        if impl is not None and "ray" in ctx.names():
            handles = {t: ctx.handle(t, "ray") for t in _TPCH_TABLES if t in ctx.tables}
            fns["ray"] = lambda: impl(handles)
        return fns

    return build


def _take(ds: Any) -> list[dict]:
    """Materialize a (small, post-aggregation) Ray Dataset to a list of row dicts."""
    return ds.take_all()


# --------------------------------------------------------------------------- #
# q1 — scan → filter → two-key aggregate (sum/avg/count). The canonical
# aggregate workload; no joins, the shape Ray Data's groupby targets.
# --------------------------------------------------------------------------- #
@_impl("tpch-q1")
def _q1(h: dict[str, Any]) -> pa.Table:
    from ray.data.aggregate import Count, Mean, Sum

    cutoff = pa.scalar(date(1998, 9, 2), pa.date32())  # 1998-12-01 minus 90 days

    def prep(b: pa.Table) -> pa.Table:
        b = b.filter(pc.less_equal(b["l_shipdate"], cutoff))
        disc = pc.multiply(b["l_extendedprice"], pc.subtract(1.0, b["l_discount"]))
        charge = pc.multiply(disc, pc.add(1.0, b["l_tax"]))
        return pa.table(
            {
                "l_returnflag": b["l_returnflag"],
                "l_linestatus": b["l_linestatus"],
                "l_quantity": b["l_quantity"],
                "l_extendedprice": b["l_extendedprice"],
                "disc": disc,
                "charge": charge,
                "l_discount": b["l_discount"],
            }
        )

    ds = h["lineitem"].map_batches(prep, batch_format="pyarrow")
    g = ds.groupby(["l_returnflag", "l_linestatus"]).aggregate(
        Sum("l_quantity"),
        Sum("l_extendedprice"),
        Sum("disc"),
        Sum("charge"),
        Mean("l_quantity"),
        Mean("l_extendedprice"),
        Mean("l_discount"),
        Count(),
    )
    rows = _take(g)
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


# --------------------------------------------------------------------------- #
# q6 — scan → filter → single global sum. A pure streaming reduction.
# --------------------------------------------------------------------------- #
@_impl("tpch-q6")
def _q6(h: dict[str, Any]) -> pa.Table:
    lo = pa.scalar(date(1994, 1, 1), pa.date32())
    hi = pa.scalar(date(1995, 1, 1), pa.date32())

    def rev(b: pa.Table) -> pa.Table:
        mask = pc.and_(
            pc.and_(
                pc.greater_equal(b["l_shipdate"], lo),
                pc.less(b["l_shipdate"], hi),
            ),
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

    ds = h["lineitem"].map_batches(rev, batch_format="pyarrow")
    return pa.table({"revenue": [ds.sum("r")]})


# NOTE on coverage: queries with *chained* shuffle-joins (q3/q5/q7-q10/q21) or
# correlated subqueries (q22) are intentionally not implemented — Ray Data's repeated
# all-to-all hash-join shuffles are impractically slow at these scales (a single q3
# double-join does not finish in minutes at SF1) and several are not expressible in
# its API at all. Those queries keep Ray Data at `n/a` (never a wrong answer). The
# pipelines below cover the shapes Ray Data is built for: scan/filter/aggregate and a
# single equi-join feeding an aggregate.


# --------------------------------------------------------------------------- #
# q12 — join(orders, lineitem) → filter → conditional two-bucket aggregate.
# --------------------------------------------------------------------------- #
@_impl("tpch-q12")
def _q12(h: dict[str, Any]) -> pa.Table:
    from ray.data.aggregate import Sum

    lo = pa.scalar(date(1994, 1, 1), pa.date32())
    hi = pa.scalar(date(1995, 1, 1), pa.date32())
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

    line = h["lineitem"].map_batches(prep_line, batch_format="pyarrow")
    orders = h["orders"].map_batches(
        lambda b: pa.table(
            {"o_orderkey": b["o_orderkey"], "o_orderpriority": b["o_orderpriority"]}
        ),
        batch_format="pyarrow",
    )
    joined = line.join(
        orders,
        join_type="inner",
        num_partitions=_JOIN_PARTITIONS,
        on=("l_orderkey",),
        right_on=("o_orderkey",),
    )

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
        joined.map_batches(buckets, batch_format="pyarrow")
        .groupby("l_shipmode")
        .aggregate(Sum("high_line_count"), Sum("low_line_count"))
    )
    rows = sorted(_take(g), key=lambda r: r["l_shipmode"])
    return pa.table(
        {
            "l_shipmode": [r["l_shipmode"] for r in rows],
            "high_line_count": [r["sum(high_line_count)"] for r in rows],
            "low_line_count": [r["sum(low_line_count)"] for r in rows],
        }
    )


# --------------------------------------------------------------------------- #
# q14 — join(lineitem, part) → filter → ratio of promo revenue to total.
# --------------------------------------------------------------------------- #
@_impl("tpch-q14")
def _q14(h: dict[str, Any]) -> pa.Table:
    lo = pa.scalar(date(1995, 9, 1), pa.date32())
    hi = pa.scalar(date(1995, 10, 1), pa.date32())

    def prep_line(b: pa.Table) -> pa.Table:
        b = b.filter(pc.and_(pc.greater_equal(b["l_shipdate"], lo), pc.less(b["l_shipdate"], hi)))
        rev = pc.multiply(b["l_extendedprice"], pc.subtract(1.0, b["l_discount"]))
        return pa.table({"l_partkey": b["l_partkey"], "rev": rev})

    line = h["lineitem"].map_batches(prep_line, batch_format="pyarrow")
    part = h["part"].map_batches(
        lambda b: pa.table({"p_partkey": b["p_partkey"], "p_type": b["p_type"]}),
        batch_format="pyarrow",
    )
    joined = line.join(
        part,
        join_type="inner",
        num_partitions=_JOIN_PARTITIONS,
        on=("l_partkey",),
        right_on=("p_partkey",),
    )

    def contrib(b: pa.Table) -> pa.Table:
        promo = pc.starts_with(b["p_type"], pattern="PROMO")
        return pa.table(
            {
                "promo_rev": pc.if_else(promo, b["rev"], 0.0),
                "total_rev": b["rev"],
            }
        )

    c = joined.map_batches(contrib, batch_format="pyarrow")
    promo = c.sum("promo_rev")
    total = c.sum("total_rev")
    return pa.table({"promo_revenue": [100.0 * promo / total]})
