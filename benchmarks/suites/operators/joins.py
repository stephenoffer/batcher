"""Operator-mix: hash join + aggregate over TPC-H ``lineitem`` ⋈ ``orders``.

A join followed by a small grouped aggregate (revenue by order priority) keeps the
result tiny, so the correctness gate compares a handful of rows rather than the
multi-million-row join output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from registry import suite

from .base import sql_fanout

if TYPE_CHECKING:
    from context import Context

joins = suite("ops-joins", dataset="operators")


@joins.case("op-join-agg")
def join_agg(ctx: Context):
    """Revenue by order priority — the canonical join-then-aggregate pipeline."""
    sql = (
        "SELECT o.o_orderpriority, "
        "SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue "
        "FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey "
        "GROUP BY o.o_orderpriority"
    )
    fns = sql_fanout(ctx, sql)

    if "pyarrow" in ctx.names():
        lineitem, orders = ctx.table("lineitem"), ctx.table("orders")

        def pyarrow() -> pa.Table:
            joined = lineitem.join(
                orders.select(["o_orderkey", "o_orderpriority"]),
                keys="l_orderkey",
                right_keys="o_orderkey",
                join_type="inner",
            )
            revenue = pc.multiply(joined["l_extendedprice"], pc.subtract(1.0, joined["l_discount"]))
            joined = joined.append_column("revenue", revenue)
            a = joined.group_by("o_orderpriority").aggregate([("revenue", "sum")])
            return pa.table({"o_orderpriority": a["o_orderpriority"], "revenue": a["revenue_sum"]})

        fns["pyarrow"] = pyarrow

    return fns
