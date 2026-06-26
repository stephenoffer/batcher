"""Operator-mix: aggregation over TPC-H ``lineitem`` (group-by, global, filtered count).

SQL engines run the one SQL string; PyArrow (Acero ``group_by``) and Ray Data get
native implementations so the two non-SQL engines compete here too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from registry import suite

from .base import sql_fanout, with_native

if TYPE_CHECKING:
    from context import Context

agg = suite("ops-aggregation", dataset="operators")


@agg.case("op-groupby-sum")
def groupby_sum(ctx: Context):
    """GROUP BY l_returnflag, SUM(l_quantity) — the single-key all-to-all aggregate."""
    sql = "SELECT l_returnflag, SUM(l_quantity) AS s FROM lineitem GROUP BY l_returnflag"

    def pyarrow(t: pa.Table) -> pa.Table:
        a = t.group_by("l_returnflag").aggregate([("l_quantity", "sum")])
        return pa.table({"l_returnflag": a["l_returnflag"], "s": a["l_quantity_sum"]})

    def ray(rd) -> pa.Table:
        df = rd.groupby("l_returnflag").sum("l_quantity").to_pandas()
        return pa.Table.from_pandas(
            df.rename(columns={"sum(l_quantity)": "s"}), preserve_index=False
        )

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow, ray=ray)


@agg.case("op-groupby-2key")
def groupby_2key(ctx: Context):
    """GROUP BY l_returnflag, l_linestatus with SUM and COUNT — a two-key aggregate."""
    sql = (
        "SELECT l_returnflag, l_linestatus, SUM(l_quantity) AS s, COUNT(*) AS n "
        "FROM lineitem GROUP BY l_returnflag, l_linestatus"
    )

    def pyarrow(t: pa.Table) -> pa.Table:
        a = t.group_by(["l_returnflag", "l_linestatus"]).aggregate(
            [("l_quantity", "sum"), ("l_quantity", "count")]
        )
        return pa.table(
            {
                "l_returnflag": a["l_returnflag"],
                "l_linestatus": a["l_linestatus"],
                "s": a["l_quantity_sum"],
                "n": a["l_quantity_count"],
            }
        )

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow)


@agg.case("op-global-sum")
def global_sum(ctx: Context):
    """Global SUM(l_extendedprice) — a single mergeable reduction."""
    sql = "SELECT SUM(l_extendedprice) AS s FROM lineitem"

    def pyarrow(t: pa.Table) -> pa.Table:
        return pa.table({"s": pa.array([pc.sum(t["l_extendedprice"]).as_py()])})

    def ray(rd) -> pa.Table:
        return pa.table({"s": pa.array([rd.sum("l_extendedprice")])})

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow, ray=ray)


@agg.case("op-filter-count")
def filter_count(ctx: Context):
    """COUNT(*) WHERE l_quantity > 25 — a streaming filter reduced to a scalar."""
    sql = "SELECT COUNT(*) AS n FROM lineitem WHERE l_quantity > 25"

    def pyarrow(t: pa.Table) -> pa.Table:
        n = t.filter(pc.greater(t["l_quantity"], 25)).num_rows
        return pa.table({"n": pa.array([n], type=pa.int64())})

    def ray(rd) -> pa.Table:
        n = rd.filter(expr="l_quantity > 25").count()
        return pa.table({"n": pa.array([n], type=pa.int64())})

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow, ray=ray)
