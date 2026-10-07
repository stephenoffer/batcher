"""Grouping shapes beyond one key and a sum: `rollup`, `cube`, `max_by`, `corr`.

`rollup` and `cube` appear only inside TPC-DS SQL; here each is timed alone, through the
`Dataset` verb, against DuckDB's `GROUP BY ROLLUP/CUBE` (Polars has neither). `max_by` and `corr`
are two of the aggregates a feature pipeline uses most that the aggregation families do not
time. `max_by`'s ordering key is `lineitem`'s unique `(l_orderkey, l_linenumber)` folded into
one integer, so the row it picks is defined.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from registry import suite

from ._engines import frame_case

if TYPE_CHECKING:
    from context import Context

grouping = suite("ops-frame-grouping", dataset="operators")


@grouping.case("op-rollup")
def rollup(ctx: Context):
    """Quantity over (flag, status) with both subtotal levels and the grand total (9 rows)."""
    import batcher as bt

    def batcher(ds):
        g = ds.rollup("l_returnflag", "l_linestatus")
        return g.agg(qty=bt.col("l_quantity").sum(), n=bt.col("l_quantity").count()).to_arrow()

    sql = (
        "SELECT l_returnflag, l_linestatus, sum(l_quantity) AS qty, count(*) AS n FROM {t} "
        "GROUP BY ROLLUP (l_returnflag, l_linestatus)"
    )
    return frame_case(ctx, batcher=batcher, sql=sql)


@grouping.case("op-cube")
def cube(ctx: Context):
    """Revenue over every subset of (flag, status, ship mode): eight grouping sets at once."""
    import batcher as bt

    def batcher(ds):
        g = ds.cube("l_returnflag", "l_linestatus", "l_shipmode")
        return g.agg(rev=bt.col("l_extendedprice").sum()).to_arrow()

    sql = (
        "SELECT l_returnflag, l_linestatus, l_shipmode, sum(l_extendedprice) AS rev FROM {t} "
        "GROUP BY CUBE (l_returnflag, l_linestatus, l_shipmode)"
    )
    return frame_case(ctx, batcher=batcher, sql=sql)


@grouping.case("op-max-by")
def max_by(ctx: Context):
    """Per supplier, the price of its last line by (order, line number): `max_by`."""
    import batcher as bt

    key = bt.col("l_orderkey") * 8 + bt.col("l_linenumber")

    def batcher(ds):
        g = ds.with_columns(k=key).group_by("l_suppkey")
        return g.agg(last_px=bt.col("l_extendedprice").max_by(bt.col("k"))).to_arrow()

    def polars(lf):
        import polars as pl

        k = pl.col("l_orderkey") * 8 + pl.col("l_linenumber")
        return lf.group_by("l_suppkey").agg(
            pl.col("l_extendedprice").get(k.arg_max()).alias("last_px")
        )

    sql = (
        "SELECT l_suppkey, arg_max(l_extendedprice, l_orderkey * 8 + l_linenumber) AS last_px "
        "FROM {t} GROUP BY l_suppkey"
    )
    return frame_case(ctx, batcher=batcher, polars=polars, sql=sql)


@grouping.case("op-corr")
def corr(ctx: Context):
    """Pearson correlation of quantity and price per ship mode (`corr`, a two-pass moment)."""
    import batcher as bt

    def batcher(ds):
        g = ds.group_by("l_shipmode")
        return g.agg(r=bt.corr("l_quantity", "l_extendedprice")).to_arrow()

    def polars(lf):
        import polars as pl

        return lf.group_by("l_shipmode").agg(pl.corr("l_quantity", "l_extendedprice").alias("r"))

    sql = "SELECT l_shipmode, corr(l_quantity, l_extendedprice) AS r FROM {t} GROUP BY 1"
    return frame_case(ctx, batcher=batcher, polars=polars, sql=sql)
