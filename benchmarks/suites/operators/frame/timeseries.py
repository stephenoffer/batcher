"""Polars-style per-group sequence expressions: `cum_sum`, `diff`, `rolling_mean`, forward fill.

`ops-window` times the same shapes as SQL `OVER` clauses. A DataFrame user writes them as
expressions -- `col.cum_sum(partition_by=..., order_by=...)`, Polars' `.over(..., order_by=...)`
-- and those lower through a different path, so they are timed here as written. Every case
orders within `l_orderkey` by `l_linenumber`, which is unique inside an order, so each row's
value is defined; the per-row results are then reduced to a sum and a count, which the gate
compares within its float tolerance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from registry import suite

from ._engines import frame_case

if TYPE_CHECKING:
    from context import Context

timeseries = suite("ops-frame-timeseries", dataset="operators")

_PART = ["l_orderkey"]
_ORDER = ["l_linenumber"]
_OVER = "OVER (PARTITION BY l_orderkey ORDER BY l_linenumber"


@timeseries.case("op-ts-cum-sum")
def cum_sum(ctx: Context):
    """Running revenue within each order, as an expression, then summed."""
    import batcher as bt

    def batcher(ds):
        run = bt.col("l_extendedprice").cum_sum(partition_by=_PART, order_by=_ORDER)
        r = bt.col("r")
        return ds.with_columns(r=run).agg(total=r.sum(), n=r.count()).to_arrow()

    def polars(lf):
        import polars as pl

        run = pl.col("l_extendedprice").cum_sum().over("l_orderkey", order_by="l_linenumber")
        return lf.select(run.alias("r")).select(
            pl.col("r").sum().alias("total"), pl.col("r").count().alias("n")
        )

    sql = (
        f"SELECT sum(r) AS total, count(r) AS n FROM (SELECT sum(l_extendedprice) {_OVER}) "
        "AS r FROM {t})"
    )
    return frame_case(ctx, batcher=batcher, polars=polars, sql=sql)


@timeseries.case("op-ts-diff")
def diff(ctx: Context):
    """Price change from the previous line of the same order (`diff(1)`), as abs-sum and count."""
    import batcher as bt

    def batcher(ds):
        step = bt.col("l_extendedprice").diff(1, partition_by=_PART, order_by=_ORDER)
        d = bt.col("d")
        return ds.with_columns(d=step).agg(total=d.abs().sum(), n=d.count()).to_arrow()

    def polars(lf):
        import polars as pl

        step = pl.col("l_extendedprice").diff(1).over("l_orderkey", order_by="l_linenumber")
        return lf.select(step.alias("d")).select(
            pl.col("d").abs().sum().alias("total"), pl.col("d").count().alias("n")
        )

    sql = (
        "SELECT sum(abs(d)) AS total, count(d) AS n FROM "
        f"(SELECT l_extendedprice - lag(l_extendedprice) {_OVER}) AS d FROM {{t}})"
    )
    return frame_case(ctx, batcher=batcher, polars=polars, sql=sql)


@timeseries.case("op-ts-rolling-mean")
def rolling_mean(ctx: Context):
    """Three-line trailing mean of price per order (`rolling_mean(3, min_periods=1)`)."""
    import batcher as bt

    def batcher(ds):
        roll = bt.col("l_extendedprice").rolling_mean(
            3, min_periods=1, partition_by=_PART, order_by=_ORDER
        )
        m = bt.col("m")
        return ds.with_columns(m=roll).agg(total=m.sum(), n=m.count()).to_arrow()

    def polars(lf):
        import polars as pl

        roll = (
            pl.col("l_extendedprice")
            .rolling_mean(3, min_samples=1)
            .over("l_orderkey", order_by="l_linenumber")
        )
        return lf.select(roll.alias("m")).select(
            pl.col("m").sum().alias("total"), pl.col("m").count().alias("n")
        )

    sql = (
        "SELECT sum(m) AS total, count(m) AS n FROM (SELECT avg(l_extendedprice) "
        f"{_OVER} ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS m FROM {{t}})"
    )
    return frame_case(ctx, batcher=batcher, polars=polars, sql=sql)


def _gappy(ctx: Context) -> pa.Table:
    """`lineitem`'s discount with every 0.00 discount (~9% of lines) turned into a null."""
    line = ctx.table("lineitem")
    disc = pc.cast(line["l_discount"], pa.float64())
    gappy = pc.if_else(pc.equal(disc, 0.0), pa.scalar(None, pa.float64()), disc)
    return pa.table(
        {"l_orderkey": line["l_orderkey"], "l_linenumber": line["l_linenumber"], "d": gappy}
    )


@timeseries.case("op-ts-forward-fill")
def forward_fill(ctx: Context):
    """Carry the last non-null discount forward within each order, then sum and count."""
    import batcher as bt

    def batcher(ds):
        filled = ds.fill_null(strategy="forward", subset=["d"], order_by=_ORDER, partition_by=_PART)
        return filled.agg(total=bt.col("d").sum(), n=bt.col("d").count()).to_arrow()

    def polars(lf):
        import polars as pl

        filled = pl.col("d").forward_fill().over("l_orderkey", order_by="l_linenumber")
        return lf.select(filled.alias("d")).select(
            pl.col("d").sum().alias("total"), pl.col("d").count().alias("n")
        )

    sql = (
        "SELECT sum(f) AS total, count(f) AS n FROM (SELECT last_value(d IGNORE NULLS) "
        f"{_OVER} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS f FROM {{t}})"
    )
    return frame_case(ctx, data=_gappy(ctx), batcher=batcher, polars=polars, sql=sql)
