"""The `.dt` namespace over TPC-H `lineitem`'s three DATE columns, grouped to a few rows.

`ops-expressions` covers `EXTRACT(YEAR ...)` and interval arithmetic through SQL. These are the
temporal calls a DataFrame user writes next -- truncating to a period, formatting, the ISO
weekday, the day count between two dates -- each feeding a group-by so the result is small and
the gate compares all of it. Every engine is asked for the same output type: a truncated DATE
stays a DATE (Batcher's `preserve_type=True`, Polars' native behaviour, DuckDB cast back).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from registry import suite

from ._engines import frame_case

if TYPE_CHECKING:
    from context import Context

temporal = suite("ops-frame-temporal", dataset="operators")


def _grouped(keys: pa.Array, t: pa.Table, name: str) -> pa.Table:
    """PyArrow: group `l_quantity` by a computed key, as count and sum."""
    g = pa.table({name: keys, "q": t["l_quantity"]}).group_by(name)
    out = g.aggregate([("q", "count"), ("q", "sum")])
    return pa.table({name: out[name], "n": out["q_count"], "qty": out["q_sum"]})


@temporal.case("op-dt-truncate-month")
def truncate_month(ctx: Context):
    """Ship volume per calendar month: `dt.truncate("month")` as a group key (84 groups)."""
    import batcher as bt

    def batcher(ds):
        m = bt.col("l_shipdate").dt.truncate("month", preserve_type=True).alias("m")
        return (
            ds.with_columns(m)
            .group_by("m")
            .agg(n=bt.col("l_quantity").count(), qty=bt.col("l_quantity").sum())
            .to_arrow()
        )

    def polars(lf):
        import polars as pl

        return lf.group_by(pl.col("l_shipdate").dt.truncate("1mo").alias("m")).agg(
            pl.len().alias("n"), pl.col("l_quantity").sum().alias("qty")
        )

    sql = (
        "SELECT CAST(date_trunc('month', l_shipdate) AS DATE) AS m, count(*) AS n, "
        "sum(l_quantity) AS qty FROM {t} GROUP BY 1"
    )
    return frame_case(
        ctx,
        batcher=batcher,
        polars=polars,
        sql=sql,
        pyarrow=lambda t: _grouped(pc.floor_temporal(t["l_shipdate"], unit="month"), t, "m"),
    )


@temporal.case("op-dt-strftime")
def strftime(ctx: Context):
    """Ship volume per `"%Y-%m"` string: `dt.strftime` formatting 6M dates, then a group-by."""
    import batcher as bt

    def batcher(ds):
        ym = bt.col("l_shipdate").dt.strftime("%Y-%m").alias("ym")
        return (
            ds.with_columns(ym)
            .group_by("ym")
            .agg(n=bt.col("l_quantity").count(), qty=bt.col("l_quantity").sum())
            .to_arrow()
        )

    def polars(lf):
        import polars as pl

        return lf.group_by(pl.col("l_shipdate").dt.strftime("%Y-%m").alias("ym")).agg(
            pl.len().alias("n"), pl.col("l_quantity").sum().alias("qty")
        )

    def pyarrow(t):
        stamps = pc.cast(t["l_shipdate"], pa.timestamp("s"))
        return _grouped(pc.strftime(stamps, format="%Y-%m"), t, "ym")

    sql = (
        "SELECT strftime(l_shipdate, '%Y-%m') AS ym, count(*) AS n, sum(l_quantity) AS qty "
        "FROM {t} GROUP BY 1"
    )
    return frame_case(ctx, batcher=batcher, polars=polars, sql=sql, pyarrow=pyarrow)


@temporal.case("op-dt-weekday")
def weekday(ctx: Context):
    """Ship volume per ISO weekday (Monday 1 .. Sunday 7) of `l_shipdate`."""
    import batcher as bt

    def batcher(ds):
        return (
            ds.with_columns(bt.col("l_shipdate").dt.weekday().alias("wd"))
            .group_by("wd")
            .agg(n=bt.col("l_quantity").count(), qty=bt.col("l_quantity").sum())
            .to_arrow()
        )

    def polars(lf):
        import polars as pl

        return lf.group_by(pl.col("l_shipdate").dt.weekday().alias("wd")).agg(
            pl.len().alias("n"), pl.col("l_quantity").sum().alias("qty")
        )

    def pyarrow(t):
        wd = pc.day_of_week(t["l_shipdate"], count_from_zero=False, week_start=1)
        return _grouped(wd, t, "wd")

    sql = (
        "SELECT isodow(l_shipdate) AS wd, count(*) AS n, sum(l_quantity) AS qty FROM {t} GROUP BY 1"
    )
    return frame_case(ctx, batcher=batcher, polars=polars, sql=sql, pyarrow=pyarrow)


@temporal.case("op-dt-days-between")
def days_between(ctx: Context):
    """Mean and max transit days (`l_receiptdate - l_shipdate`) per ship mode."""
    import batcher as bt

    def batcher(ds):
        days = bt.col("l_receiptdate").dt.days_between(bt.col("l_shipdate"))
        return (
            ds.with_columns(days.alias("d"))
            .group_by("l_shipmode")
            .agg(avg_d=bt.col("d").mean(), max_d=bt.col("d").max())
            .to_arrow()
        )

    def polars(lf):
        import polars as pl

        d = (pl.col("l_receiptdate") - pl.col("l_shipdate")).dt.total_days()
        return lf.group_by("l_shipmode").agg(d.mean().alias("avg_d"), d.max().alias("max_d"))

    def pyarrow(t):
        d = pc.days_between(t["l_shipdate"], t["l_receiptdate"])
        g = pa.table({"l_shipmode": t["l_shipmode"], "d": d}).group_by("l_shipmode")
        out = g.aggregate([("d", "mean"), ("d", "max")])
        return pa.table(
            {"l_shipmode": out["l_shipmode"], "avg_d": out["d_mean"], "max_d": out["d_max"]}
        )

    sql = (
        "SELECT l_shipmode, avg(datediff('day', l_shipdate, l_receiptdate)) AS avg_d, "
        "max(datediff('day', l_shipdate, l_receiptdate)) AS max_d FROM {t} GROUP BY 1"
    )
    return frame_case(ctx, batcher=batcher, polars=polars, sql=sql, pyarrow=pyarrow)
