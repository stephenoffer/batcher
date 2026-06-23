"""Conditional-expression benchmarks (CASE / WHEN)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from batcher import col, count, lit, when
from registry import suite

if TYPE_CHECKING:
    from contexts import SyntheticContext

conditional = suite("conditional", dataset="synthetic")


@conditional.case("case-band")
def price_bands(ctx: SyntheticContext):
    """Bucket price into bands with CASE, count per band."""

    def batcher():
        band = (
            when(col("price") > 500.0)
            .then(lit("hi"))
            .when(col("price") > 250.0)
            .then(lit("mid"))
            .otherwise(lit("lo"))
        )
        return ctx.bf.group_by(band=band).agg(n=count()).collect()

    def duckdb():
        return ctx.con.sql(
            """SELECT CASE WHEN price > 500 THEN 'hi'
                           WHEN price > 250 THEN 'mid'
                           ELSE 'lo' END AS band,
                      COUNT(*) AS n
               FROM fact GROUP BY 1"""
        ).to_arrow_table()

    def polars():
        band = (
            pl.when(pl.col("price") > 500.0)
            .then(pl.lit("hi"))
            .when(pl.col("price") > 250.0)
            .then(pl.lit("mid"))
            .otherwise(pl.lit("lo"))
            .alias("band")
        )
        return ctx.pf.lazy().group_by(band).agg(pl.len().alias("n")).collect().to_arrow()

    return {"batcher": batcher, "duckdb": duckdb, "polars": polars}
