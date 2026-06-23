"""Grouped aggregation benchmarks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from batcher import col, count
from registry import suite

if TYPE_CHECKING:
    from contexts import SyntheticContext

aggregation = suite("aggregation", dataset="synthetic")


@aggregation.case("groupby-agg")
def groupby_single_key(ctx: SyntheticContext):
    """GROUP BY k1 with sum/count/avg/min/max."""

    def batcher():
        return (
            ctx.bf.group_by("k1")
            .agg(
                s=col("price").sum(),
                n=count(),
                a=col("price").mean(),
                mn=col("price").min(),
                mx=col("price").max(),
            )
            .collect()
        )

    def duckdb():
        return ctx.con.sql(
            """SELECT k1,
                      SUM(price) AS s, COUNT(*) AS n, AVG(price) AS a,
                      MIN(price) AS mn, MAX(price) AS mx
               FROM fact GROUP BY k1"""
        ).to_arrow_table()

    def polars():
        return (
            ctx.pf.lazy()
            .group_by("k1")
            .agg(
                pl.col("price").sum().alias("s"),
                pl.len().alias("n"),
                pl.col("price").mean().alias("a"),
                pl.col("price").min().alias("mn"),
                pl.col("price").max().alias("mx"),
            )
            .collect()
            .to_arrow()
        )

    return {"batcher": batcher, "duckdb": duckdb, "polars": polars}


@aggregation.case("groupby-2key")
def groupby_two_keys(ctx: SyntheticContext):
    """GROUP BY k1, k2 with sum/count."""

    def batcher():
        return ctx.bf.group_by("k1", "k2").agg(s=col("qty").sum(), n=count()).collect()

    def duckdb():
        return ctx.con.sql(
            "SELECT k1, k2, SUM(qty) AS s, COUNT(*) AS n FROM fact GROUP BY k1, k2"
        ).to_arrow_table()

    def polars():
        return (
            ctx.pf.lazy()
            .group_by("k1", "k2")
            .agg(pl.col("qty").sum().alias("s"), pl.len().alias("n"))
            .collect()
            .to_arrow()
        )

    return {"batcher": batcher, "duckdb": duckdb, "polars": polars}
