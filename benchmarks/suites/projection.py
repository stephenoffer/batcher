"""Projection benchmarks: per-row arithmetic over columns."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from batcher import col
from registry import suite

if TYPE_CHECKING:
    from contexts import SyntheticContext

projection = suite("projection", dataset="synthetic")


@projection.case("projection")
def compound_arithmetic(ctx: SyntheticContext):
    """total = price*qty - x/2, summed to one scalar (drives full per-row arithmetic)."""

    def batcher():
        return (
            ctx.bf.select(total=col("price") * col("qty") - col("x") / 2.0)
            .group_by()
            .agg(s=col("total").sum())
            .collect()
        )

    def duckdb():
        return ctx.con.sql("SELECT SUM(price * qty - x / 2.0) AS s FROM fact").to_arrow_table()

    def polars():
        return (
            ctx.pf.lazy()
            .select((pl.col("price") * pl.col("qty") - pl.col("x") / 2.0).alias("total"))
            .select(pl.col("total").sum().alias("s"))
            .collect()
            .to_arrow()
        )

    return {"batcher": batcher, "duckdb": duckdb, "polars": polars}
