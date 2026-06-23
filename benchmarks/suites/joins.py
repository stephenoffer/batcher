"""Join benchmarks: inner, left, semi, anti."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pyarrow as pa

from batcher import col, count
from registry import suite

if TYPE_CHECKING:
    from contexts import SyntheticContext

joins = suite("joins", dataset="synthetic")


@joins.case("join+groupby")
def inner_join_groupby(ctx: SyntheticContext):
    """Inner join fact to dim, then GROUP BY region."""

    def batcher():
        return (
            ctx.bf.join(ctx.bd, on="dim_key", how="inner")
            .group_by("region")
            .agg(s=col("price").sum(), n=count())
            .collect()
        )

    def duckdb():
        return ctx.con.sql(
            """SELECT d.region AS region, SUM(f.price) AS s, COUNT(*) AS n
               FROM fact f JOIN dim d ON f.dim_key = d.dim_key
               GROUP BY d.region"""
        ).to_arrow_table()

    def polars():
        return (
            ctx.pf.lazy()
            .join(ctx.pd_dim.lazy(), on="dim_key", how="inner")
            .group_by("region")
            .agg(pl.col("price").sum().alias("s"), pl.len().alias("n"))
            .collect()
            .to_arrow()
        )

    return {"batcher": batcher, "duckdb": duckdb, "polars": polars}


@joins.case("join-left")
def left_join(ctx: SyntheticContext):
    """Left join to a half-size dim: count rows and non-null regions."""

    def batcher():
        return (
            ctx.bf.join(ctx.bd2, on="dim_key", how="left")
            .group_by()
            .agg(n=count(), nr=col("region").count())
            .collect()
        )

    def duckdb():
        return ctx.con.sql(
            """SELECT COUNT(*) AS n, COUNT(d.region) AS nr
               FROM fact f LEFT JOIN dim2 d ON f.dim_key = d.dim_key"""
        ).to_arrow_table()

    def polars():
        return (
            ctx.pf.lazy()
            .join(ctx.pd_dim2.lazy(), on="dim_key", how="left")
            .select(pl.len().alias("n"), pl.col("region").count().alias("nr"))
            .collect()
            .to_arrow()
        )

    return {"batcher": batcher, "duckdb": duckdb, "polars": polars}


@joins.case("join-semi")
def semi_join(ctx: SyntheticContext):
    """Semi join: keep fact rows whose key matches the dim subset."""

    def batcher():
        n = ctx.bf.join(ctx.bd2, on="dim_key", how="semi").count()
        return pa.table({"n": pa.array([n], type=pa.int64())})

    def duckdb():
        return ctx.con.sql(
            "SELECT COUNT(*) AS n FROM fact WHERE dim_key IN (SELECT dim_key FROM dim2)"
        ).to_arrow_table()

    def polars():
        out = (
            ctx.pf.lazy()
            .join(ctx.pd_dim2.lazy(), on="dim_key", how="semi")
            .select(pl.len().alias("n"))
        )
        return out.collect().to_arrow().cast(pa.schema([("n", pa.int64())]))

    return {"batcher": batcher, "duckdb": duckdb, "polars": polars}


@joins.case("join-anti")
def anti_join(ctx: SyntheticContext):
    """Anti join: keep fact rows with no matching key in the dim subset."""

    def batcher():
        n = ctx.bf.join(ctx.bd2, on="dim_key", how="anti").count()
        return pa.table({"n": pa.array([n], type=pa.int64())})

    def duckdb():
        return ctx.con.sql(
            "SELECT COUNT(*) AS n FROM fact WHERE dim_key NOT IN (SELECT dim_key FROM dim2)"
        ).to_arrow_table()

    def polars():
        out = (
            ctx.pf.lazy()
            .join(ctx.pd_dim2.lazy(), on="dim_key", how="anti")
            .select(pl.len().alias("n"))
        )
        return out.collect().to_arrow().cast(pa.schema([("n", pa.int64())]))

    return {"batcher": batcher, "duckdb": duckdb, "polars": polars}
