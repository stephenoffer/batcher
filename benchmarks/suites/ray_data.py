"""Head-to-head benchmarks vs **Ray Data** — the Ray Data-competitive operator mix.

Each case expresses the same query for Batcher, DuckDB, Polars, *and* Ray Data, and
the harness checks all engines agree before timing (the same correctness gate every
other suite uses — a fast wrong answer is never reported). The ``ray`` engine is
only in the lineup when ``ray`` + ``pandas`` are importable; elsewhere these cases
still run the other three.

This is the acceptance gate for the competitive claim: Batcher's numbers are only
meaningful next to the system it claims to beat, on identical inputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pyarrow as pa

from batcher import col
from registry import suite

if TYPE_CHECKING:
    from contexts import SyntheticContext

ray_data = suite("ray-data", dataset="synthetic")


@ray_data.case("ray:filter-count")
def filter_count(ctx: SyntheticContext):
    """COUNT(*) WHERE x > 0 — a streaming filter, reduced to a scalar."""

    def batcher():
        n = ctx.bf.filter(col("x") > 0).count()
        return pa.table({"n": pa.array([n], type=pa.int64())})

    def duckdb():
        return ctx.con.sql("SELECT COUNT(*) AS n FROM fact WHERE x > 0").to_arrow_table()

    def polars():
        out = ctx.pf.lazy().filter(pl.col("x") > 0).select(pl.len().alias("n")).collect()
        return out.to_arrow().cast(pa.schema([("n", pa.int64())]))

    def ray():
        n = ctx.ray("fact", ctx.fact).filter(expr="x > 0").count()
        return pa.table({"n": pa.array([n], type=pa.int64())})

    return {"batcher": batcher, "duckdb": duckdb, "polars": polars, "ray": ray}


@ray_data.case("ray:groupby-sum")
def groupby_sum(ctx: SyntheticContext):
    """GROUP BY k1, SUM(x) — the all-to-all aggregate (Ray Data's expensive path)."""

    def batcher():
        return ctx.bf.group_by("k1").agg(s=col("x").sum()).collect()

    def duckdb():
        return ctx.con.sql("SELECT k1, SUM(x) AS s FROM fact GROUP BY k1").to_arrow_table()

    def polars():
        return ctx.pf.lazy().group_by("k1").agg(pl.col("x").sum().alias("s")).collect().to_arrow()

    def ray():
        df = ctx.ray("fact", ctx.fact).groupby("k1").sum("x").to_pandas()
        df = df.rename(columns={"sum(x)": "s"})
        return pa.Table.from_pandas(df, preserve_index=False)

    return {"batcher": batcher, "duckdb": duckdb, "polars": polars, "ray": ray}


@ray_data.case("ray:global-sum")
def global_sum(ctx: SyntheticContext):
    """Global SUM(x) — a single mergeable reduction."""

    def batcher():
        return ctx.bf.group_by().agg(s=col("x").sum()).collect()

    def duckdb():
        return ctx.con.sql("SELECT SUM(x) AS s FROM fact").to_arrow_table()

    def polars():
        return ctx.pf.lazy().select(pl.col("x").sum().alias("s")).collect().to_arrow()

    def ray():
        total = ctx.ray("fact", ctx.fact).sum("x")
        return pa.table({"s": pa.array([total], type=pa.int64())})

    return {"batcher": batcher, "duckdb": duckdb, "polars": polars, "ray": ray}
