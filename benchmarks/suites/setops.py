"""Set-operation benchmarks: distinct, union, intersect, except."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pyarrow as pa

from batcher import col
from registry import suite

if TYPE_CHECKING:
    from contexts import SyntheticContext

setops = suite("setops", dataset="synthetic")


@setops.case("distinct")
def distinct_pairs(ctx: SyntheticContext):
    """DISTINCT (k1, k2) pairs."""

    def batcher():
        return ctx.bf.select("k1", "k2").distinct().collect()

    def duckdb():
        return ctx.con.sql("SELECT DISTINCT k1, k2 FROM fact").to_arrow_table()

    def polars():
        return ctx.pf.lazy().select("k1", "k2").unique().collect().to_arrow()

    return {"batcher": batcher, "duckdb": duckdb, "polars": polars}


@setops.case("union")
def union_all_and_distinct(ctx: SyntheticContext):
    """UNION ALL and UNION over two overlapping id subsets."""

    def batcher():
        a = ctx.bf.filter(col("k2") < 10).select("id")
        b = ctx.bf.filter(col("k1") < 10).select("id")
        n_all = a.union(b).count()
        n_dist = a.union(b, distinct=True).count()
        return pa.table({"n_all": pa.array([n_all]), "n_dist": pa.array([n_dist])})

    def duckdb():
        return ctx.con.sql(
            """SELECT
                 (SELECT COUNT(*) FROM (
                    SELECT id FROM fact WHERE k2 < 10
                    UNION ALL SELECT id FROM fact WHERE k1 < 10)) AS n_all,
                 (SELECT COUNT(*) FROM (
                    SELECT id FROM fact WHERE k2 < 10
                    UNION SELECT id FROM fact WHERE k1 < 10)) AS n_dist"""
        ).to_arrow_table()

    def polars():
        a = ctx.pf.lazy().filter(pl.col("k2") < 10).select("id")
        b = ctx.pf.lazy().filter(pl.col("k1") < 10).select("id")
        n_all = pl.concat([a, b]).collect().height
        n_dist = pl.concat([a, b]).unique().collect().height
        return pa.table({"n_all": pa.array([n_all]), "n_dist": pa.array([n_dist])})

    return {"batcher": batcher, "duckdb": duckdb, "polars": polars}


@setops.case("intersect")
def intersect_keys(ctx: SyntheticContext):
    """Distinct k1 present in both halves of the table."""

    def batcher():
        a = ctx.bf.filter(col("k2") < 25).select("k1").distinct()
        b = ctx.bf.filter(col("k2") >= 25).select("k1").distinct()
        n = a.intersect(b).count()
        return pa.table({"n": pa.array([n], type=pa.int64())})

    def duckdb():
        return ctx.con.sql(
            """SELECT COUNT(*) AS n FROM (
                 SELECT DISTINCT k1 FROM fact WHERE k2 < 25
                 INTERSECT SELECT DISTINCT k1 FROM fact WHERE k2 >= 25)"""
        ).to_arrow_table()

    def polars():
        a = ctx.pf.lazy().filter(pl.col("k2") < 25).select("k1").unique()
        b = ctx.pf.lazy().filter(pl.col("k2") >= 25).select("k1").unique()
        out = a.join(b, on="k1", how="semi").select(pl.len().alias("n"))
        return out.collect().to_arrow().cast(pa.schema([("n", pa.int64())]))

    return {"batcher": batcher, "duckdb": duckdb, "polars": polars}


@setops.case("except")
def except_keys(ctx: SyntheticContext):
    """Distinct k1 that never appear in the k2 < 25 half."""

    def batcher():
        a = ctx.bf.select("k1").distinct()
        b = ctx.bf.filter(col("k2") < 25).select("k1").distinct()
        n = a.except_(b).count()
        return pa.table({"n": pa.array([n], type=pa.int64())})

    def duckdb():
        return ctx.con.sql(
            """SELECT COUNT(*) AS n FROM (
                 SELECT DISTINCT k1 FROM fact
                 EXCEPT SELECT DISTINCT k1 FROM fact WHERE k2 < 25)"""
        ).to_arrow_table()

    def polars():
        a = ctx.pf.lazy().select("k1").unique()
        b = ctx.pf.lazy().filter(pl.col("k2") < 25).select("k1").unique()
        out = a.join(b, on="k1", how="anti").select(pl.len().alias("n"))
        return out.collect().to_arrow().cast(pa.schema([("n", pa.int64())]))

    return {"batcher": batcher, "duckdb": duckdb, "polars": polars}
