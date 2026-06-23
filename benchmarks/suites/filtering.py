"""Filter benchmarks: row selection by predicate."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pyarrow as pa

from batcher import col
from registry import suite

if TYPE_CHECKING:
    from contexts import SyntheticContext

filtering = suite("filtering", dataset="synthetic")


@filtering.case("filter+count")
def filter_count(ctx: SyntheticContext):
    """COUNT(*) WHERE x > 0, aggregated to a scalar so the comparison stays tiny."""

    def batcher():
        n = ctx.bf.filter(col("x") > 0).count()
        return pa.table({"n": pa.array([n], type=pa.int64())})

    def duckdb():
        return ctx.con.sql("SELECT COUNT(*) AS n FROM fact WHERE x > 0").to_arrow_table()

    def polars():
        out = ctx.pf.lazy().filter(pl.col("x") > 0).select(pl.len().alias("n")).collect()
        return out.to_arrow().cast(pa.schema([("n", pa.int64())]))

    return {"batcher": batcher, "duckdb": duckdb, "polars": polars}
