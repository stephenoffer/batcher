"""String-function benchmarks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pyarrow as pa

from batcher import col, count
from registry import suite

if TYPE_CHECKING:
    from contexts import SyntheticContext

strings = suite("strings", dataset="synthetic")

NEEDLE = "eta"  # matches "beta", "eta", "theta", "zeta"


@strings.case("string-filter")
def contains_filter(ctx: SyntheticContext):
    """COUNT(*) WHERE category contains a substring."""

    def batcher():
        n = ctx.bf.filter(col("category").str.contains(NEEDLE)).count()
        return pa.table({"n": pa.array([n], type=pa.int64())})

    def duckdb():
        return ctx.con.sql(
            f"SELECT COUNT(*) AS n FROM fact WHERE category LIKE '%{NEEDLE}%'"
        ).to_arrow_table()

    def polars():
        out = (
            ctx.pf.lazy()
            .filter(pl.col("category").str.contains(NEEDLE, literal=True))
            .select(pl.len().alias("n"))
            .collect()
        )
        return out.to_arrow().cast(pa.schema([("n", pa.int64())]))

    return {"batcher": batcher, "duckdb": duckdb, "polars": polars}


@strings.case("string-upper")
def group_by_upper(ctx: SyntheticContext):
    """GROUP BY upper(category)."""

    def batcher():
        return ctx.bf.group_by(cu=col("category").str.upper()).agg(n=count()).collect()

    def duckdb():
        return ctx.con.sql(
            "SELECT upper(category) AS cu, COUNT(*) AS n FROM fact GROUP BY upper(category)"
        ).to_arrow_table()

    def polars():
        return (
            ctx.pf.lazy()
            .group_by(pl.col("category").str.to_uppercase().alias("cu"))
            .agg(pl.len().alias("n"))
            .collect()
            .to_arrow()
        )

    return {"batcher": batcher, "duckdb": duckdb, "polars": polars}
