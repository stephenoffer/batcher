"""Date-function benchmarks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from batcher import col, count
from registry import suite

if TYPE_CHECKING:
    from contexts import SyntheticContext

dates = suite("dates", dataset="synthetic")


@dates.case("date-year")
def group_by_year(ctx: SyntheticContext):
    """GROUP BY EXTRACT(year FROM ts)."""

    def batcher():
        return ctx.bf.group_by(y=col("ts").dt.year()).agg(n=count()).collect()

    def duckdb():
        return ctx.con.sql(
            "SELECT EXTRACT(year FROM ts) AS y, COUNT(*) AS n FROM fact GROUP BY 1"
        ).to_arrow_table()

    def polars():
        return (
            ctx.pf.lazy()
            .group_by(pl.col("ts").dt.year().alias("y"))
            .agg(pl.len().alias("n"))
            .collect()
            .to_arrow()
        )

    return {"batcher": batcher, "duckdb": duckdb, "polars": polars}
