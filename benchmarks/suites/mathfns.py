"""Math-function benchmarks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from batcher import col
from registry import suite

if TYPE_CHECKING:
    from contexts import SyntheticContext

mathfns = suite("math", dataset="synthetic")


@mathfns.case("math-agg")
def integer_exact_sums(ctx: SyntheticContext):
    """Integer-exact SUM(abs(x)) and SUM(floor(price)) (no float-order sensitivity)."""

    def batcher():
        return ctx.bf.group_by().agg(a=col("x").abs().sum(), b=col("price").floor().sum()).collect()

    def duckdb():
        return ctx.con.sql(
            "SELECT SUM(abs(x)) AS a, SUM(floor(price)) AS b FROM fact"
        ).to_arrow_table()

    def polars():
        return (
            ctx.pf.lazy()
            .select(
                pl.col("x").abs().sum().alias("a"),
                pl.col("price").floor().sum().alias("b"),
            )
            .collect()
            .to_arrow()
        )

    return {"batcher": batcher, "duckdb": duckdb, "polars": polars}
