"""Window-function benchmarks: ranking, partition aggregates, and ROWS frames."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from batcher import col
from registry import suite

if TYPE_CHECKING:
    from contexts import SyntheticContext

window = suite("window", dataset="synthetic")


@window.case("window(rn+psum)")
def row_number_and_partition_sum(ctx: SyntheticContext):
    """row_number() over (k2 ORDER BY price DESC, id) + whole-partition SUM, top-3 per partition."""

    def batcher():
        # rn is ordered; tot is a whole-partition sum (separate window, no ORDER
        # BY) to match DuckDB/Polars; an ordered SUM window would be a running total.
        w = ctx.bf.window(
            partition_by=["k2"],
            order_by=[("price", True), ("id", False)],
            functions={"rn": "row_number"},
        ).window(
            partition_by=["k2"],
            functions={"tot": ("sum", "price")},
        )
        return w.filter(col("rn") <= 3).select("k2", "id", "rn", "tot").collect()

    def duckdb():
        return ctx.con.sql(
            """SELECT k2, id, rn, tot FROM (
                   SELECT k2, id,
                          row_number() OVER (PARTITION BY k2 ORDER BY price DESC, id ASC) AS rn,
                          SUM(price) OVER (PARTITION BY k2) AS tot
                   FROM fact
               ) WHERE rn <= 3"""
        ).to_arrow_table()

    def polars():
        return (
            ctx.pf.lazy()
            .with_columns(
                pl.struct([(-pl.col("price")).alias("_np"), pl.col("id")])
                .rank("ordinal")
                .over("k2")
                .alias("rn"),
                pl.col("price").sum().over("k2").alias("tot"),
            )
            .filter(pl.col("rn") <= 3)
            .select("k2", "id", "rn", "tot")
            .collect()
            .to_arrow()
        )

    return {"batcher": batcher, "duckdb": duckdb, "polars": polars}


@window.case("window-frame")
def moving_sum_frame(ctx: SyntheticContext):
    """Trailing 3-row moving SUM with an explicit ROWS frame (batcher vs duckdb).

    Polars does not express ROWS frames cleanly inside over(), so it is marked n/a.
    The first 3 rows per partition are kept so the comparison stays small.
    """

    def batcher():
        w = ctx.bf.window(
            partition_by=["k2"],
            order_by=[("id", False)],
            functions={"rn": "row_number"},
        ).window(
            partition_by=["k2"],
            order_by=[("id", False)],
            functions={"mov": ("sum", "price")},
            frame=(-2, 0),
        )
        return w.filter(col("rn") <= 3).select("k2", "id", "rn", "mov").collect()

    def duckdb():
        return ctx.con.sql(
            """SELECT k2, id, rn, mov FROM (
                   SELECT k2, id,
                          row_number() OVER (PARTITION BY k2 ORDER BY id) AS rn,
                          SUM(price) OVER (PARTITION BY k2 ORDER BY id
                                          ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS mov
                   FROM fact
               ) WHERE rn <= 3"""
        ).to_arrow_table()

    return {"batcher": batcher, "duckdb": duckdb, "polars": None}
