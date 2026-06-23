"""Ordering benchmarks: sort and top-N."""

from __future__ import annotations

from typing import TYPE_CHECKING

from registry import suite

if TYPE_CHECKING:
    from contexts import SyntheticContext

ordering = suite("ordering", dataset="synthetic")

TOPN = 20


@ordering.case("sort+limit(top20)")
def sort_limit(ctx: SyntheticContext):
    """ORDER BY price DESC, id ASC LIMIT 20 (id tie-break makes the top-N deterministic)."""

    def batcher():
        return (
            ctx.bf.sort("price", "id", descending=[True, False])
            .limit(TOPN)
            .select("id", "price")
            .collect()
        )

    def duckdb():
        return ctx.con.sql(
            "SELECT id, price FROM fact ORDER BY price DESC, id ASC LIMIT 20"
        ).to_arrow_table()

    def polars():
        return (
            ctx.pf.lazy()
            .sort(["price", "id"], descending=[True, False])
            .limit(TOPN)
            .select("id", "price")
            .collect()
            .to_arrow()
        )

    return {"batcher": batcher, "duckdb": duckdb, "polars": polars}
