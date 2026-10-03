"""Operator-mix: the non-breaking row shapes over TPC-H ``lineitem`` -- project, explode, unpivot.

Filter + projection is one SQL string fanned out to every engine. ``Dataset.explode``
(``UNNEST``, Polars ``explode``) and ``Dataset.unpivot`` (``UNPIVOT``, Polars ``unpivot``) are
spelled differently on each engine, so they are a native callable per engine over inputs built
once outside the timed region. Each result is aggregated to a few rows, so the correctness gate
compares the whole answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from registry import EngineQueries, suite

from .base import ray_to_arrow, sql_fanout, with_native

if TYPE_CHECKING:
    from context import Context

projection = suite("ops-projection", dataset="operators")


@projection.case("op-filter-project")
def filter_project(ctx: Context):
    """Project a derived column over a filtered scan — no pipeline breaker."""
    sql = "SELECT l_orderkey, l_extendedprice * 2 AS p2 FROM lineitem WHERE l_extendedprice > 50000"

    def pyarrow(t: pa.Table) -> pa.Table:
        f = t.filter(pc.greater(t["l_extendedprice"], 50000))
        return pa.table({"l_orderkey": f["l_orderkey"], "p2": pc.multiply(f["l_extendedprice"], 2)})

    def ray(rd) -> pa.Table:
        from ray.data.expressions import col

        def project(b: pa.Table) -> pa.Table:
            return pa.table(
                {"l_orderkey": b["l_orderkey"], "p2": pc.multiply(b["l_extendedprice"], 2)}
            )

        out = rd.filter(expr=col("l_extendedprice") > 50000).map_batches(
            project, batch_format="pyarrow"
        )
        return ray_to_arrow(out)

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow, ray=ray)


def _lists(ctx: Context) -> pa.Table:
    """Each order's line quantities as one list per order (~4 elements each)."""
    line = ctx.table("lineitem").select(["l_orderkey", "l_quantity"])
    lists = line.group_by("l_orderkey").aggregate([("l_quantity", "list")])
    return lists.rename_columns(["l_orderkey", "qs"])


@projection.case("op-explode")
def explode(ctx: Context) -> EngineQueries:
    """One row per list element (`UNNEST`), then a count and a sum over the elements."""
    lists = _lists(ctx)
    names = ctx.names()
    fns: EngineQueries = {}
    if "batcher" in names:
        import batcher as bt

        ds = bt.from_arrow(lists)
        fns["batcher"] = lambda: (
            ds.explode("qs").agg(n=bt.col("qs").count(), total=bt.col("qs").sum()).to_arrow()
        )
    if "duckdb" in names:
        import duckdb

        con = duckdb.connect()
        con.register("t", lists)
        sql = "SELECT count(*) AS n, sum(q) AS total FROM (SELECT unnest(qs) AS q FROM t)"
        fns["duckdb"] = lambda: con.sql(sql).to_arrow_table()
    if "polars" in names:
        import polars as pl

        df = pl.from_arrow(lists)
        fns["polars"] = lambda: (
            df.lazy()
            .explode("qs")
            .select(pl.len().alias("n"), pl.col("qs").sum().alias("total"))
            .collect()
            .to_arrow()
        )
    return fns


_MEASURES = ["l_quantity", "l_extendedprice", "l_discount", "l_tax"]


def _wide(ctx: Context) -> pa.Table:
    """Four numeric measures per line, all `float64` so they can share one value column."""
    line = ctx.table("lineitem")
    cols = {"l_orderkey": line["l_orderkey"]}
    cols.update({m: pc.cast(line[m], pa.float64()) for m in _MEASURES})
    return pa.table(cols)


@projection.case("op-unpivot")
def unpivot(ctx: Context) -> EngineQueries:
    """Wide to long over four measures (`UNPIVOT`), then a sum and a count per measure."""
    wide = _wide(ctx)
    names = ctx.names()
    fns: EngineQueries = {}
    if "batcher" in names:
        import batcher as bt

        ds = bt.from_arrow(wide)
        fns["batcher"] = lambda: (
            ds.unpivot(index="l_orderkey", on=_MEASURES)
            .group_by("variable")
            .agg(s=bt.col("value").sum(), n=bt.col("value").count())
            .to_arrow()
        )
    if "duckdb" in names:
        import duckdb

        con = duckdb.connect()
        con.register("t", wide)
        sql = (
            "SELECT variable, sum(value) AS s, count(*) AS n FROM "
            f"(UNPIVOT t ON {', '.join(_MEASURES)} INTO NAME variable VALUE value) "
            "GROUP BY variable"
        )
        fns["duckdb"] = lambda: con.sql(sql).to_arrow_table()
    if "polars" in names:
        import polars as pl

        df = pl.from_arrow(wide)
        fns["polars"] = lambda: (
            df.lazy()
            .unpivot(index="l_orderkey", on=_MEASURES, variable_name="variable", value_name="value")
            .group_by("variable")
            .agg(pl.col("value").sum().alias("s"), pl.len().alias("n"))
            .collect()
            .to_arrow()
        )
    return fns
