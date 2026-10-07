"""The join verbs that are not an equi-join: `join_where` (an inequality join) and `cross_join`.

`ops-joins` times the inequality join as SQL text; here it is `Dataset.join_where`, against
Polars' `join_where` and DuckDB's range join on its native storage. The bands are sixty
contiguous, non-overlapping $20 ranges of `p_retailprice` from $900 to $2,100 (TPC-H's spec
formula puts every part's price between $900.00 and $2,099.00), so each part matches exactly
one band and the result has one row per part. The cross join pairs `region` with `part`
(5 x 200,000 at sf1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from registry import EngineQueries, suite

if TYPE_CHECKING:
    from context import Context

joins = suite("ops-frame-joins", dataset="operators")

_BAND = 20.0


def _bands() -> pa.Table:
    """Sixty [lo, hi) price bands covering TPC-H's whole $900-$2,099 retail-price range."""
    lo = [900.0 + _BAND * i for i in range(60)]
    return pa.table({"band": list(range(60)), "lo": lo, "hi": [x + _BAND for x in lo]})


def _duck(tables: dict[str, pa.Table]):
    """A DuckDB connection holding `tables` in native storage, as the `duckdb` engine runs."""
    import duckdb

    from engines.duckdb import match_batcher_budget

    con = duckdb.connect()
    match_batcher_budget(con)
    for name, table in tables.items():
        con.register(f"{name}_arrow", table)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_arrow")
        con.unregister(f"{name}_arrow")
    return con


@joins.case("op-join-where")
def join_where(ctx: Context) -> EngineQueries:
    """Each part to its price band by `lo <= price < hi`: an inequality-only join."""
    part = ctx.table("part").select(["p_partkey", "p_retailprice"])
    bands = _bands()
    names = ctx.names()
    fns: EngineQueries = {}
    if "batcher" in names:
        import batcher as bt

        p_ds, b_ds = bt.from_arrow(part), bt.from_arrow(bands)
        price = bt.col("p_retailprice")
        fns["batcher"] = lambda: (
            p_ds.join_where(b_ds, price >= bt.col("lo"), price < bt.col("hi"))
            .group_by("band")
            .agg(n=bt.col("p_partkey").count(), rev=price.sum())
            .to_arrow()
        )
    if "polars" in names:
        import polars as pl

        p_pl, b_pl = pl.from_arrow(part).lazy(), pl.from_arrow(bands).lazy()
        price_pl = pl.col("p_retailprice")
        fns["polars"] = lambda: (
            p_pl.join_where(b_pl, price_pl >= pl.col("lo"), price_pl < pl.col("hi"))
            .group_by("band")
            .agg(pl.len().alias("n"), price_pl.sum().alias("rev"))
            .collect()
            .to_arrow()
        )
    if "duckdb" in names:
        con = _duck({"part": part, "bands": bands})
        sql = (
            "SELECT band, count(*) AS n, sum(p_retailprice) AS rev FROM part JOIN bands "
            "ON p_retailprice >= lo AND p_retailprice < hi GROUP BY band"
        )
        fns["duckdb"] = lambda: con.sql(sql).to_arrow_table()
    return fns


@joins.case("op-cross-join")
def cross_join(ctx: Context) -> EngineQueries:
    """Every (region, part) pair, then a count and a sum per region: `cross_join`."""
    region = ctx.table("region").select(["r_regionkey"])
    part = ctx.table("part").select(["p_partkey", "p_retailprice"])
    names = ctx.names()
    fns: EngineQueries = {}
    if "batcher" in names:
        import batcher as bt

        r_ds, p_ds = bt.from_arrow(region), bt.from_arrow(part)
        fns["batcher"] = lambda: (
            r_ds.cross_join(p_ds)
            .group_by("r_regionkey")
            .agg(n=bt.col("p_partkey").count(), rev=bt.col("p_retailprice").sum())
            .to_arrow()
        )
    if "polars" in names:
        import polars as pl

        r_pl, p_pl = pl.from_arrow(region).lazy(), pl.from_arrow(part).lazy()
        fns["polars"] = lambda: (
            r_pl.join(p_pl, how="cross")
            .group_by("r_regionkey")
            .agg(pl.len().alias("n"), pl.col("p_retailprice").sum().alias("rev"))
            .collect()
            .to_arrow()
        )
    if "duckdb" in names:
        con = _duck({"region": region, "part": part})
        sql = (
            "SELECT r_regionkey, count(*) AS n, sum(p_retailprice) AS rev "
            "FROM region CROSS JOIN part GROUP BY r_regionkey"
        )
        fns["duckdb"] = lambda: con.sql(sql).to_arrow_table()
    return fns
