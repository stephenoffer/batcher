"""The `.list` namespace over each order's line quantities (~1.5M lists of 1-7 elements at sf1).

`op-explode` (one directory up) flattens these lists; a list-typed column is just as often
reduced in place, without exploding it, which is what these cases time: the per-list length,
sum and max, a membership test, and a sort-then-index and a distinct. The input is the same
`_lists` reshaping of real `lineitem` rows, built once outside the timed region.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from registry import suite

from ..projection import _lists
from ._engines import frame_case

if TYPE_CHECKING:
    from context import Context

lists = suite("ops-frame-lists", dataset="operators")


@lists.case("op-list-reduce")
def list_reduce(ctx: Context):
    """Total elements, total quantity and the largest per-list max: `list.len/sum/max`."""
    import batcher as bt

    def batcher(ds):
        qs = bt.col("qs")
        per = ds.select(ln=qs.list.len(), s=qs.list.sum(), mx=qs.list.max())
        return per.agg(
            elems=bt.col("ln").sum(), qty=bt.col("s").sum(), top=bt.col("mx").max()
        ).to_arrow()

    def polars(lf):
        import polars as pl

        qs = pl.col("qs")
        per = lf.select(
            qs.list.len().cast(pl.Int64).alias("ln"),
            qs.list.sum().alias("s"),
            qs.list.max().alias("mx"),
        )
        return per.select(
            pl.col("ln").sum().alias("elems"),
            pl.col("s").sum().alias("qty"),
            pl.col("mx").max().alias("top"),
        )

    def pyarrow(t):
        flat = pc.list_flatten(t["qs"])
        return pa.table(
            {
                "elems": [pc.sum(pc.list_value_length(t["qs"])).as_py()],
                "qty": [pc.sum(flat).as_py()],
                "top": [pc.max(flat).as_py()],
            }
        )

    sql = (
        "SELECT sum(ln) AS elems, sum(s) AS qty, max(mx) AS top FROM "
        "(SELECT len(qs) AS ln, list_sum(qs) AS s, list_max(qs) AS mx FROM {t})"
    )
    return frame_case(
        ctx, data=_lists(ctx), batcher=batcher, polars=polars, sql=sql, pyarrow=pyarrow
    )


@lists.case("op-list-contains")
def list_contains(ctx: Context):
    """Orders holding a line of exactly 50 units: `list.contains(50)` per list, then a count."""
    import batcher as bt

    def batcher(ds):
        return (
            ds.filter(bt.col("qs").list.contains(50)).agg(n=bt.col("l_orderkey").count()).to_arrow()
        )

    def polars(lf):
        import polars as pl

        return lf.filter(pl.col("qs").list.contains(50)).select(pl.len().alias("n"))

    sql = "SELECT count(*) AS n FROM {t} WHERE list_contains(qs, 50)"
    return frame_case(ctx, data=_lists(ctx), batcher=batcher, polars=polars, sql=sql)


@lists.case("op-list-sort-unique")
def list_sort_unique(ctx: Context):
    """Sum of each list's smallest element (sort, take the first) and of its distinct count."""
    import batcher as bt

    def batcher(ds):
        qs = bt.col("qs")
        per = ds.select(lo=qs.list.sort().list.get(0), nd=qs.list.unique().list.len())
        return per.agg(lo=bt.col("lo").sum(), nd=bt.col("nd").sum()).to_arrow()

    def polars(lf):
        import polars as pl

        qs = pl.col("qs")
        per = lf.select(
            qs.list.sort().list.get(0).alias("lo"),
            qs.list.unique().list.len().cast(pl.Int64).alias("nd"),
        )
        return per.select(pl.col("lo").sum(), pl.col("nd").sum())

    sql = (
        "SELECT sum(lo) AS lo, sum(nd) AS nd FROM "
        "(SELECT list_sort(qs)[1] AS lo, len(list_distinct(qs)) AS nd FROM {t})"
    )
    return frame_case(ctx, data=_lists(ctx), batcher=batcher, polars=polars, sql=sql)
