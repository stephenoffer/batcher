"""Reshaping and ranking verbs a DataFrame user calls by name: `pivot`, `top_k`, `value_counts`.

`op-unpivot` (one directory up) covers long-to-wide's inverse. `pivot` turns `lineitem` wide on
its seven ship modes; `top_k` keeps the ten most expensive lines; `value_counts` is the
one-call frequency table. The top-k keys break every tie (`l_orderkey, l_linenumber` is
`lineitem`'s key), so the ten rows are one well-defined set; the case asks for the set, not
an order, because neither `top_k` nor Polars' promises one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from registry import suite

from ._engines import frame_case

if TYPE_CHECKING:
    from context import Context

reshape = suite("ops-frame-reshape", dataset="operators")

_MODES = ["AIR", "FOB", "MAIL", "RAIL", "REG AIR", "SHIP", "TRUCK"]


@reshape.case("op-pivot")
def pivot(ctx: Context):
    """Quantity per return flag (rows) by ship mode (seven columns): `pivot(... "sum")`."""

    def batcher(ds):
        wide = ds.pivot(index="l_returnflag", on="l_shipmode", values="l_quantity", aggregate="sum")
        return wide.select("l_returnflag", *_MODES).to_arrow()

    def polars(lf):
        wide = lf.collect().pivot(
            on="l_shipmode", index="l_returnflag", values="l_quantity", aggregate_function="sum"
        )
        return wide.select("l_returnflag", *_MODES)

    modes = ", ".join(f"'{m}'" for m in _MODES)
    sql = (
        f"PIVOT (SELECT l_returnflag, l_shipmode, l_quantity FROM {{t}}) ON l_shipmode "
        f"IN ({modes}) USING sum(l_quantity) GROUP BY l_returnflag"
    )
    return frame_case(ctx, batcher=batcher, polars=polars, sql=sql)


_TOP_KEYS = ["l_extendedprice", "l_orderkey", "l_linenumber"]


@reshape.case("op-top-k")
def top_k(ctx: Context):
    """The ten most expensive lines, all 16 columns: `top_k` over the 6M-row table."""

    def batcher(ds):
        return ds.top_k(10, by=_TOP_KEYS).to_arrow()

    def polars(lf):
        return lf.top_k(10, by=_TOP_KEYS)

    def pyarrow(t):
        idx = pc.select_k_unstable(t, k=10, sort_keys=[(k, "descending") for k in _TOP_KEYS])
        return t.take(idx)

    # The ORDER BY picks the ten; the outer SELECT says the set is the answer, not an order,
    # which is all `top_k` (and Polars' `top_k`) promise.
    sql = (
        "SELECT * FROM (SELECT * FROM {t} ORDER BY l_extendedprice DESC, l_orderkey DESC, "
        "l_linenumber DESC LIMIT 10) AS top"
    )
    return frame_case(ctx, batcher=batcher, polars=polars, sql=sql, pyarrow=pyarrow)


@reshape.case("op-value-counts")
def value_counts(ctx: Context):
    """Frequency of each `l_partkey` (200k distinct at sf1): the one-call `value_counts`."""

    def batcher(ds):
        return ds.value_counts("l_partkey", name="n", sort=False).to_arrow()

    def polars(lf):
        return lf.collect().get_column("l_partkey").value_counts(name="n")

    def pyarrow(t):
        counts = pc.value_counts(t["l_partkey"])
        return pa.table({"l_partkey": counts.field("values"), "n": counts.field("counts")})

    sql = "SELECT l_partkey, count(*) AS n FROM {t} GROUP BY l_partkey"
    return frame_case(ctx, batcher=batcher, polars=polars, sql=sql, pyarrow=pyarrow)
