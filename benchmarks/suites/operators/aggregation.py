"""Operator-mix: aggregation over TPC-H ``lineitem`` (group-by, global, filtered count).

SQL engines run the one SQL string; PyArrow (Acero ``group_by``) and Ray Data get
native implementations so the two non-SQL engines compete here too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from registry import suite

from .base import sql_fanout, with_native

if TYPE_CHECKING:
    from context import Context

agg = suite("ops-aggregation", dataset="operators")


@agg.case("op-groupby-sum")
def groupby_sum(ctx: Context):
    """GROUP BY l_returnflag, SUM(l_quantity) — the single-key all-to-all aggregate."""
    sql = "SELECT l_returnflag, SUM(l_quantity) AS s FROM lineitem GROUP BY l_returnflag"

    def pyarrow(t: pa.Table) -> pa.Table:
        a = t.group_by("l_returnflag").aggregate([("l_quantity", "sum")])
        return pa.table({"l_returnflag": a["l_returnflag"], "s": a["l_quantity_sum"]})

    def ray(rd) -> pa.Table:
        df = rd.groupby("l_returnflag").sum("l_quantity").to_pandas()
        return pa.Table.from_pandas(
            df.rename(columns={"sum(l_quantity)": "s"}), preserve_index=False
        )

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow, ray=ray)


@agg.case("op-groupby-2key")
def groupby_2key(ctx: Context):
    """GROUP BY l_returnflag, l_linestatus with SUM and COUNT — a two-key aggregate."""
    sql = (
        "SELECT l_returnflag, l_linestatus, SUM(l_quantity) AS s, COUNT(*) AS n "
        "FROM lineitem GROUP BY l_returnflag, l_linestatus"
    )

    def pyarrow(t: pa.Table) -> pa.Table:
        a = t.group_by(["l_returnflag", "l_linestatus"]).aggregate(
            [("l_quantity", "sum"), ("l_quantity", "count")]
        )
        return pa.table(
            {
                "l_returnflag": a["l_returnflag"],
                "l_linestatus": a["l_linestatus"],
                "s": a["l_quantity_sum"],
                "n": a["l_quantity_count"],
            }
        )

    def ray(rd) -> pa.Table:
        from ray.data.aggregate import Count, Sum

        g = rd.groupby(["l_returnflag", "l_linestatus"]).aggregate(Sum("l_quantity"), Count())
        df = g.to_pandas().rename(columns={"sum(l_quantity)": "s", "count()": "n"})
        return pa.Table.from_pandas(df, preserve_index=False)

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow, ray=ray)


@agg.case("op-groupby-multi-int")
def groupby_multi_int(ctx: Context):
    """GROUP BY on four integer keys — the shape TPC-DS spends its aggregate time in.

    The suite already covers a single key (`op-groupby-sum`) and a two-key *byte* key
    (`op-groupby-2key`), and both take specialized paths: a dense direct map and a packed
    one-byte-per-column integer. Neither reaches the composite **integer** hash path, which
    is what every level of a `ROLLUP` over an integer dimension pays and what
    `competitor_technique_review.md` item 23 measured as the residue of the TPC-DS board.
    A bottleneck with no case in the suite is how item 9's string sort stayed invisible.

    The keys are chosen so their value ranges multiply past the dense-map budget, which is
    what forces the hash path rather than the mixed-radix direct map — `l_partkey` alone
    exceeds it. `l_linenumber` and `l_quantity` are narrow, so the pair in front of them
    does not make the composite trivially unique either.
    """
    sql = (
        "SELECT l_linenumber, l_quantity, l_suppkey, l_partkey, "
        "SUM(l_extendedprice) AS s, COUNT(*) AS n "
        "FROM lineitem GROUP BY l_linenumber, l_quantity, l_suppkey, l_partkey"
    )
    keys = ["l_linenumber", "l_quantity", "l_suppkey", "l_partkey"]

    def pyarrow(t: pa.Table) -> pa.Table:
        a = t.group_by(keys).aggregate([("l_extendedprice", "sum"), ("l_extendedprice", "count")])
        cols = {k: a[k] for k in keys}
        cols["s"] = a["l_extendedprice_sum"]
        cols["n"] = a["l_extendedprice_count"]
        return pa.table(cols)

    def ray(rd) -> pa.Table:
        from ray.data.aggregate import Count, Sum

        g = rd.groupby(keys).aggregate(Sum("l_extendedprice"), Count())
        df = g.to_pandas().rename(columns={"sum(l_extendedprice)": "s", "count()": "n"})
        return pa.Table.from_pandas(df, preserve_index=False)

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow, ray=ray)


@agg.case("op-global-sum")
def global_sum(ctx: Context):
    """Global SUM(l_extendedprice) — a single mergeable reduction.

    **Batcher does not execute this case, and its number must not be read as if it did.**
    An unfiltered aggregate over a bare column of an immutable in-memory relation is
    answered from a recorded column statistic, and that statistic is computed by the
    *first* run and read back by every later one — so the harness's best-of-5 measures a
    memo lookup (0.1 ms) where DuckDB scans the column (1.3 ms). Computing the sum takes
    Batcher ~5 ms, which is 3.8x slower than DuckDB rather than 13x faster.

    The shortcut is a real capability and a real user-visible latency, so it stays; what
    must not happen is quoting 0.10x as an execution ratio. `op-filter-count` below and
    ClickBench q01-q05 are the same shape; `BENCHMARK_RESULTS.md` carries the suite
    geomeans with and without them.
    """
    sql = "SELECT SUM(l_extendedprice) AS s FROM lineitem"

    def pyarrow(t: pa.Table) -> pa.Table:
        return pa.table({"s": pa.array([pc.sum(t["l_extendedprice"]).as_py()])})

    def ray(rd) -> pa.Table:
        return pa.table({"s": pa.array([rd.sum("l_extendedprice")])})

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow, ray=ray)


@agg.case("op-filter-count")
def filter_count(ctx: Context):
    """COUNT(*) WHERE l_quantity > 25 — a streaming filter reduced to a scalar.

    Answered from a memoized statistic on every run, exactly as `op-global-sum` above is,
    and with the same caveat: 0.22 ms measured against ~6.1 ms to actually count.
    """
    sql = "SELECT COUNT(*) AS n FROM lineitem WHERE l_quantity > 25"

    def pyarrow(t: pa.Table) -> pa.Table:
        n = t.filter(pc.greater(t["l_quantity"], 25)).num_rows
        return pa.table({"n": pa.array([n], type=pa.int64())})

    def ray(rd) -> pa.Table:
        n = rd.filter(expr="l_quantity > 25").count()
        return pa.table({"n": pa.array([n], type=pa.int64())})

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow, ray=ray)
