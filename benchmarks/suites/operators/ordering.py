"""Operator-mix: sort + limit (top-N) over TPC-H ``lineitem`` — a pipeline breaker.

A deterministic tie-break (l_orderkey, l_linenumber) keeps the top-N a single answer
across engines, so the correctness gate compares a well-defined result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from registry import suite

from .base import sql_fanout, with_native

if TYPE_CHECKING:
    from context import Context

ordering = suite("ops-ordering", dataset="operators")


@ordering.case("op-sort-limit", ordered_by="l_extendedprice DESC, l_orderkey, l_linenumber")
def sort_limit(ctx: Context):
    """Top-100 line items by extended price, tie-broken for a deterministic result."""
    sql = (
        "SELECT l_orderkey, l_linenumber, l_extendedprice FROM lineitem "
        "ORDER BY l_extendedprice DESC, l_orderkey, l_linenumber LIMIT 100"
    )

    def pyarrow(t: pa.Table) -> pa.Table:
        cols = t.select(["l_orderkey", "l_linenumber", "l_extendedprice"])
        ordered = cols.sort_by(
            [
                ("l_extendedprice", "descending"),
                ("l_orderkey", "ascending"),
                ("l_linenumber", "ascending"),
            ]
        )
        return ordered.slice(0, 100)

    def ray(rd) -> pa.Table:
        cols = rd.select_columns(["l_orderkey", "l_linenumber", "l_extendedprice"])
        ordered = cols.sort(
            ["l_extendedprice", "l_orderkey", "l_linenumber"],
            descending=[True, False, False],
        )
        return pa.Table.from_pandas(ordered.limit(100).to_pandas(), preserve_index=False)

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow, ray=ray)


@ordering.case("op-sort-string", ordered_by="l_comment")
def sort_string(ctx: Context):
    """Full sort on one high-cardinality string key — the shape with no coverage until now.

    Every other ordering case here sorts on a *fixed-width* key, or on several keys at
    once. Both route away from the single string key path (`stable_sort_indices_str` and
    the string range-partitioner), which is measured at ~0.30x of DuckDB on 10M rows and
    was therefore losing badly with nothing tracking it.

    No tie-break, deliberately. The harness compares row multisets, so a full sort's
    result is well defined even with ties, and adding a second key would turn this into
    the row-encoded multi-key sort — a different path, and the one already covered above.
    """
    sql = "SELECT l_comment FROM lineitem ORDER BY l_comment"

    def pyarrow(t: pa.Table) -> pa.Table:
        return t.select(["l_comment"]).sort_by([("l_comment", "ascending")])

    def ray(rd) -> pa.Table:
        ordered = rd.select_columns(["l_comment"]).sort(["l_comment"])
        return pa.Table.from_pandas(ordered.to_pandas(), preserve_index=False)

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow, ray=ray)


@ordering.case("op-sort-string-lowcard", ordered_by="l_shipmode")
def sort_string_lowcard(ctx: Context):
    """Full sort on a low-cardinality string key, where range-partitioning has little to cut on.

    `l_shipmode` has seven distinct values, so sampled quantile boundaries cannot separate
    the ranges the parallel sample-sort wants and the work piles into a few of them. That
    is the opposite end of the same operator from `op-sort-string`, and the two fail in
    different ways, so tracking only one would hide the other.
    """
    sql = "SELECT l_shipmode FROM lineitem ORDER BY l_shipmode"

    def pyarrow(t: pa.Table) -> pa.Table:
        return t.select(["l_shipmode"]).sort_by([("l_shipmode", "ascending")])

    def ray(rd) -> pa.Table:
        ordered = rd.select_columns(["l_shipmode"]).sort(["l_shipmode"])
        return pa.Table.from_pandas(ordered.to_pandas(), preserve_index=False)

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow, ray=ray)


@ordering.case("op-sort-string-limit", ordered_by="l_comment, l_orderkey, l_linenumber")
def sort_string_limit(ctx: Context):
    """Top-100 by a string key, tie-broken so the surviving rows are a single answer.

    A `LIMIT` does need the tie-break the full sorts above do not: with ties straddling
    the k-th row, *which* rows survive is otherwise engine-dependent and the correctness
    gate would be comparing an ambiguous result rather than a wrong one.
    """
    sql = (
        "SELECT l_comment, l_orderkey, l_linenumber FROM lineitem "
        "ORDER BY l_comment, l_orderkey, l_linenumber LIMIT 100"
    )

    def pyarrow(t: pa.Table) -> pa.Table:
        cols = t.select(["l_comment", "l_orderkey", "l_linenumber"])
        ordered = cols.sort_by(
            [
                ("l_comment", "ascending"),
                ("l_orderkey", "ascending"),
                ("l_linenumber", "ascending"),
            ]
        )
        return ordered.slice(0, 100)

    def ray(rd) -> pa.Table:
        cols = rd.select_columns(["l_comment", "l_orderkey", "l_linenumber"])
        ordered = cols.sort(["l_comment", "l_orderkey", "l_linenumber"])
        return pa.Table.from_pandas(ordered.limit(100).to_pandas(), preserve_index=False)

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow, ray=ray)
