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


@ordering.case("op-sort-limit")
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

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow)
