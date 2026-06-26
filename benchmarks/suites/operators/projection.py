"""Operator-mix: filter + projection over TPC-H ``lineitem`` (a streaming scan)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from registry import suite

from .base import sql_fanout, with_native

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

    return with_native(ctx, sql_fanout(ctx, sql), pyarrow=pyarrow)
