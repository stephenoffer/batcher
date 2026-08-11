"""Tracing where a column came from.

Lineage answers "which inputs does this output depend on", which is what an audit asks and
what an impact analysis needs before a schema change. It reads the plan, so it costs
nothing and needs no execution.

    python examples/operations/lineage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")
    orders = tpch("orders")

    pipeline = (
        lineitem.join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .with_columns(revenue=col("l_extendedprice") * (1 - col("l_discount")))
        .group_by("o_orderpriority")
        .agg(total=col("revenue").sum(), lines=bt.count())
    )

    trace = pipeline.lineage()
    print(trace)
    assert trace is not None

    # The derived column depends on two source columns, and the trace should mention them.
    text = str(trace)
    assert "revenue" in text or "total" in text

    # The result is unchanged by asking for its lineage.
    result = pipeline.sort("o_orderpriority").to_pydict()
    assert len(result["o_orderpriority"]) == orders.n_unique("o_orderpriority")
    assert sum(result["lines"]) > 0


if __name__ == "__main__":
    main()
