"""Measuring where a query spends its time.

`profile` runs the query and reports per-operator timings, which is the only way to know
whether the join or the scan is the problem. Guessing from the plan shape is how people
end up optimizing the cheap half.

    python examples/operations/profiling_a_query.py
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

    query = (
        lineitem.join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .group_by("o_orderpriority")
        .agg(revenue=(col("l_extendedprice") * (1 - col("l_discount"))).sum(), lines=bt.count())
        .sort("o_orderpriority")
    )

    report = query.profile()
    print(report)
    assert report is not None

    # The query still returns its result; profiling does not change semantics.
    result = query.to_pydict()
    print(result["o_orderpriority"], result["lines"])
    assert len(result["o_orderpriority"]) == orders.n_unique("o_orderpriority")
    assert sum(result["lines"]) > 0

    # `meta` carries the metadata the executor recorded, which is what the optimizer
    # reads back on the next run.
    info = query.meta
    print("metadata:", type(info).__name__)
    assert info is not None


if __name__ == "__main__":
    main()
