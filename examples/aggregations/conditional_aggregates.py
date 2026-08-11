"""Counting and summing subsets without a second query.

Every "and also, how many of those were X" is a masked aggregate over the pass you are
already making. Two scans become one, and the two numbers are guaranteed to describe the
same snapshot of the data — which two separate queries are not.

    python examples/aggregations/conditional_aggregates.py
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

    returned = col("l_returnflag") == "R"
    revenue = col("l_extendedprice") * (1 - col("l_discount"))

    summary = (
        lineitem.group_by("l_shipmode")
        .agg(
            lines=bt.count(),
            returned_lines=bt.count_if(returned),
            returned_revenue=bt.when(returned).then(revenue).otherwise(0.0).sum(),
            total_revenue=revenue.sum(),
            any_returned=bt.bool_or(returned),
            all_returned=bt.bool_and(returned),
        )
        .with_columns(return_rate=col("returned_lines") / col("lines"))
        .sort("l_shipmode")
        .to_pydict()
    )
    print(summary["l_shipmode"], [round(rate, 4) for rate in summary["return_rate"]])

    # A masked count is bounded by the unmasked one, and the rate follows.
    for index in range(len(summary["l_shipmode"])):
        assert summary["returned_lines"][index] <= summary["lines"][index]
        assert 0.0 <= summary["return_rate"][index] <= 1.0
        assert summary["returned_revenue"][index] <= summary["total_revenue"][index]
        # `bool_or` is true exactly when the masked count is non-zero.
        assert summary["any_returned"][index] == (summary["returned_lines"][index] > 0)
        assert summary["all_returned"][index] == (
            summary["returned_lines"][index] == summary["lines"][index]
        )


if __name__ == "__main__":
    main()
