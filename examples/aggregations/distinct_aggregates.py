"""Counting distinct values inside a group, and the cost of doing it exactly.

An exact distinct count per group needs the distinct set of every group in memory at once.
That is the aggregate most likely to be the reason a query does not fit, and the sketch
version is the one that does — at a bounded error.

    python examples/aggregations/distinct_aggregates.py
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

    per_mode = (
        lineitem.group_by("l_shipmode")
        .agg(
            lines=bt.count(),
            parts=col("l_partkey").n_unique(),
            approx_parts=bt.approx_n_unique(col("l_partkey")),
            suppliers=col("l_suppkey").n_unique(),
        )
        .sort("l_shipmode")
        .to_pydict()
    )

    for index, mode in enumerate(per_mode["l_shipmode"]):
        exact = per_mode["parts"][index]
        approx = per_mode["approx_parts"][index]
        print(
            f"{mode:<9} lines={per_mode['lines'][index]:>6} parts={exact:>6} "
            f"approx={approx:>6} error={abs(approx - exact) / exact:.4%}"
        )

    # A distinct count never exceeds the row count of its group.
    assert all(
        distinct <= lines
        for distinct, lines in zip(per_mode["parts"], per_mode["lines"], strict=True)
    )

    # The sketch stays within a few percent.
    assert all(
        abs(approx - exact) / exact < 0.05
        for exact, approx in zip(per_mode["parts"], per_mode["approx_parts"], strict=True)
    )

    # Distinct counts do not add up across groups: the same part can ship by several
    # modes, so summing the per-group counts overstates the global one.
    global_parts = lineitem.n_unique("l_partkey")
    summed = sum(per_mode["parts"])
    print(f"global distinct parts {global_parts}, sum of per-group {summed}")
    assert summed > global_parts


if __name__ == "__main__":
    main()
