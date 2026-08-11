"""The five reductions every report starts with, whole-table and per group.

`agg` on a Dataset collapses it to one row. The same expressions after a `group_by`
collapse each group instead. That symmetry is deliberate: one aggregate definition, two
scopes, and no separate "grouped" spelling to learn.

    python examples/aggregations/basic_reductions.py
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

    overall = lineitem.agg(
        lines=bt.count(),
        total_qty=col("l_quantity").sum(),
        avg_price=col("l_extendedprice").mean(),
        cheapest=col("l_extendedprice").min(),
        dearest=col("l_extendedprice").max(),
    ).to_pydict()
    print(overall)

    assert overall["lines"][0] == lineitem.count()
    assert overall["cheapest"][0] <= overall["avg_price"][0] <= overall["dearest"][0]

    # The same expressions, per ship mode.
    per_mode = (
        lineitem.group_by("l_shipmode")
        .agg(
            lines=bt.count(),
            total_qty=col("l_quantity").sum(),
            avg_price=col("l_extendedprice").mean(),
        )
        .sort("l_shipmode")
        .to_pydict()
    )
    print(per_mode["l_shipmode"], per_mode["lines"])

    # The groups partition the table, so counts and sums must reconcile exactly.
    assert sum(per_mode["lines"]) == overall["lines"][0]
    assert abs(sum(per_mode["total_qty"]) - overall["total_qty"][0]) < 1e-6


if __name__ == "__main__":
    main()
