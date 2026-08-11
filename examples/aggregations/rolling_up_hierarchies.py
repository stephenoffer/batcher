"""Rolling a fine aggregate up to a coarse one without re-scanning.

An aggregate at the finest grain can be summed up to any coarser grain, as long as the
aggregate is additive. Counts and sums are; averages and distinct counts are not, and
trying to roll up an average is the most common way a summary quietly goes wrong.

    python examples/aggregations/rolling_up_hierarchies.py
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

    fine = lineitem.group_by("l_shipmode", "l_returnflag").agg(
        lines=bt.count(),
        qty=col("l_quantity").sum(),
        mean_qty=col("l_quantity").mean(),
    )

    # Additive aggregates roll up correctly.
    rolled = (
        fine.group_by("l_shipmode")
        .agg(lines=col("lines").sum(), qty=col("qty").sum())
        .sort("l_shipmode")
        .to_pydict()
    )
    direct = (
        lineitem.group_by("l_shipmode")
        .agg(lines=bt.count(), qty=col("l_quantity").sum())
        .sort("l_shipmode")
        .to_pydict()
    )
    assert rolled["lines"] == direct["lines"]
    assert all(abs(a - b) < 1e-6 for a, b in zip(rolled["qty"], direct["qty"], strict=True))
    print("additive roll-up matches:", rolled["lines"])

    # A mean does not: averaging the averages ignores the group sizes.
    naive = (
        fine.group_by("l_shipmode")
        .agg(mean_qty=col("mean_qty").mean())
        .sort("l_shipmode")
        .to_pydict()
    )
    true_mean = (
        lineitem.group_by("l_shipmode")
        .agg(mean_qty=col("l_quantity").mean())
        .sort("l_shipmode")
        .to_pydict()
    )
    differences = [
        abs(a - b) for a, b in zip(naive["mean_qty"], true_mean["mean_qty"], strict=True)
    ]
    print("mean-of-means error:", [round(value, 6) for value in differences])
    assert max(differences) > 1e-9

    # The fix is to carry the pieces and divide at the end.
    correct = (
        fine.group_by("l_shipmode")
        .agg(qty=col("qty").sum(), lines=col("lines").sum())
        .with_columns(mean_qty=col("qty") / col("lines"))
        .sort("l_shipmode")
        .to_pydict()
    )
    assert all(
        abs(a - b) < 1e-9 for a, b in zip(correct["mean_qty"], true_mean["mean_qty"], strict=True)
    )


if __name__ == "__main__":
    main()
