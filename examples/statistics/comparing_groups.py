"""Comparing a metric across groups, with the spread that says whether it means anything.

A difference in means is not a finding until you know the spread and the sample size. The
standard error of each group is the cheapest version of that, and it comes out of the same
pass as the mean.

    python examples/statistics/comparing_groups.py
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
            n=bt.count(),
            mean=col("l_extendedprice").mean(),
            std=bt.std(col("l_extendedprice")),
            sem=bt.sem(col("l_extendedprice")),
        )
        .sort("mean", descending=True)
        .to_pydict()
    )

    for index, mode in enumerate(per_mode["l_shipmode"]):
        low = per_mode["mean"][index] - 1.96 * per_mode["sem"][index]
        high = per_mode["mean"][index] + 1.96 * per_mode["sem"][index]
        print(
            f"{mode:<9} n={per_mode['n'][index]:>6} mean={per_mode['mean'][index]:>10,.1f} "
            f"95% CI [{low:,.0f}, {high:,.0f}]"
        )

    assert per_mode["mean"] == sorted(per_mode["mean"], reverse=True)
    assert all(value > 0 for value in per_mode["sem"])

    # The standard error is the standard deviation over the root of the sample size.
    assert all(
        abs(sem - std / count**0.5) < 1e-6
        for std, sem, count in zip(per_mode["std"], per_mode["sem"], per_mode["n"], strict=True)
    )

    # Ship mode is assigned independently of price in TPC-H, so the intervals overlap —
    # which is the correct reading, and the reason to compute them.
    widest_low = per_mode["mean"][0] - 1.96 * per_mode["sem"][0]
    narrowest_high = per_mode["mean"][-1] + 1.96 * per_mode["sem"][-1]
    print("intervals overlap:", widest_low < narrowest_high)


if __name__ == "__main__":
    main()
