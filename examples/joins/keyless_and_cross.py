"""Joining with no key at all, and keeping it safe.

A cross join is the only join with no key, so it is the only one whose output size you can
compute in advance: left times right. Computing that before you run it is the guard rail —
the failure mode is not slowness, it is a machine falling over.

    python examples/joins/keyless_and_cross.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    nation = tpch("nation").select("n_nationkey", "n_name")
    region = tpch("region").select("r_regionkey", "r_name")

    # Compute the size before running it.
    projected_rows = nation.count() * region.count()
    print(f"{nation.count()} x {region.count()} = {projected_rows} rows")
    assert projected_rows == 125

    grid = nation.cross_join(region)
    assert grid.count() == projected_rows
    assert grid.width == nation.width + region.width

    # A cross join followed by a predicate is how a non-equi join is expressed. Here it
    # reproduces the real foreign key.
    reconstructed = (
        grid.filter(col("n_regionkey") == col("r_regionkey"))
        if ("n_regionkey" in grid.columns)
        else None
    )
    if reconstructed is None:
        full_nation = tpch("nation")
        grid = full_nation.cross_join(region)
        reconstructed = grid.filter(col("n_regionkey") == col("r_regionkey"))

    direct = tpch("nation").join(region, left_on="n_regionkey", right_on="r_regionkey")
    print("reconstructed:", reconstructed.count(), "direct:", direct.count())
    assert reconstructed.count() == direct.count() == 25

    # The guard rail: refuse a cross join whose size you would not accept.
    limit = 1_000_000
    lineitem = tpch("lineitem")
    would_be = lineitem.count() * nation.count()
    print(f"lineitem x nation would be {would_be:,} rows")
    assert would_be > limit
    # ...so this one does not run.

    # A one-row right side is the safe case, and is how a scalar parameter is attached.
    parameters = bt.from_pydict({"cutoff": [30.0]})
    filtered = lineitem.cross_join(parameters).filter(col("l_quantity") > col("cutoff"))
    assert filtered.count() == lineitem.filter(col("l_quantity") > 30.0).count()


if __name__ == "__main__":
    main()
