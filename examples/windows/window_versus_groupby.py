"""Window or group-by: the same aggregate, two different output shapes.

A group-by *replaces* the rows with one row per group. A window *adds a column* and keeps
every row. Reach for the window when downstream steps still need the detail, and the
group-by when they do not — carrying 200,000 rows to produce 7 is the usual waste.

    python examples/windows/window_versus_groupby.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem").select("l_shipmode", "l_extendedprice")

    grouped = (
        lineitem.group_by("l_shipmode")
        .agg(total=col("l_extendedprice").sum())
        .sort("l_shipmode")
        .to_pydict()
    )
    print("group_by rows:", len(grouped["l_shipmode"]))

    windowed = lineitem.with_columns(
        total=col("l_extendedprice").sum().over(partition_by=["l_shipmode"])
    )
    print("window rows:", windowed.count())

    # Same numbers, different shapes.
    assert windowed.count() == lineitem.count()
    assert len(grouped["l_shipmode"]) < windowed.count()

    from_window = windowed.select("l_shipmode", "total").distinct().sort("l_shipmode").to_pydict()
    assert from_window["l_shipmode"] == grouped["l_shipmode"]
    assert all(
        abs(left - right) < 1e-3
        for left, right in zip(from_window["total"], grouped["total"], strict=True)
    )

    # The window keeps enough detail to answer a row-level question the group-by cannot:
    # which lines are above their own ship mode's average.
    above = lineitem.with_columns(
        mode_mean=col("l_extendedprice").mean().over(partition_by=["l_shipmode"])
    ).filter(col("l_extendedprice") > col("mode_mean"))
    print("lines above their mode average:", above.count())
    assert 0 < above.count() < lineitem.count()
    assert bt is not None


if __name__ == "__main__":
    main()
