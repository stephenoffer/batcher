"""Building a complete grid with a cross join, and filling the gaps.

A group-by only produces rows for combinations that occur. When a report needs every
combination — every month for every region, including the empty ones — the grid comes from
a cross join and the data is left-joined onto it.

    python examples/relational/cross_join_grids.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderdate", "o_orderpriority", "o_totalprice")

    observed = (
        orders.with_columns(year=col("o_orderdate").dt.year())
        .group_by("year", "o_orderpriority")
        .agg(revenue=col("o_totalprice").sum())
    )
    print("observed combinations:", observed.count())

    # The complete grid: every year crossed with every priority.
    years = observed.select("year").distinct()
    priorities = observed.select("o_orderpriority").distinct()
    grid = years.cross_join(priorities)
    print("grid size:", grid.count())
    assert grid.count() == years.count() * priorities.count()

    # Left-join the data onto the grid, then fill.
    complete = grid.join(observed, on=["year", "o_orderpriority"], how="left").with_columns(
        revenue=bt.coalesce(col("revenue"), bt.lit(0.0))
    )
    assert complete.count() == grid.count()
    assert complete.filter(col("revenue").is_null()).count() == 0

    # The filled grid carries the same total as the observed data.
    filled_total = complete.agg(t=col("revenue").sum()).to_pydict()["t"][0]
    observed_total = observed.agg(t=col("revenue").sum()).to_pydict()["t"][0]
    assert abs(filled_total - observed_total) < 1e-3

    # Any zero rows are combinations that genuinely did not occur.
    empty = complete.filter(col("revenue") == 0.0).count()
    print(f"{empty} combinations had no orders")
    assert empty == grid.count() - observed.count()


if __name__ == "__main__":
    main()
