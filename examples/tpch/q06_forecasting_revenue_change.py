"""TPC-H Q6 — the single-table scan query: three predicates and one sum.

Q6 has no joins and no grouping, so it measures exactly one thing: how fast the engine
gets through a filtered scan. It is also the query where predicate pushdown into the
Parquet reader shows up most clearly, because two of the three predicates are on
columns the sum never reads.

    python examples/tpch/q06_forecasting_revenue_change.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")

    start = dt.date(1994, 1, 1)
    end = dt.date(1995, 1, 1)

    selected = lineitem.filter(
        (col("l_shipdate") >= bt.lit(start))
        & (col("l_shipdate") < bt.lit(end))
        & (col("l_discount") >= 0.05)
        & (col("l_discount") <= 0.07)
        & (col("l_quantity") < 24)
    )

    result = selected.agg(revenue=(col("l_extendedprice") * col("l_discount")).sum()).to_pydict()
    print("forgone revenue:", f"{result['revenue'][0]:,.2f}")

    assert result["revenue"][0] > 0

    # The plan is worth a look: the filter sits directly on the scan, so the reader can
    # skip row groups whose statistics rule the date range out.
    plan = selected.explain()
    print(plan)
    assert "filter" in plan.lower()


if __name__ == "__main__":
    main()
