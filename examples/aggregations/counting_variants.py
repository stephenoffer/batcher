"""Counting: rows, non-nulls, matches, and distinct values.

`bt.count()` counts rows. `col(x).count()` counts non-null values of x. Those differ the
moment a column has nulls, and confusing them is the most common way an average comes out
wrong — because the denominator changed without anyone noticing.

Nulls here come from a left join, which is where they come from in practice. There is no
null literal to reach for: `bt.nullif(a, b)` is the expression that produces one.

    python examples/aggregations/counting_variants.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders")
    lineitem = tpch("lineitem")

    # A left join is the honest way to get a column that is null for some rows: orders
    # beyond the slice of `lineitem` we hold simply have no matching line.
    first_lines = lineitem.filter(col("l_linenumber") == 1).select("l_orderkey", "l_extendedprice")
    with_gaps = orders.join(first_lines, left_on="o_orderkey", right_on="l_orderkey", how="left")

    counts = with_gaps.agg(
        rows=bt.count(),
        non_null=col("l_extendedprice").count(),
        matching=bt.count_if(col("o_orderstatus") == "F"),
        distinct_clerks=bt.n_unique(col("o_clerk")),
        approx_clerks=bt.approx_n_unique(col("o_clerk")),
        null_share=bt.null_rate(col("l_extendedprice")),
        present_share=bt.non_null_rate(col("l_extendedprice")),
    ).to_pydict()
    print(counts)

    assert counts["rows"][0] == orders.count()
    # Some orders matched a line and some did not, so the two counts genuinely differ.
    assert 0 < counts["non_null"][0] < counts["rows"][0]

    # The null rate is exactly the complement of the non-null count.
    expected_rate = 1.0 - counts["non_null"][0] / counts["rows"][0]
    assert abs(counts["null_share"][0] - expected_rate) < 1e-9
    assert abs(counts["null_share"][0] + counts["present_share"][0] - 1.0) < 1e-9

    # The sketch-backed approximation is close, and much cheaper at scale.
    exact = counts["distinct_clerks"][0]
    approx = counts["approx_clerks"][0]
    print(f"clerks exact={exact} approx={approx}")
    assert abs(approx - exact) / exact < 0.05

    # `nullif` is the expression that makes a null: it returns null where the two sides
    # are equal, and the left side otherwise.
    blanked = orders.select(status=bt.nullif(col("o_orderstatus"), bt.lit("F")))
    assert blanked.agg(n=col("status").count()).to_pydict()["n"][0] < orders.count()


if __name__ == "__main__":
    main()
