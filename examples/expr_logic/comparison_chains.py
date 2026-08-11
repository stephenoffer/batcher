"""Range checks, and comparing columns to each other.

A range check written as two comparisons joined by an ampersand is the portable spelling.
The parentheses around each half are mandatory, because Python binds the bitwise operators
tighter than the comparisons — leaving them off raises rather than quietly reassociating,
which is the lucky version of that mistake.

    python examples/expr_logic/comparison_chains.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")

    mid_range = lineitem.filter((col("l_quantity") >= 10) & (col("l_quantity") <= 20))
    print("quantities 10-20:", mid_range.count())
    assert 0 < mid_range.count() < lineitem.count()

    # The complement, which must account for every remaining row.
    outside = lineitem.filter((col("l_quantity") < 10) | (col("l_quantity") > 20))
    assert mid_range.count() + outside.count() == lineitem.count()

    # Comparing two columns rather than a column to a constant.
    late = lineitem.filter(col("l_receiptdate") > col("l_commitdate"))
    on_time = lineitem.filter(col("l_receiptdate") <= col("l_commitdate"))
    print("late:", late.count(), "on time:", on_time.count())
    assert late.count() + on_time.count() == lineitem.count()

    # Comparing derived values.
    discounted = lineitem.with_columns(net=col("l_extendedprice") * (1 - col("l_discount"))).filter(
        col("net") < col("l_extendedprice")
    )
    assert discounted.count() > 0

    # `is_in` against a set, and its negation.
    modes = ["AIR", "SHIP"]
    picked = lineitem.filter(col("l_shipmode").is_in(modes))
    rest = lineitem.filter(~col("l_shipmode").is_in(modes))
    assert picked.count() + rest.count() == lineitem.count()
    assert set(picked.select("l_shipmode").distinct().to_pydict()["l_shipmode"]) == set(modes)


if __name__ == "__main__":
    main()
