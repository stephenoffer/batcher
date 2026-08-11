"""Nulls: detecting them, filling them, and the arithmetic they poison.

Any arithmetic touching a null is null, and any comparison against one is null rather than
false. That is three-valued logic, and it is why `x != 1` does not return the rows where x
is null. `coalesce` is the fix, applied deliberately.

    python examples/expr_logic/null_handling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    # The full order table reaches past the slice of `lineitem` held here, so some
    # orders genuinely have no matching line — which is where the nulls come from.
    orders = tpch("orders").select("o_orderkey", "o_totalprice")
    lineitem = tpch("lineitem").select("l_orderkey", "l_extendedprice")

    # A left join is where nulls come from.
    joined = orders.join(lineitem, left_on="o_orderkey", right_on="l_orderkey", how="left")

    flags = joined.select(
        "o_orderkey",
        "l_extendedprice",
        missing=col("l_extendedprice").is_null(),
        present=col("l_extendedprice").is_not_null(),
        filled=bt.coalesce(col("l_extendedprice"), bt.lit(0.0)),
        # Arithmetic with a null yields a null.
        sum_with_null=col("o_totalprice") + col("l_extendedprice"),
    )

    nulls = flags.filter(col("missing")).count()
    print("rows with no matching line:", nulls)
    assert nulls > 0

    sample = flags.filter(col("missing")).head(3).to_pydict()
    print(sample)
    assert all(value is None for value in sample["sum_with_null"])
    assert all(value == 0.0 for value in sample["filled"])

    # `is_null` and `is_not_null` partition the rows exactly; a comparison does not.
    assert (
        flags.filter(col("missing")).count() + flags.filter(col("present")).count() == flags.count()
    )
    inequality = flags.filter(col("l_extendedprice") != -1.0).count()
    print(f"'!= -1' keeps {inequality} of {flags.count()} rows")
    assert inequality == flags.count() - nulls

    # `fill_null` is the frame-level version of coalesce.
    repaired = joined.fill_null(0.0)
    assert repaired.filter(col("l_extendedprice").is_null()).count() == 0


if __name__ == "__main__":
    main()
