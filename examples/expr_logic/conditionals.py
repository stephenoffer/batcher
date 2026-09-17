"""Branching in an expression: when/then/otherwise, and its shorthands.

A CASE builder without `otherwise` is SQL's `CASE WHEN ... END`: rows no branch matches
are null, typed like the branch values. Chain `when` calls for more than two branches.

    python examples/expr_logic/conditionals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderkey", "o_totalprice").limit(5_000)

    banded = orders.with_columns(
        band=bt.when(col("o_totalprice") < 50_000)
        .then(bt.lit("small"))
        .when(col("o_totalprice") < 150_000)
        .then(bt.lit("medium"))
        .otherwise(bt.lit("large")),
        flag=bt.iff(col("o_totalprice") > 100_000, bt.lit("big"), bt.lit("ordinary")),
    )

    counts = banded.value_counts("band").sort("band").to_pydict()
    print(counts)
    assert set(counts["band"]) <= {"small", "medium", "large"}
    assert sum(counts["count"]) == orders.count()

    # The bands really are ordered and disjoint.
    ranges = (
        banded.group_by("band")
        .agg(low=col("o_totalprice").min(), high=col("o_totalprice").max())
        .sort("low")
        .to_pydict()
    )
    print(ranges)
    assert all(
        high < next_low for high, next_low in zip(ranges["high"], ranges["low"][1:], strict=False)
    )

    # `iff` is the two-branch shorthand and agrees with the long form.
    checked = banded.filter(
        col("flag")
        != bt.when(col("o_totalprice") > 100_000).then(bt.lit("big")).otherwise(bt.lit("ordinary"))
    )
    assert checked.count() == 0

    # Without `otherwise`, an unmatched row is null -- the same as `otherwise(None)`.
    open_ended = orders.select(
        big=bt.when(col("o_totalprice") > 100_000).then(bt.lit("big")),
        explicit=bt.when(col("o_totalprice") > 100_000).then(bt.lit("big")).otherwise(None),
    ).to_pydict()
    print("rows with no matching branch:", open_ended["big"].count(None))
    assert open_ended["big"] == open_ended["explicit"]
    assert None in open_ended["big"] and "big" in open_ended["big"]


if __name__ == "__main__":
    main()
