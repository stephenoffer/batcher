"""Branching in an expression: when/then/otherwise, and its shorthands.

A CASE builder needs a terminating `otherwise` — an unfinished one is an error rather than
an implicit null, which catches the most common way these go wrong. Chain `when` calls for
more than two branches.

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
    orders = tpch("orders").select("o_orderkey", "o_totalprice").head(5_000)

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

    # An unfinished builder is an error, not an implicit null.
    try:
        orders.select(x=bt.when(col("o_totalprice") > 1).then(bt.lit(1)))
    except Exception as error:
        print("unfinished CASE refused:", str(error)[:70])
    else:
        raise AssertionError("expected an error for an unterminated CASE")


if __name__ == "__main__":
    main()
