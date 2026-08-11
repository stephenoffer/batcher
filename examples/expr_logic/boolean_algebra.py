"""Boolean columns: combining them, and counting what they say.

A predicate is a column like any other, so it can be stored, combined and aggregated. That
is what makes a rule engine expressible as a projection: each rule is a boolean column, and
the verdict is a fold over them.

    python examples/expr_logic/boolean_algebra.py
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

    rules = lineitem.select(
        big=col("l_quantity") > 30,
        discounted=col("l_discount") > 0.05,
        returned=col("l_returnflag") == "R",
    )

    combined = rules.with_columns(
        all_three=col("big") & col("discounted") & col("returned"),
        any_one=col("big") | col("discounted") | col("returned"),
        none=~(col("big") | col("discounted") | col("returned")),
    )

    counts = combined.agg(
        big=bt.count_if(col("big")),
        all_three=bt.count_if(col("all_three")),
        any_one=bt.count_if(col("any_one")),
        none=bt.count_if(col("none")),
        rows=bt.count(),
    ).to_pydict()
    print(counts)

    # The conjunction is bounded by each of its terms.
    assert counts["all_three"][0] <= counts["big"][0]
    # The disjunction is at least as large as any term.
    assert counts["any_one"][0] >= counts["big"][0]
    # De Morgan: "none" is the exact complement of "any".
    assert counts["any_one"][0] + counts["none"][0] == counts["rows"][0]

    # Counting satisfied rules per row, which is the score a rule engine reports.
    scored = rules.select(
        score=col("big").cast("int64")
        + col("discounted").cast("int64")
        + col("returned").cast("int64")
    )
    distribution = scored.value_counts("score").sort("score").to_pydict()
    print(distribution)
    assert set(distribution["score"]) <= {0, 1, 2, 3}
    assert sum(distribution["count"]) == lineitem.count()


if __name__ == "__main__":
    main()
