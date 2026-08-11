"""Three ways to structure a long pipeline, and what each costs.

Method chaining, `pipe` with named functions, and a list of transformations applied in a fold
all build the same plan. The choice is about which one a reader can follow and which one a
test can exercise a piece of.

    python examples/relational/pipeline_composition_patterns.py
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")

    chained = (
        lineitem.filter(col("l_quantity") > 20)
        .with_columns(revenue=col("l_extendedprice") * (1 - col("l_discount")))
        .filter(col("revenue") > 20_000)
        .group_by("l_shipmode")
        .agg(total=col("revenue").sum())
        .sort("l_shipmode")
    )

    def big_lines(dataset: bt.Dataset) -> bt.Dataset:
        return dataset.filter(col("l_quantity") > 20)

    def with_revenue(dataset: bt.Dataset) -> bt.Dataset:
        return dataset.with_columns(revenue=col("l_extendedprice") * (1 - col("l_discount")))

    def valuable(dataset: bt.Dataset) -> bt.Dataset:
        return dataset.filter(col("revenue") > 20_000)

    def summarize(dataset: bt.Dataset) -> bt.Dataset:
        return dataset.group_by("l_shipmode").agg(total=col("revenue").sum()).sort("l_shipmode")

    piped = lineitem.pipe(big_lines).pipe(with_revenue).pipe(valuable).pipe(summarize)

    steps: list[Callable[[bt.Dataset], bt.Dataset]] = [big_lines, with_revenue, valuable, summarize]
    folded = lineitem
    for step in steps:
        folded = step(folded)

    reference = chained.to_pydict()
    print(reference["l_shipmode"], [round(value) for value in reference["total"]])

    # All three build the same plan and give the same answer.
    assert piped.explain() == chained.explain()
    assert folded.explain() == chained.explain()
    assert piped.to_pydict() == reference
    assert folded.to_pydict() == reference

    # The named steps are individually testable, which the chain is not.
    assert big_lines(lineitem).count() < lineitem.count()
    assert "revenue" in with_revenue(lineitem).columns


if __name__ == "__main__":
    main()
