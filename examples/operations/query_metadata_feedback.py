"""The feedback loop: what the executor measured, and what the optimizer does with it.

Core measures, Kyber decides. Running a query records actual cardinalities, and the next
run of the same shape plans against measurements rather than estimates. That is the whole
adaptive story, and it is visible in the estimates in `explain`.

    python examples/operations/query_metadata_feedback.py
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

    query = (
        lineitem.filter(col("l_quantity") > 45)
        .group_by("l_shipmode")
        .agg(lines=bt.count())
        .sort("l_shipmode")
    )

    # The plan before anything has run: the estimate is a guess.
    first_plan = query.explain()
    print("--- before running ---")
    print(first_plan)

    result = query.to_pydict()
    print("result:", result["lines"])
    actual = sum(result["lines"])
    assert actual > 0

    # The plan after: same shape, and the query still returns the same answer.
    second_plan = query.explain()
    print("--- after running ---")
    print(second_plan)
    assert query.to_pydict() == result

    # Re-optimization never changes the result — that is the invariant that makes the
    # loop safe to leave on.
    adaptive = query.collect(adaptive=True).to_pydict()
    static = query.collect(adaptive=False).to_pydict()
    assert adaptive == static == result


if __name__ == "__main__":
    main()
