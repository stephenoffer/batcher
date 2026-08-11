"""Making a run reproducible, and finding out which parts are not.

Sampling, shuffling and any unordered result are the three sources of run-to-run variation. A
seed fixes the first two; an explicit sort fixes the third. Everything else already is
reproducible, and asserting that is how you keep it so.

    python examples/operations/reproducible_runs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem").select("l_orderkey", "l_quantity", "l_shipmode")

    # An aggregate is reproducible without any help.
    def summary() -> dict:
        return (
            lineitem.group_by("l_shipmode")
            .agg(lines=bt.count(), qty=col("l_quantity").sum())
            .sort("l_shipmode")
            .to_pydict()
        )

    assert summary() == summary()
    print("aggregates are reproducible")

    # A sample needs a seed.
    seeded = lineitem.sample(n=500, seed=42).to_pydict()
    again = lineitem.sample(n=500, seed=42).to_pydict()
    assert seeded == again
    different = lineitem.sample(n=500, seed=43).to_pydict()
    assert different != seeded
    print("a seeded sample is reproducible; a different seed is not the same rows")

    # A shuffle needs a seed too.
    shuffled = lineitem.select("l_orderkey").shuffle(seed=7).head(20).to_pydict()
    reshuffled = lineitem.select("l_orderkey").shuffle(seed=7).head(20).to_pydict()
    assert shuffled == reshuffled

    # An unsorted head is not a defined result, so it needs an ordering to be compared.
    ordered = lineitem.sort("l_orderkey", "l_quantity", "l_shipmode").head(20).to_pydict()
    ordered_again = lineitem.sort("l_orderkey", "l_quantity", "l_shipmode").head(20).to_pydict()
    assert ordered == ordered_again
    print("a totally ordered head is reproducible")

    # And the partition count must not matter for any of it.
    assert (
        lineitem.group_by("l_shipmode")
        .agg(lines=bt.count())
        .sort("l_shipmode")
        .collect(num_partitions=8)
        .to_pydict()["lines"]
        == summary()["lines"]
    )


if __name__ == "__main__":
    main()
