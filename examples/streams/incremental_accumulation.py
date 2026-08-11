"""Accumulating state across batches with a mergeable aggregate.

A streaming aggregate is the same `partial → combine → finalize` algebra a distributed one
uses. Running it batch by batch and comparing against the one-shot answer is the check that
the operator really is mergeable rather than merely appearing to work.

    python examples/streams/incremental_accumulation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem").select("l_shipmode", "l_quantity", "l_extendedprice")

    # Split the input into "arrivals" and fold each one in.
    arrivals = [lineitem.slice(start, 40_000) for start in range(0, 200_000, 40_000)]
    assert sum(part.count() for part in arrivals) == lineitem.count()

    running: bt.Dataset | None = None
    for index, part in enumerate(arrivals):
        partial = part.group_by("l_shipmode").agg(lines=bt.count(), qty=col("l_quantity").sum())
        running = partial if running is None else running.union(partial)
        print(f"  after arrival {index}: {running.count()} partial groups")

    assert running is not None

    # Combine the partials, which is the step that has to be associative.
    combined = (
        running.group_by("l_shipmode")
        .agg(lines=col("lines").sum(), qty=col("qty").sum())
        .sort("l_shipmode")
        .to_pydict()
    )

    one_shot = (
        lineitem.group_by("l_shipmode")
        .agg(lines=bt.count(), qty=col("l_quantity").sum())
        .sort("l_shipmode")
        .to_pydict()
    )

    print(combined["l_shipmode"], combined["lines"])
    assert combined["l_shipmode"] == one_shot["l_shipmode"]
    assert combined["lines"] == one_shot["lines"]
    assert all(abs(a - b) < 1e-6 for a, b in zip(combined["qty"], one_shot["qty"], strict=True))


if __name__ == "__main__":
    main()
