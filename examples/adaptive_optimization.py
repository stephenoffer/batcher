"""Adaptive re-optimization: the moat.

Every other engine plans a query once, before it has seen a row, then commits to that
plan. Batcher re-optimizes *during* the query: at each pipeline breaker it has
*measured* the data it just processed (real row counts, not estimates) and re-plans
the rest on those numbers. ``explain()`` shows the plan the optimizer chose; running
with ``adaptive=True`` lets it revise mid-flight. The contract is that re-planning
never changes the relation — the adaptive result is identical to the static one.

    python examples/adaptive_optimization.py
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

import batcher as bt
from batcher import col


def _rowset(table: pa.Table) -> set:
    return {tuple(r.values()) for r in table.to_pylist()}


def main() -> None:
    # A large fact table joined to a small dimension, then aggregated — a join feeding
    # a group-by is a multi-stage query with a pipeline breaker to re-optimize at.
    fact = bt.from_arrow(pa.table({"k": np.arange(100_000) % 50, "v": np.arange(100_000) % 7}))
    dim = bt.from_arrow(pa.table({"k": np.arange(50), "label": [f"d{i}" for i in range(50)]}))

    def query() -> bt.Dataset:
        return fact.join(dim, on="k").group_by("label").agg(total=col("v").sum())

    # The optimized plan: cardinality estimates drive the join build-side choice.
    print(query().explain())

    static = query().collect()
    adaptive = query().collect(adaptive=True)

    # The moat's guarantee: adaptive re-planning never changes the result.
    assert _rowset(static) == _rowset(adaptive)
    print(f"static == adaptive: {static.num_rows} groups, identical results")


if __name__ == "__main__":
    main()
