"""Performance: caching a reused result and spilling under a tiny memory budget.

Two everyday performance levers, both result-invariant:

- ``cache()`` marks a dataset's result to be stored in memory the first time it is
  computed, so later terminals on the same dataset skip re-execution.
- A small ``memory.max_memory_bytes`` budget makes the engine run stateful operators
  out of core (partition-and-spill) instead of running out of memory; the result is
  identical to the in-memory run.

Run it directly::

    python examples/performance_caching.py
"""

from __future__ import annotations

import batcher as bt
from batcher.config import Config, MemoryConfig, config_context


def main() -> None:
    events = bt.from_pydict(
        {
            "region": ["us", "eu", "us", "eu", "us", "eu"],
            "status": ["active", "active", "churned", "active", "active", "churned"],
            "amount": [10.0, 5.0, 99.0, 7.0, 3.0, 8.0],
        }
    )

    # Cache the expensive upstream (the filter) once, then reuse it across queries.
    hot = events.filter(bt.col("status") == "active").cache()

    first = hot.to_pydict()  # computed once, then stored in the result cache
    second = hot.to_pydict()  # cache hit — no re-execution
    assert first == second, "a cached dataset returns the same result"
    assert sorted(first["region"]) == ["eu", "eu", "us", "us"]

    # Two terminals on the cached handle: the active rows are materialized once.
    totals = hot.group_by("region").agg(total=bt.col("amount").sum()).sort("region").to_pydict()
    assert totals == {"region": ["eu", "us"], "total": [12.0, 13.0]}
    assert hot.count() == 4

    # Out-of-core spilling: a deliberately tiny budget forces the spill path on a
    # small dataset so this runs anywhere. The result must equal the in-memory run.
    big = bt.from_pydict({"k": [i % 50 for i in range(2000)], "v": list(range(2000))})

    def grouped(ds: bt.Dataset) -> dict:
        return ds.group_by("k").agg(total=bt.col("v").sum()).sort("k").to_pydict()

    in_memory = grouped(big)

    tiny_budget = Config().replace(memory=MemoryConfig(max_memory_bytes=1))
    with config_context(tiny_budget):
        spilled = grouped(big)

    assert in_memory == spilled, "the spilled result must equal the in-memory result"
    assert len(spilled["k"]) == 50

    print("cache reuse + out-of-core spill: results identical")
    print(f"  active groups: {totals}")
    print(f"  spilled groups: {len(spilled['k'])} (== in-memory)")


if __name__ == "__main__":
    main()
