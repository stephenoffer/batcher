"""Bounded memory: caching a reused branch and spilling under a tight budget.

``cache()`` is an execution hint, not a semantic change: the result is identical with or
without it. Spilling is the same idea for memory -- under a small budget the engine goes
out of core rather than failing, and the answer does not change.

    python examples/operations/memory_and_caching.py
"""

from __future__ import annotations

import dataclasses

import batcher as bt
from batcher import col
from batcher.config import active_config, config_context


def main() -> None:
    data = bt.from_pydict(
        {
            "grp": [f"g{i % 20}" for i in range(2000)],
            "v": list(range(2000)),
        }
    )

    # A branch used twice. Caching computes it once and reuses the result.
    filtered = data.filter(col("v") % 2 == 0).cache()
    total = filtered.select(t=col("v").sum()).to_pydict()
    n = filtered.count()
    print("cached branch:", total, n)
    assert n == 1000
    assert total["t"] == [sum(range(0, 2000, 2))]

    # The same query without the cache gives the identical answer.
    uncached = data.filter(col("v") % 2 == 0).select(t=col("v").sum()).to_pydict()
    assert uncached == total

    # `persist` is the eager sibling: materialize now, reuse later.
    persisted = data.filter(col("v") > 1900).persist()
    assert persisted.count() == 99

    # Run the same aggregate under a deliberately tight memory budget. The engine spills
    # rather than failing, and the result is unchanged.
    cfg = active_config()
    tight = cfg.replace(memory=dataclasses.replace(cfg.memory, default_total_bytes=8 * 1024 * 1024))

    def grouped() -> dict[str, list]:
        return data.group_by("grp").agg(total=col("v").sum(), n=bt.count()).sort("grp").to_pydict()

    baseline = grouped()
    with config_context(tight):
        spilled = grouped()

    print("groups:", len(baseline["grp"]))
    assert len(baseline["grp"]) == 20
    # Out-of-core execution is a scheduling decision, not a semantic one.
    assert spilled == baseline

    # Memory accounting for the current plan.
    usage = data.memory_usage()
    print("memory usage:", usage)
    assert usage is not None

    # Streaming keeps peak memory bounded regardless of the table size.
    seen = 0
    for batch in data.iter_batches(batch_size=256):
        seen += batch.num_rows
    assert seen == 2000


if __name__ == "__main__":
    main()
