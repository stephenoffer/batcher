"""Bounded memory: caching a reused branch, storage levels, and spilling under a tight budget.

``cache()`` is an execution hint, not a semantic change: the result is identical with or
without it, at every storage level and on either tier. Spilling is the same idea for
memory -- under a small budget the engine goes out of core rather than failing, and the
answer does not change.

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

    # `persist` is the Spark spelling of the same marker.
    persisted = data.filter(col("v") > 1900).cache()
    assert persisted.count() == 99

    # A storage level says which media the stored result may occupy. The default,
    # MEMORY_AND_DISK, demotes to a local disk tier when the memory budget evicts, so a
    # working set larger than the budget costs a read-back rather than a recompute.
    # DISK_ONLY never charges the memory budget at all.
    on_disk = data.group_by("grp").agg(n=bt.count()).cache(bt.StorageLevel.DISK_ONLY)
    assert on_disk.count() == 20
    assert on_disk.count() == 20  # the second one is served from the disk tier
    # Where a result is stored never changes what it is.
    assert on_disk.to_pydict() == data.group_by("grp").agg(n=bt.count()).to_pydict()

    # `cache_stats()` is how you tell whether the cache is earning its budget. The
    # counters are lifetime figures for the process, so read a difference across a span
    # of work rather than an absolute.
    before = bt.cache_stats()
    reused = data.filter(col("v") > 1000).cache()
    reused.collect()  # miss: computed and stored
    reused.collect()  # hit: served from the cache
    after = bt.cache_stats()
    print("cache hits over that span:", after["hits"] - before["hits"])
    assert after["hits"] - before["hits"] == 1

    # `uncache` (Spark: `unpersist`) gives one result's memory and disk back at a known
    # moment; `clear_cache()` does it for every cached result at once.
    reused.uncache()
    bt.clear_cache()
    assert bt.cache_stats()["entries"] == 0
    # Dropping a cached result changes what a query costs, never what it returns.
    assert reused.count() == 999

    # A terminal that materializes the result fills the cache; count(), is_empty() and
    # iter_batches() read a warm one but never fill it, because filling it would mean
    # materializing the result those three exist to avoid materializing.
    warm = data.filter(col("v") > 1000).cache()
    warm.collect()
    hits = bt.cache_stats()["hits"]
    assert warm.count() == 999
    assert bt.cache_stats()["hits"] == hits + 1

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
