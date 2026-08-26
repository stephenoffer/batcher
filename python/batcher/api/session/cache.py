"""Session-level control of the process result cache: what it holds, and dropping it.

`Dataset.cache()` opts one result in and `Dataset.uncache()` drops that one result;
these two are the whole-cache view. `clear_cache` is Spark's ``catalog.clearCache()``
by another spelling, and `cache_stats` is the reading that says whether the cache is
earning the memory and disk it is holding — a question no per-dataset call can answer,
because the budget is shared across every cached result in the process.
"""

from __future__ import annotations

__all__ = ["cache_stats", "clear_cache"]


def cache_stats() -> dict[str, int | float]:
    """Return how the process result cache is performing, across both of its tiers.

    Read this before tuning ``memory.result_cache_max_bytes``, because the two numbers
    that matter point in opposite directions. A low ``hit_rate`` with many ``evictions``
    means the budget is too small — the results were dropped before anything read them
    again. A low ``hit_rate`` with *no* evictions means the cache is not useful for this
    workload at all, and a bigger budget will not change that.

    ``demotions`` against ``disk_hits`` answers the same question for the disk tier:
    demotions with no disk hits means results are being written down that nothing reads
    back, and the disk budget is being spent for nothing.

    When ``memory.shared_cache_uri`` is set, ``shared_*`` keys report the cross-process
    store as well. Watch ``shared_errors`` first there: a shared cache degrades to
    recompute rather than failing a query, so an unreachable store looks exactly like a
    cold one until that count moves.

    The counts are lifetime figures for the process and are deliberately **not** reset by
    `clear_cache`: they are how you judge whether the cache is worth its budget, and a
    figure that resets whenever the cache is emptied cannot answer that. Take a difference
    across two readings to measure one span of work.

    Returns:
        The hit, miss, eviction, demotion and promotion counts, the aggregate hit-rate,
        the entry count, and the bytes held against each tier's budget. Every value is
        zero on a process that has cached nothing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> before = bt.cache_stats()
            >>> ds = bt.from_pydict({"k": [1, 1, 2], "v": [10, 20, 30]}).cache()
            >>> _ = ds.collect()  # miss: computed and stored
            >>> _ = ds.collect()  # hit: served from the cache
            >>> bt.cache_stats()["hits"] - before["hits"]
            1
    """
    from batcher import carbonite
    from batcher.carbonite.cache_shared import current_shared_cache

    stats = carbonite.result_cache().stats()
    shared = current_shared_cache()
    if shared is not None:
        stats.update({f"shared_{name}": value for name, value in shared.stats().items()})
    return stats


def clear_cache() -> None:
    """Drop every cached result, returning all cache memory and disk (Spark ``clearCache``).

    Results are recomputed on demand afterwards, so this changes what a query *costs*,
    never what it returns. The cache already evicts under its budget and yields its RAM
    back under memory pressure, so reach for this only when you want the whole cache gone
    at a known moment: between benchmark runs, or after rewriting data that cached
    results were derived from.

    This clears **this process's** tiers. A shared store (``memory.shared_cache_uri``) is
    left alone, because it belongs to every process reading it and one of them emptying it
    on the others' behalf is not a decision a local call should make. Its entries are keyed
    by their inputs' content versions, so rewriting the data those results came from
    already retires them.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> _ = bt.from_pydict({"x": [1, 2]}).cache().collect()
            >>> bt.clear_cache()
            >>> bt.cache_stats()["entries"]
            0
    """
    from batcher import carbonite

    cache = carbonite.current_result_cache()
    if cache is not None:
        cache.clear()
