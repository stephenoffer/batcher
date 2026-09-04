"""What the result cache's tiers and the lookup join actually cost.

Three questions this answers, each with the number that settles it:

1. **Is the disk tier worth its disk?** A cached result the memory budget cannot hold is
   demoted rather than dropped. That is only useful if reading it back beats recomputing
   it, so the comparison is `cold` against `disk`.
2. **Does the cache cost anything when nothing is cached?** Every query in the engine pays
   the guards this added; only a cached one benefits. An uncached query must be unchanged.
3. **Does the lookup join scale with distinct keys rather than with rows?** That is the
   feature's whole claim, and it is a claim about *Python* work — an assembly step that
   walks the batch row by row makes it false while returning perfectly correct answers.

Run it with ``python benchmarks/run.py --benchmark cache``, or directly as
``python benchmarks/internals/cache_bench.py``.

**On comparing the lookup join to a hash join.** The store here is in-process, so the hash
join has no network to pay and the lookup join has no round trips to save — this measures
the *overhead*, not the win. A hash join against Redis would have to pull the whole
dimension over the wire first, which is the case the lookup join exists for and which this
box cannot measure. Read the ratio here as a ceiling on what the mechanism costs when it is
free to be beaten, not as the reason to use it.

Also: measure with ``collect()``, never with ``count()``. A count over a row-preserving
plan is answered from metadata without executing, so a ``count()``-based comparison times a
real join against a no-op and reports a 268x difference that is entirely an artifact. That
happened while this file was being written.
"""

from __future__ import annotations

import dataclasses
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

import batcher as bt
from batcher.config import active_config, config_context

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envinfo import machine_fingerprint, require_quiet_box, require_release_build

#: Rows in the cached aggregate's input. Large enough that a recompute is measurable and
#: small enough that the whole suite runs in well under a minute on a shared box.
_CACHE_ROWS = 4_000_000
_CACHE_GROUPS = 50_000
#: The lookup-join shape: a dimension far larger than what the probe touches, which is the
#: only shape where a lookup join is the right answer.
_DIM_ROWS = 2_000_000
_PROBE_ROWS = 2_000_000
_DISTINCT_KEYS = 5_000


def _median_ms(fn: Callable[[], Any], n: int = 5) -> float:
    """Median wall-clock milliseconds over `n` runs.

    Median rather than mean: this box is shared, and one descheduled run should not move
    the reported figure.
    """
    times = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def _cache_tiers() -> None:
    """Cold, memory-tier and disk-tier timings for one cached aggregate."""
    base = bt.range(0, _CACHE_ROWS).with_columns(g=bt.col("value") % _CACHE_GROUPS)

    def query(cached: bool):
        agg = base.group_by("g").agg(n=bt.count(), s=bt.col("value").sum())
        return agg.cache() if cached else agg

    bt.clear_cache()
    cold = _median_ms(lambda: query(False).collect())
    cold_count = _median_ms(lambda: query(False).count())

    bt.clear_cache()
    hot = query(True)
    hot.collect()
    memory = _median_ms(hot.collect)
    memory_count = _median_ms(hot.count)

    # A memory budget too small to hold the result, so it is written straight to disk.
    cfg = active_config()
    tiny = cfg.replace(
        memory=dataclasses.replace(
            cfg.memory, result_cache_max_bytes=1 << 20, result_cache_disk_max_bytes=4 << 30
        )
    )
    bt.clear_cache()
    from batcher.carbonite.cache import reset_result_cache

    with config_context(tiny):
        reset_result_cache()
        demoted = query(True)
        demoted.collect()
        held = bt.cache_stats()
        disk = _median_ms(demoted.collect)
    reset_result_cache()
    bt.clear_cache()

    print(f"=== result cache ({_CACHE_ROWS:,} rows -> {_CACHE_GROUPS:,} groups) ===")
    print(f"{'path':34s} {'ms':>10s}  {'vs cold':>9s}")
    print("-" * 56)
    print(f"{'cold (uncached) collect()':34s} {cold:10.1f}  {'-':>9s}")
    print(f"{'warm, memory tier':34s} {memory:10.1f}  {cold / memory:8.1f}x")
    print(f"{'warm, disk tier':34s} {disk:10.1f}  {cold / disk:8.1f}x")
    print(f"{'cold (uncached) count()':34s} {cold_count:10.1f}  {'-':>9s}")
    speedup = cold_count / memory_count
    print(f"{'warm count(), memory tier':34s} {memory_count:10.1f}  {speedup:8.1f}x")
    megabytes = held["disk_used_bytes"] / 1e6
    print(f"\ndisk tier held {megabytes:.1f} MB for {held['disk_entries']} entry\n")


def _lookup_assembly() -> None:
    """The per-batch assembly step, row-wise against vectorized.

    Isolated from the rest of the join because it is where the feature's scaling property
    is won or lost, and because a whole-query timing buries a 37x difference in the noise
    of everything around it.
    """
    from batcher.io.lookup import backends
    from batcher.io.lookup.base import lookup_arrays

    dim = pa.table(
        {
            "k": [f"k{i}" for i in range(_DIM_ROWS)],
            "label": [f"l{i % 97}" for i in range(_DIM_ROWS)],
        }
    )
    lookup = backends.InMemoryLookup(dim, "k")
    schema = lookup.value_schema()
    keys = pa.array([f"k{i % _DISTINCT_KEYS}" for i in range(1_000_000)])
    resolved = lookup.multi_get(pc.unique(keys).to_pylist())

    def row_wise() -> list[pa.Array]:
        """What `lookup_arrays` replaced: one dict lookup per row per column."""
        key_list = keys.to_pylist()
        return [
            pa.array(
                [
                    None if key is None or (row := resolved.get(key)) is None else row.get(f.name)
                    for key in key_list
                ],
                type=f.type,
            )
            for f in schema
        ]

    slow = _median_ms(row_wise, n=3)
    fast = _median_ms(lambda: lookup_arrays(keys, resolved, schema), n=3)
    print("=== lookup-join assembly (1,000,000 rows, 5,000 distinct keys, 1 column) ===")
    print(f"{'row-wise Python':34s} {slow:10.1f}")
    print(f"{'vectorized (index_in + take)':34s} {fast:10.1f}  {slow / fast:8.1f}x\n")


def _lookup_join() -> None:
    """The whole lookup join against a hash join over the same dimension, in-process."""
    from batcher.io.lookup import backends, spec

    dim = pa.table(
        {
            "k": [f"k{i}" for i in range(_DIM_ROWS)],
            "label": [f"l{i % 97}" for i in range(_DIM_ROWS)],
        }
    )
    lookup = backends.InMemoryLookup(dim, "k")
    fetched: list[int] = []

    class _Counting:
        def multi_get(self, keys: list[str]) -> dict:
            fetched.append(len(keys))
            return lookup.multi_get(keys)

        def value_schema(self) -> pa.Schema:
            return lookup.value_schema()

        def close(self) -> None:
            pass

    original = spec.build_lookup
    spec.build_lookup = lambda *_args, **_kwargs: _Counting()
    try:
        probe = bt.from_pydict(
            {
                "k": [f"k{i % _DISTINCT_KEYS}" for i in range(_PROBE_ROWS)],
                "v": list(range(_PROBE_ROWS)),
            }
        )
        dim_ds = bt.from_arrow(dim)
        hash_ms = _median_ms(lambda: probe.join(dim_ds, on="k", how="left").collect(), n=3)
        rows = []
        for label, batch_size in (
            ("lookup_join (engine batching)", None),
            ("lookup_join (batch_size=16k)", 16_384),
            ("lookup_join (batch_size=512k)", 524_288),
        ):
            fetched.clear()
            ms = _median_ms(
                lambda bs=batch_size: probe.lookup_join(
                    "bench://",
                    on="k",
                    schema={"label": "string"},
                    batch_size=bs,
                    num_workers=1,
                ).collect(),
                n=3,
            )
            rows.append((label, ms, sum(fetched) / 3))
    finally:
        spec.build_lookup = original

    print(
        f"=== lookup join ({_DIM_ROWS:,}-row dimension, {_PROBE_ROWS:,}-row probe, "
        f"{_DISTINCT_KEYS:,} distinct keys, one worker) ==="
    )
    print(f"{'path':34s} {'ms':>10s}  {'vs join':>9s}  {'keys fetched':>13s}")
    print("-" * 72)
    print(f"{'join (reads the dimension)':34s} {hash_ms:10.1f}  {'-':>9s}  {_DIM_ROWS:13,}")
    for label, ms, keys in rows:
        print(f"{label:34s} {ms:10.1f}  {ms / hash_ms:8.1f}x  {keys:13,.0f}")
    print()
    print("Read two things here. The keys column is the point: the join touches a fraction of")
    print("the store, which is what makes a dimension that does not fit joinable at all. And")
    print("the batch_size spread is the trap: assembly costs one unit of Python per *distinct")
    print("key per batch*, so batches smaller than the distinct-key count pay for the same keys")
    print("over and over. The engine's own batching is large enough here; an explicitly small")
    print("one is several times worse.")
    print()
    print("The store is in-process, so the ratio against the hash join is this mechanism's")
    print("overhead, not its win: a hash join against Redis would have to pull the whole")
    print("dimension over the wire first, which is the case that has no ratio to report.")
    print()


def main() -> int:
    """Run every section and print its table.

    Returns:
        ``0``. There is no pass/fail here: the numbers are the output, and the regression
        judgement is a human comparing them against `benchmarks/BENCHMARK_RESULTS.md`.
    """
    # Refuse to time a dev-profile engine: it is 8-60x slower, so a number taken from one
    # compares an unoptimized Batcher against release competitors. `BENCH_ALLOW_DEBUG_BUILD=1`
    # overrides deliberately.
    require_release_build()
    # Print the machine before any number: a timing is only reproducible beside the
    # box that produced it, and this file's own history has ratios quoted across four
    # different machines as if they were comparable.
    print(machine_fingerprint())
    # ...and refuse a contended one: a neighbour's load is not a fact about any
    # engine. `BENCH_ALLOW_BUSY_BOX=1` overrides.
    require_quiet_box()
    print(f"Batcher cache + lookup benchmark  (engine {bt.engine_version()})\n")
    _cache_tiers()
    _lookup_assembly()
    _lookup_join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
