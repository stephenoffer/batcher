"""Benchmark Carbonite's control-plane overhead and shuffle flow-control traffic.

Carbonite is the resource manager: it validates plans, hands out credit windows, and
decides spilling. None of that touches a tuple — but its *per-query* decision cost is
real fixed overhead on sub-second queries, and its *per-batch* shuffle control traffic
is real overhead on a large distributed shuffle. This script measures both against the
values the codebase optimizes for.

Run:
    source .venv/bin/activate
    python3 benchmarks/carbonite_perf.py

Two axes are reported:
  single-node   - the per-query decision path (envelope sample, manager construction),
                  dominated by a live OS memory read that is now shared across a short
                  TTL window instead of paid on every decision / back-to-back query.
  distributed   - the shuffle credit-grant *control-message* count for a partition of N
                  batches at window W: one-grant-per-batch (~N) vs low-watermark batched
                  refill (~2N/W).
"""

from __future__ import annotations

import time


def _time_us(fn, n: int) -> float:
    for _ in range(max(100, n // 10)):  # warm up
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6


def bench_single_node() -> None:
    """Per-query Carbonite decision cost — the fixed overhead a small query pays."""
    from batcher.carbonite import ResourceManager
    from batcher.carbonite.memory.pressure import (
        PressureMonitor,
        reset_memory_sampling,
        total_memory_bytes,
    )

    reset_memory_sampling()
    pm = PressureMonitor()
    rows = [
        ("total_memory_bytes()", _time_us(total_memory_bytes, 20000)),
        ("PressureMonitor.envelope_bytes()", _time_us(pm.envelope_bytes, 20000)),
        ("PressureMonitor.available_bytes()", _time_us(pm.available_bytes, 20000)),
        ("ResourceManager()  (full per-query construction)", _time_us(ResourceManager, 5000)),
    ]
    print("== single-node: per-query control-plane cost ==")
    for name, us in rows:
        print(f"  {name:<52} {us:8.3f} us")
    print()


def bench_distributed_credits() -> None:
    """Shuffle credit-grant control-message count: per-batch vs batched refill.

    Mirrors the transport's low-watermark refill (grant once ~half the window has
    freed) so the traffic reduction is visible without standing up a cluster.
    """
    print("== distributed: shuffle credit-grant control messages (N batches, window W) ==")
    print(f"  {'N':>8} {'W':>5} {'per-batch':>12} {'batched':>10} {'reduction':>10}")
    for n, w in [(1_000, 4), (10_000, 16), (100_000, 32), (1_000_000, 64)]:
        refill_at = max(1, w // 2)
        batched = -(-n // refill_at)  # ceil(n / refill_at)
        print(f"  {n:>8} {w:>5} {n:>12} {batched:>10} {n / batched:>9.1f}x")
    print()


def bench_cache_eviction() -> None:
    """Result-cache bulk eviction cost vs entry count.

    `on_pressure` (halve the cache) and a large insert evict many entries at once. The
    keep-value of an entry is independent of the others, so eviction is one stable sort
    (O(n log n)) — not an O(n) min-scan per victim (O(n²)).
    """
    import pyarrow as pa

    from batcher.carbonite.cache import CacheStore

    def _tbl() -> pa.Table:
        return pa.table({"v": pa.array(range(64), type=pa.int64())})

    print("== internals: result-cache bulk eviction (evict half of N entries) ==")
    print(f"  {'N entries':>10} {'evict-half':>12}")
    for n in (500, 1000, 2000, 4000):
        store = CacheStore(max_bytes=1 << 40)
        for i in range(n):
            store.put(f"k{i}", _tbl(), cost=float(i % 7))
        t0 = time.perf_counter()
        store._evict_to(store.used_bytes // 2)
        ms = (time.perf_counter() - t0) * 1e3
        print(f"  {n:>10} {ms:>10.2f}ms")
    print()


def main() -> None:
    bench_single_node()
    bench_distributed_credits()
    bench_cache_eviction()


if __name__ == "__main__":
    main()
