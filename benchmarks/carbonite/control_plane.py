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
  internals     - the three per-bucket / per-fetch costs the subsystem pays inside a
                  query: bulk cache eviction, the spill store's free-disk probe, and the
                  shuffle ticket's wire form.
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


def bench_spill_disk_probe() -> None:
    """Free-space probing: one `statvfs` per bucket vs a short TTL window.

    The spill store consults free space once per bucket open, and a 4,096-way partitioned
    spill therefore asked the kernel 4,096 times for a figure that moves on a human
    timescale. The TTL still catches a disk filling *during* a query, which is the whole
    reason the store re-measures at all.
    """
    import shutil
    import tempfile

    from batcher.carbonite.spill import disk

    d = tempfile.mkdtemp()
    try:
        raw = _time_us(lambda: disk.read_free_disk_bytes(d), 20000)
        disk.reset_disk_sampling()
        ttl = _time_us(lambda: disk.free_disk_bytes(d), 20000)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("== spill: free-disk probe per bucket open ==")
    print(f"  {'raw statvfs':<52} {raw:8.3f} us")
    print(f"  {'TTL-windowed':<52} {ttl:8.3f} us   ({raw / max(ttl, 1e-9):.1f}x)")
    print()


def bench_ticket_rendering() -> None:
    """Shuffle-ticket wire form: rebuilt per fetch vs built once.

    An all-to-all shuffle has `workers**2` tickets and every transport call re-formats the
    one it is handed — `gather_*` builds a fresh `[(addr, str(ticket))]` list per round —
    so the string was on the per-fetch path.
    """
    from batcher.carbonite.transfer.server import ShuffleTicket

    ticket = ShuffleTicket(1, 2, 3, 4)

    def formatted() -> str:
        return (
            f"{ticket.plan_id}/{ticket.stage_id}/{ticket.src_partition}/"
            f"{ticket.dst_partition}/{ticket.epoch}"
        )

    fmt = _time_us(formatted, 200000)
    cached = _time_us(lambda: str(ticket), 200000)
    print("== transfer: shuffle-ticket wire form per fetch ==")
    print(f"  {'rebuilt (f-string)':<52} {fmt:8.3f} us")
    print(f"  {'built once':<52} {cached:8.3f} us   ({fmt / max(cached, 1e-9):.1f}x)")
    print()


def main() -> None:
    bench_single_node()
    bench_distributed_credits()
    bench_cache_eviction()
    bench_spill_disk_probe()
    bench_ticket_rendering()


if __name__ == "__main__":
    main()
