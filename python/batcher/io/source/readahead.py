"""How much of a remote Parquet read one scan keeps in flight, and how it keeps the order.

A remote read is bound by the requests outstanding, not by any one request's speed, and a
row group's projected column chunks coalesce into ONE small GET — TPC-H SF1000 q6's four
`lineitem` columns are ~1.8 MB per row group. So a read-ahead counted in row groups is a
throughput decision taken blind to the projection: 32 row groups kept four 8-row-group
windows in flight, and every 16-core node of an 8-node fleet read S3 at ~190 MB/s with 8%
CPU busy, a sixth of its NIC. Held in bytes, the same scan keeps ~1,000 of those row groups
outstanding and reads at ~1.0-1.2 GB/s per node, the link rate; q6 62 s -> 14 s. A wide
projection gets proportionally fewer windows, so the resident footprint is the budget
whatever the width. `BATCHER_NATIVE_READ_BUDGET_BYTES` overrides the derived figure.

Neutral IO: it knows about splits and memory, not about plans or the distributed executor.
"""

from __future__ import annotations

from typing import Any

from batcher._internal.hardware.memory import machine_memory_bytes
from batcher._internal.logging import note_suppressed
from batcher.config.env import env_int

__all__ = [
    "native_read_budget",
    "native_read_depth",
    "ordered_concurrent",
    "projected_rg_bytes",
    "task_memory_share",
]

_BUDGET_FLOOR = 256 << 20
_BUDGET_CEILING = 4 << 30


def task_memory_share(node_processes: int) -> int:
    """The bytes of this node's RAM that belong to the calling task, or `0` when unknown.

    A distributed scan task is granted a share of a node's cores — the whole node, on the
    one-worker-per-node fleet — and its share of the RAM is the same fraction. Read from the
    task's own grant, so a 16-core worker on a 16-core node is sized against the node and a
    1-core task on it against a sixteenth; outside a Ray task, the node divided among its
    `node_processes`.
    """
    total = machine_memory_bytes()
    if not total:
        return 0
    node_processes = max(1, node_processes)
    try:
        import ray

        if ray.is_initialized():
            granted = float(ray.get_runtime_context().get_assigned_resources().get("CPU", 0.0))
            if granted > 0:
                return int(total * min(1.0, granted / node_processes))
    except Exception as exc:
        note_suppressed("io", "read task CPU grant", exc)
    return total // node_processes


def native_read_budget(node_processes: int) -> int:
    """Projected bytes one native scan keeps in flight: an eighth of its memory share."""
    override = env_int("BATCHER_NATIVE_READ_BUDGET_BYTES", 0, floor=0)
    if override:
        return override
    share = task_memory_share(node_processes)
    if not share:
        return _BUDGET_FLOOR
    return max(_BUDGET_FLOOR, min(_BUDGET_CEILING, share // 8))


def native_read_depth(
    units: list[tuple[str, list[int]]],
    rg_bytes: float | None,
    fallback_row_groups: int,
    node_processes: int = 1,
) -> int:
    """How many row-group windows to keep in flight, bounded by projected BYTES.

    `rg_bytes` is the projected, decoded size of one row group. Knowing it, the depth is the
    read budget divided by a window's size, so a narrow projection keeps many small requests
    outstanding and a wide one few large ones. Without it the budget falls back to
    `fallback_row_groups` row groups, the per-split reader's own bound.

    Args:
        units: The `(uri, row_groups)` windows the read was cut into.
        rg_bytes: Projected bytes per row group, or `None` when the footer did not say.
        fallback_row_groups: The row-group budget to use when `rg_bytes` is unknown.
        node_processes: Worker processes sharing this node's RAM.

    Returns:
        The number of windows to read concurrently, never below 1.
    """
    widest = max((len(window) for _uri, window in units), default=1)
    if rg_bytes and rg_bytes > 0:
        budget = native_read_budget(node_processes)
        return max(1, min(len(units), int(budget // (rg_bytes * max(1, widest)))))
    return max(1, fallback_row_groups // max(1, widest))


def projected_rg_bytes(splits: list[Any], cols: list[str] | None) -> float | None:
    """Mean projected bytes per row group across `splits`, from their footer sizes.

    The footer records a row group's whole uncompressed size, not a column's, so the
    projection is charged by its share of the columns. That is an estimate — a string
    column outweighs a flag — but the budget it feeds is a throughput knob with a memory
    ceiling, not a correctness bound, and within a factor of two is what it needs.
    """
    sized = [(s.nbytes, len(s.row_groups)) for s in splits if getattr(s, "nbytes", None)]
    if not sized:
        return None
    per_rg = sum(n for n, _ in sized) / max(1, sum(k for _, k in sized))
    if cols is None:
        return per_rg
    try:
        width = len(splits[0].schema())
    except Exception as exc:
        note_suppressed("io", "read split schema width", exc)
        return per_rg
    return per_rg * min(1.0, max(1, len(cols)) / max(1, width))


def ordered_concurrent(units, read_one, depth: int, on_error=None):
    """Yield ``read_one(unit)`` for each of `units` **in order**, `depth` reads in flight.

    The one definition of "read ahead on a thread pool without reordering" this module has,
    shared by the per-split pyarrow reader and the native row-group reader. Both are
    object-store-LATENCY-bound — a single connection sits far below a node's bandwidth and
    each request waits tens of milliseconds — so what caps throughput is how many requests
    are outstanding, not how fast any one of them is.

    A FIFO of futures is what keeps the order: a unit is submitted early but only yielded
    when the reader reaches it, so a caller that assumes file/split order is unaffected.
    Memory is bounded to at most `depth` in-flight reads, and the next read is submitted
    *before* the current one is drained so a skipped unit still advances the window.

    `read_one` must release the GIL for the overlap to be real. Both callers do:
    ``bc_py::read_parquet`` wraps its object-store fetch in ``py.allow_threads``, and
    PyArrow's readers release it too.

    Args:
        units: The work items to read, in the order their results must be yielded.
        read_one: Called on each unit; its return value is yielded.
        depth: Reads to keep in flight; ``<= 1`` (or a single unit) runs plain sequentially.
        on_error: Called as ``on_error(unit, exc)`` when a read raises. Returning True
            skips that unit; returning False (the default when omitted) re-raises.

    Yields:
        Each unit's `read_one` result, in `units` order.
    """
    if depth <= 1 or len(units) <= 1:
        for unit in units:
            try:
                out = read_one(unit)
            except Exception as exc:
                if on_error is not None and on_error(unit, exc):
                    continue
                raise
            yield out
        return

    import collections
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=depth) as pool:
        pending: collections.deque = collections.deque()
        it = iter(units)
        for unit in _take(it, depth):
            pending.append((unit, pool.submit(read_one, unit)))
        while pending:
            unit, fut = pending.popleft()
            # Submit the next read BEFORE draining this one so a failed unit still
            # advances the prefetch window (keeps the pipeline full under skip).
            nxt = next(it, None)
            if nxt is not None:
                pending.append((nxt, pool.submit(read_one, nxt)))
            try:
                out = fut.result()  # raises if the read failed
            except Exception as exc:
                if on_error is not None and on_error(unit, exc):
                    continue
                raise
            yield out


def _take(it, n: int):
    """The next ≤`n` items of `it` (priming the prefetch window)."""
    out = []
    for _ in range(n):
        x = next(it, None)
        if x is None:
            break
        out.append(x)
    return out
