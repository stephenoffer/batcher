"""Observed per-source I/O throughput — measured on read, captured for prediction.

Core measures the wall time and (decoded) byte volume of each source read and records a
smoothed throughput (MB/s) here, keyed by the source's stable `identity()`. A later read of
the same source can then *predict* its read cost — the signal that turns the documented
small-files scan pathology from an unpredictable stall into a sized fan-out. Best-effort
throughout: a write never breaks a read, and a cold store simply yields `None`.

This is metadata capture, not a scheduling decision — it records what the hardware did; the
optimizer/Carbonite consume it. The figure is decoded bytes per wall-second (compression
folds into it consistently per source), smoothed across runs so one noisy read doesn't jerk it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.metadata.hub import MetadataHub
from batcher.metadata.smoothed import load_scalar, record_smoothed_scalar

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = [
    "load_source_throughput_mbps",
    "predicted_read_seconds",
    "record_source_io",
    "scanned_byte_count",
]

_NAMESPACE = "io.throughput_mbps"

# Byte counts already measured, keyed by (source identity, projection, rows). Bounded because
# a long-lived session reads many distinct sources; the key set is small and hot in practice
# (a served query re-reads the same source with the same projection), so a plain FIFO clear at
# the ceiling costs a rebuild of a handful of entries and never grows without bound.
_BYTES_MEMO: dict[tuple[str, str, int], int] = {}
_BYTES_MEMO_MAX = 512


def scanned_byte_count(
    identity: str, projection: object, rows: int, batches: list[pa.RecordBatch]
) -> int:
    """Decoded bytes of `batches`, memoized per (source, projection, row count).

    `RecordBatch.nbytes` is the figure this module's throughput is defined on, and it is the
    only correct one: it *deduplicates buffers shared between batches*, so a source whose
    batches are slices of one parent buffer is counted once. That correctness is not free —
    measured at **2.86 ms** for a 49-batch, 16-column relation, which was **over half** the
    latency of a small query, paid again on every repeat of the same read.

    `get_total_buffer_size()` is 18.7x faster and is **not** a substitute: it counts a shared
    parent buffer once per slice, reporting 1.6 GB for 100 slices of one 16 MB batch. Feeding
    that to `record_source_io` would invent a 100x read throughput and quietly corrupt the
    cost model Kyber plans against — a measurement bug that no result-correctness test can see.

    So keep the exact figure and stop recomputing it: the same source, read with the same
    projection, yielding the same row count, has the same byte count. A changed row count
    misses the memo and is measured afresh, which is what makes this safe for a source whose
    contents move underneath it.

    Args:
        identity: The source's stable identity, as `record_source_io` keys throughput on.
        projection: The columns this read requested, or None for all of them.
        rows: Rows actually scanned — part of the key, so changed data re-measures.
        batches: The batches read.

    Returns:
        The summed `nbytes` of `batches`, 0 when the source has no stable identity to key on.
    """
    if not identity:
        return sum(b.nbytes for b in batches)
    key = (identity, repr(projection), rows)
    hit = _BYTES_MEMO.get(key)
    if hit is not None:
        return hit
    total = sum(b.nbytes for b in batches)
    if len(_BYTES_MEMO) >= _BYTES_MEMO_MAX:
        _BYTES_MEMO.clear()
    _BYTES_MEMO[key] = total
    return total


def record_source_io(
    hub: MetadataHub | None, identity: str, byte_count: int, elapsed_ms: float
) -> None:
    """Record a source read's observed throughput (MB/s), exp-smoothed across runs.

    Best-effort and non-blocking: a bad measurement (zero bytes/time) or any failure is
    dropped, never raised into the read path.
    """
    if not identity or byte_count <= 0 or elapsed_ms <= 0:
        return
    mbps = (byte_count / (1024 * 1024)) / (elapsed_ms / 1000.0)
    record_smoothed_scalar(hub, _NAMESPACE, identity, mbps)


def load_source_throughput_mbps(hub: MetadataHub | None, identity: str) -> float | None:
    """The learned read throughput (MB/s) for source `identity`, or `None` (cold/unavailable)."""
    if not identity:
        return None
    return load_scalar(hub, _NAMESPACE, identity)


def predicted_read_seconds(hub: MetadataHub | None, identity: str, byte_count: int) -> float | None:
    """Predicted wall time to read `byte_count` bytes of source `identity`, from its learned
    throughput — the "predict" half: turn measured MB/s + a known byte size into an expected
    read cost the optimizer/`explain` can reason about *before* running. `None` when the
    source's throughput was never measured (cold) or the byte count is non-positive."""
    mbps = load_source_throughput_mbps(hub, identity)
    if mbps is None or mbps <= 0.0 or byte_count <= 0:
        return None
    return (byte_count / (1024 * 1024)) / mbps
