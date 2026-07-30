"""Observed per-source I/O throughput — measured on read, captured for prediction.

Core measures the wall time and (decoded) byte volume of each source read and records a
smoothed throughput (MB/s) here, keyed by the source's stable `identity()`. A later read of
the same source can then *predict* its read cost — the signal that turns the documented
small-files scan pathology from an unpredictable stall into a sized fan-out. Best-effort
throughout: a write never breaks a read, and a cold store simply yields `None`.

This is metadata capture, not a scheduling decision — it records what the hardware did, and
`relative_read_cost` is how the optimizer spends it: the cost model prices a scanned byte the
same everywhere, which is one coefficient fitted across a cold object-store read and a
page-cache hit at once, so a plan joining one of each is ranked as though reading either were
equally cheap. The figure is decoded bytes per wall-second (compression folds into it
consistently per source), smoothed across runs so one noisy read doesn't jerk it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.metadata.hardware_scope import scoped
from batcher.metadata.hub import MetadataHub
from batcher.metadata.smoothed import load_scalar, record_smoothed_scalar

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = [
    "load_source_throughput_mbps",
    "predicted_read_seconds",
    "record_source_io",
    "relative_read_cost",
    "scanned_byte_count",
]

# Scoped to the hardware fingerprint, because MB/s is a machine-unit measurement in the exact
# sense `hardware_scope` defines: it is the NIC, the disk, the page cache, and the cores that
# decompress — none of which transfers. Unscoped, a heterogeneous Ray cluster blends the
# driver's local-NVMe read of a path with a worker's cold S3 read of the same path into one
# figure that is wrong for both, and an autoscaling group that mixes instance generations does
# the same more slowly. The *source identity* stays the key: which data it is does transfer,
# and splitting on it too would fragment the statistic pointlessly.
#
# `hardware_scope` also explains why nothing is migrated across the rename: a value learned
# under the unscoped name is a blend of unknown provenance, and adopting it into whichever
# machine class asks first would reinstate exactly the failure being removed. Throughput
# re-converges after one read per source.
_NAMESPACE = "io.throughput_mbps"

# Byte counts already measured, keyed by (source identity, projection, rows). Bounded because
# a long-lived session reads many distinct sources; the key set is small and hot in practice
# (a served query re-reads the same source with the same projection), so a plain FIFO clear at
# the ceiling costs a rebuild of a handful of entries and never grows without bound.
_BYTES_MEMO: dict[tuple[str, str, int], int] = {}
_BYTES_MEMO_MAX = 512

# Floors below which a read is not a throughput measurement at all.
#
# Throughput is bytes over time, and the time of a small read is dominated by everything that
# is not byte movement: resolving the source, opening a handle or issuing a request, crossing
# FFI, allocating the first batch. Dividing a tiny byte count by that fixed cost does not
# produce a noisy bandwidth, it produces an *inverted* one — the fewer bytes, the slower the
# source appears — so a small dimension table reads as the most expensive thing in the query.
#
# The byte floor is set where sequential movement is the clear majority of the work for any
# storage class the engine reads from, and the time floor exists because a sub-millisecond
# interval is at the resolution limit of what the surrounding Python timing can resolve.
_MIN_MEASURED_BYTES = 8 * 1024 * 1024
_MIN_MEASURED_MS = 1.0


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

    A read too small to *be* a throughput measurement is dropped rather than recorded. Below
    the floors, the elapsed time is the fixed cost of opening and dispatching the read, not the
    cost of moving the bytes, so the ratio measures overhead and calls it bandwidth. That is
    not a noisy signal, it is an inverted one: a 25-row dimension table takes about as long to
    read as a 25,000-row one, so the smaller the relation the *slower* it appears — and a
    consumer ranking sources by cost would conclude that the tiny table is the expensive one.
    Measured on TPC-H sf1, where `nation` (25 rows) and `region` (5 rows) came out sixteen
    times more expensive per byte than `part`.

    Best-effort and non-blocking: a bad measurement or any failure is dropped, never raised
    into the read path.
    """
    if not identity or byte_count < _MIN_MEASURED_BYTES or elapsed_ms < _MIN_MEASURED_MS:
        return
    mbps = (byte_count / (1024 * 1024)) / (elapsed_ms / 1000.0)
    record_smoothed_scalar(hub, scoped(_NAMESPACE), identity, mbps)


def load_source_throughput_mbps(hub: MetadataHub | None, identity: str) -> float | None:
    """The learned read throughput (MB/s) for source `identity`, or `None` (cold/unavailable)."""
    if not identity:
        return None
    return load_scalar(hub, scoped(_NAMESPACE), identity)


def predicted_read_seconds(hub: MetadataHub | None, identity: str, byte_count: int) -> float | None:
    """Predicted wall time to read `byte_count` bytes of source `identity`, from its learned
    throughput — the "predict" half: turn measured MB/s + a known byte size into an expected
    read cost the optimizer/`explain` can reason about *before* running. `None` when the
    source's throughput was never measured (cold) or the byte count is non-positive."""
    mbps = load_source_throughput_mbps(hub, identity)
    if mbps is None or mbps <= 0.0 or byte_count <= 0:
        return None
    return (byte_count / (1024 * 1024)) / mbps


# How far a single source's read cost may be moved relative to the plan's median source.
# The real spread is wider than this — a cold object-store read of many small files against a
# page-cache hit is two orders of magnitude — but the clamp is not trying to represent that
# spread. It is bounding how much *one* learned number, smoothed from a handful of reads that
# may have been unlucky, may re-rank a plan. Four-fold in either direction is enough to move a
# join order or a build side, which is the decision this exists to inform, and not enough for
# a mismeasured source to dominate the model.
_READ_COST_CLAMP = 4.0

# Below this many *measured* sources in a plan there is no relative information to extract, so
# every source keeps the reference cost and the ranking is byte-for-byte what it was. That
# covers the single-source plan, which is most of them.
_MIN_MEASURED_SOURCES = 2

# How far from the median a source's throughput must sit before it is priced differently at
# all. Inside the band the factor is exactly 1.0.
#
# The band is the difference between spending a measurement and spending noise. Two relations
# on the same storage do not read at identical MB/s — compression ratio, column widths, page
# cache warmth, and where the scheduler happened to run all move the figure — and none of that
# is a reason to re-rank a plan. What the band leaves through is the case this exists for: a
# cold object store against a local file, a network volume against NVMe, a spinning disk
# against anything. Those are five-fold and up, not fifty percent.
_DEAD_BAND = 1.5


def relative_read_cost(hub: MetadataHub | None, identities: list[str]) -> list[float]:
    """Per-source multipliers on the cost of a scanned byte, relative to this plan's median.

    The cost model prices every scanned byte the same. That is one coefficient fitted across
    two populations that differ by orders of magnitude — a cold object-store read of many
    small files, and a page-cache hit on a local file — and a plan that joins one of each is
    ranked as though reading either were equally cheap. Which side to build, which to
    broadcast, and which order to join in all turn on that comparison.

    Core already measures the answer: `record_source_io` records each source's observed MB/s.
    This turns those figures into what a cost model can actually use, and the choice of
    baseline is what makes it safe. The multipliers are relative to the **median measured
    source in this plan**, not to an absolute reference:

    * A cost model ranks alternatives, so only the *ratio* between this plan's sources can
      change a decision. An absolute anchor would additionally re-scale the `io` axis against
      `cpu` — a re-tuning of the model, not a sharpening of it, and one whose correct value
      depends on a coefficient calibrated on a different machine.
    * It cannot drift. A fleet that gets uniformly faster moves every source together and
      every multiplier stays at 1.0, which is right: nothing about the *comparison* changed.
    * A plan with fewer than `_MIN_MEASURED_SOURCES` measured sources has no ratio to read, so
      it gets all-1.0 and is ranked exactly as before.

    An unmeasured source in a plan that does have measured ones is given 1.0 — the reference,
    not a guess. Charging it as slow would punish a source for being new, and charging it as
    fast would reward the same thing.

    Args:
        hub: The metadata hub, or `None` when learning is off.
        identities: Each bound source's stable identity, positionally. An empty string means
            the source cannot be identified, so nothing can be learned about it.

    Returns:
        One multiplier per entry of `identities`, each in
        ``[1 / _READ_COST_CLAMP, _READ_COST_CLAMP]``. All 1.0 when nothing is learnable, which
        leaves every existing ranking unchanged.
    """
    if hub is None or len(identities) < _MIN_MEASURED_SOURCES:
        return [1.0] * len(identities)
    measured = [load_source_throughput_mbps(hub, ident) for ident in identities]
    known = sorted(m for m in measured if m is not None and m > 0.0)
    if len(known) < _MIN_MEASURED_SOURCES:
        return [1.0] * len(identities)
    middle = len(known) // 2
    median = known[middle] if len(known) % 2 else (known[middle - 1] + known[middle]) / 2.0
    if median <= 0.0:  # pragma: no cover - `known` holds only positives
        return [1.0] * len(identities)
    return [1.0 if m is None or m <= 0.0 else _factor(median / m) for m in measured]


def _factor(ratio: float) -> float:
    """A raw median-to-source throughput ratio as a priced multiplier.

    Inside `_DEAD_BAND` the answer is exactly 1.0, so two relations on the same storage — whose
    measured MB/s differ by compression, column width, cache warmth, and scheduling luck — are
    priced the same and no plan moves. Outside it the ratio passes through, clamped, so the
    case the measurement exists for still reaches the cost model.
    """
    if 1.0 / _DEAD_BAND <= ratio <= _DEAD_BAND:
        return 1.0
    return min(_READ_COST_CLAMP, max(1.0 / _READ_COST_CLAMP, ratio))
