"""Column-statistics measurement — Core's lane.

The optimizer wants per-column distinct counts (to sharpen equality selectivity to
`1/ndv` and join cardinality) and quantile boundaries (for range selectivity). Those
are *measured* from data, which is Core's job, not Kyber's — so this is the one place
that reads values, and it does so in Rust (`bc-sketches` via the native engine), never
by touching tuples in Python. The mergeable `ColumnStats` sketch is built over the
already-in-memory scan batches (no extra I/O); Kyber persists and consumes the result
through the MetadataHub.
"""

from __future__ import annotations

import pyarrow as pa

from batcher._internal.logging import note_suppressed
from batcher._internal.native import engine
from batcher.config import active_config
from batcher.plan.stats import arrow_ordinal_axis

__all__ = [
    "column_ndv",
    "column_statistics",
    "heavy_hitters",
    "tail_quantiles",
    "tdigest_partial",
    "tdigest_quantile",
]


#: Rows below which splitting a batch for the sketch costs more than the parallelism buys.
#: The same figure `bc_interp::stream::parallel::MIN_ROWS_TO_SHARD` uses for the identical
#: decision on the execution side — four morsels — expressed in the control plane's own units
#: so it tracks a retuned morsel rather than drifting from it.
_MIN_SKETCH_SHARD_MORSELS = 4


def _sketch_shards(batches: list[pa.RecordBatch]) -> list[pa.RecordBatch]:
    """`batches` re-sliced so the native sketch has a piece per usable core.

    Every sketch below parallelizes **across batches**, which makes the batch count — not
    the row count — decide how much of the machine measures a column. That is invisible
    until a caller hands over one big batch, and the most natural way to give Batcher data
    does exactly that: `pa.table(...)` built from NumPy arrays, anything through
    `combine_chunks()`, and every `Table` PyArrow hands back after a `to_table()` are all a
    single chunk per column.

    Measured on 100M `int64` rows, one column, sketching distinct counts:

    | batches | seconds | ns/cell |
    |--------:|--------:|--------:|
    |       1 |   0.553 |    5.53 |
    |       2 |   0.288 |    2.88 |
    |       8 |   0.107 |    1.07 |
    |      32 |   0.032 |    0.32 |
    |      96 |   0.031 |    0.31 |

    So a single-batch column is sketched **17x** slower than the same rows in 32 pieces, and
    the multi-batch figure is the ~0.4 ns/cell the rest of the engine's sizing assumes —
    `optimizer.ndv_sketch_max_cells` is a *cell* budget, and it was set against the batched
    rate. On one batch the same ceiling admits fourteen times the wall-clock it was sized
    for, which is how a 200M-row in-memory query came to spend most of its time in the
    planner rather than in the engine.

    `RecordBatch.slice` is O(1) and shares buffers, so this costs a few objects and no data
    movement. Batches already numerous enough are returned untouched, and a relation too
    small to be worth splitting keeps its shape — the parallelism is not free below a few
    morsels, which is the same trade the execution side makes in `MIN_ROWS_TO_SHARD`.

    Args:
        batches: The batches about to be sketched.

    Returns:
        The same rows, in enough pieces to occupy the machine.
    """
    from batcher._internal.hardware import available_cpu_count

    target = available_cpu_count()
    if len(batches) >= target:
        return batches
    floor = max(1, active_config().execution.morsel_rows * _MIN_SKETCH_SHARD_MORSELS)
    out: list[pa.RecordBatch] = []
    for b in batches:
        rows = b.num_rows
        # How many ways this batch alone should go, bounded by the whole target so a single
        # huge batch does not produce more pieces than there are cores to sketch them on.
        pieces = min(target, max(1, rows // floor))
        if pieces <= 1:
            out.append(b)
            continue
        per = -(-rows // pieces)  # ceil, so the last piece is the short one
        out.extend(b.slice(off, min(per, rows - off)) for off in range(0, rows, per))
    return out


def column_ndv(batches: list[pa.RecordBatch], columns: list[str]) -> dict[str, float]:
    """Measure each column's distinct-count estimate (HLL), in parallel across batches.

    The distinct-count-only counterpart to `column_statistics`, for the one statistic
    the optimizer cannot get from a file footer: a Parquet footer carries row counts,
    null counts, and min/max, but no `ndv`. Skipping the quantile sketch makes this
    roughly seven times cheaper per row, so a source's join-key distinct counts can be
    seeded on the query path rather than only after a run has been measured.

    Best-effort: returns an empty dict if the native engine is unavailable or the inputs
    are empty. Estimates are approximate (HyperLogLog, ~1% relative error) and must never
    be labelled `EXACT`.

    Args:
        batches: Arrow batches to sketch. Never iterated row-wise in Python.
        columns: Column names to measure. A name absent from `batches` is omitted.

    Returns:
        Column name to its estimated distinct count.
    """
    if not batches or not columns:
        return {}
    try:
        _native = engine()
        return _native.column_ndv(list(columns), _sketch_shards(batches))
    except Exception as exc:  # pragma: no cover - measurement must never break a query
        note_suppressed("core", "measure column ndv", exc)
        return {}


def column_statistics(
    batches: list[pa.RecordBatch],
    columns: list[str],
    probs: tuple[float, ...] | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, list[float]]], dict[str, float]]:
    """Measure per-column ndv, quantile boundaries, and average byte width.

    Returns `(ndv, quantiles, avg_bytes)` where `ndv` is `{col: distinct_estimate}`,
    `quantiles` is `{col: {"probs": [...], "values": [...], "axis": ...}}` (only for
    ordered columns with a full set of boundaries), and `avg_bytes` is `{col: bytes/row}`
    (the measured per-row width that turns Kyber's memory/broadcast estimates
    byte-true). The quantile grid defaults to `OptimizerConfig.quantile_probs` (a
    coarse min/quartiles/max grid, enough to interpolate `fraction <= literal`
    without a full histogram). Best-effort: returns empty dicts if the native
    engine is unavailable or the inputs are empty.

    The native sketch measures a column in its **storage** unit — epoch days for a
    `date32`, epoch microseconds for a `timestamp[us]` — while the estimator consults the
    grid with a Python `date`/`datetime` from the predicate. Each grid is therefore
    converted onto the shared axis named by `plan.stats.arrow_ordinal_axis` and carries
    that axis, so a consumer can only read it against a literal on the same number line.
    """
    if probs is None:
        probs = active_config().optimizer.quantile_probs
    if not batches or not columns:
        return {}, {}, {}
    try:
        _native = engine()
        # One sketch pass for both summary stats and quantiles (the native side builds
        # each column's HLL+KLL once), instead of two FFI calls that each rebuilt it.
        stats, quants = _native.column_stats_full(
            list(columns), _sketch_shards(batches), list(probs)
        )
    except Exception as exc:  # pragma: no cover - measurement must never break execution
        note_suppressed("core", "measure column statistics", exc)
        return {}, {}, {}

    ndv = {c: d["ndv"] for c, d in stats.items() if d.get("ndv") is not None}
    quantiles = _quantiles_on_axis(quants, batches[0].schema, probs)
    avg_bytes = {c: d["avg_bytes"] for c, d in stats.items() if d.get("avg_bytes") is not None}
    return ndv, quantiles, avg_bytes


def _quantiles_on_axis(
    quants: dict[str, list[float]], schema: pa.Schema, probs: tuple[float, ...]
) -> dict[str, dict[str, list[float] | str]]:
    """Move each measured grid onto its column's shared ordinal axis and label it.

    A column whose type has no linear order (and so no axis) is dropped rather than
    recorded on an axis no literal can name.
    """
    out: dict[str, dict[str, list[float] | str]] = {}
    for col, values in quants.items():
        if not values:
            continue
        field = schema.field(col) if col in schema.names else None
        axis = arrow_ordinal_axis(field.type) if field is not None else None
        if axis is None:
            continue
        name, divisor = axis
        scaled = list(values) if divisor == 1.0 else [v / divisor for v in values]
        out[col] = {"probs": list(probs), "values": scaled, "axis": name}
    return out


def tail_quantiles(
    batches: list[pa.RecordBatch], columns: list[str], probs: tuple[float, ...]
) -> dict[str, list[float]]:
    """Measure tail-accurate quantiles (TDigest) for numeric `columns`.

    Returns `{col: [value at each prob]}`; non-numeric/empty columns are omitted.
    Where `column_statistics` builds a coarse KLL grid for selectivity, this is
    accurate in the tails — what an `approx_quantile` answer needs. Best-effort:
    empty dict if the native engine is unavailable or inputs are empty.
    """
    if not batches or not columns:
        return {}
    try:
        _native = engine()
        out = _native.tail_quantiles(list(columns), _sketch_shards(batches), list(probs))
    except Exception as exc:  # pragma: no cover - measurement must never break execution
        note_suppressed("core", "measure tail quantiles", exc)
        return {}
    return {c: v for c, v in out.items() if v}


def tdigest_partial(batches: list[pa.RecordBatch], column: str) -> bytes | None:
    """Build a serialized TDigest over `column` — the partial step of a mergeable
    approximate quantile. None for a non-numeric/empty column. Paired with
    `tdigest_quantile`, so a quantile streams chunk-by-chunk with no full collect."""
    if not batches:
        return None
    try:
        _native = engine()
        return _native.tdigest_partial(column, _sketch_shards(batches))
    except Exception as exc:  # pragma: no cover - measurement must never break execution
        note_suppressed("core", "build a t-digest partial", exc)
        return None


def tdigest_quantile(sketches: list[bytes], q: float) -> float | None:
    """Merge serialized TDigest `sketches` and return the value at quantile `q` (the
    combine+finalize step). None if no sketch carried data."""
    if not sketches:
        return None
    try:
        _native = engine()
        return _native.tdigest_quantile(list(sketches), float(q))
    except Exception as exc:  # pragma: no cover - measurement must never break execution
        note_suppressed("core", "read a t-digest quantile", exc)
        return None


def heavy_hitters(
    batches: list[pa.RecordBatch], columns: list[str], fraction: float
) -> dict[str, list[tuple[str, int]]]:
    """Measure heavy hitters (Misra-Gries) for `columns` — the skew signal.

    Returns `{col: [(value, estimated_count), ...]}` for values exceeding
    `fraction` of the rows. Kyber consumes this for skew-aware decisions (a hot
    join key → salting). Best-effort: empty dict on any failure.
    """
    if not batches or not columns:
        return {}
    try:
        _native = engine()
        out = _native.heavy_hitters(list(columns), _sketch_shards(batches), float(fraction))
    except Exception as exc:  # pragma: no cover - measurement must never break execution
        note_suppressed("core", "measure heavy hitters", exc)
        return {}
    return {c: [(v, int(n)) for v, n in hits] for c, hits in out.items() if hits}
