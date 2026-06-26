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

from batcher.config import active_config

__all__ = [
    "column_statistics",
    "heavy_hitters",
    "tail_quantiles",
    "tdigest_partial",
    "tdigest_quantile",
]


def column_statistics(
    batches: list[pa.RecordBatch],
    columns: list[str],
    probs: tuple[float, ...] | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, list[float]]], dict[str, float]]:
    """Measure per-column ndv, quantile boundaries, and average byte width.

    Returns `(ndv, quantiles, avg_bytes)` where `ndv` is `{col: distinct_estimate}`,
    `quantiles` is `{col: {"probs": [...], "values": [...]}}` (only for numeric
    columns with a full set of boundaries), and `avg_bytes` is `{col: bytes/row}`
    (the measured per-row width that turns Kyber's memory/broadcast estimates
    byte-true). The quantile grid defaults to `OptimizerConfig.quantile_probs` (a
    coarse min/quartiles/max grid, enough to interpolate `fraction <= literal`
    without a full histogram). Best-effort: returns empty dicts if the native
    engine is unavailable or the inputs are empty.
    """
    if probs is None:
        probs = active_config().optimizer.quantile_probs
    if not batches or not columns:
        return {}, {}, {}
    try:
        import batcher._native as _native

        stats = _native.column_stats(list(columns), batches)
        quants = _native.column_quantiles(list(columns), batches, list(probs))
    except Exception:  # pragma: no cover - measurement must never break execution
        return {}, {}, {}

    ndv = {c: d["ndv"] for c, d in stats.items() if d.get("ndv") is not None}
    quantiles = {c: {"probs": list(probs), "values": vals} for c, vals in quants.items() if vals}
    avg_bytes = {c: d["avg_bytes"] for c, d in stats.items() if d.get("avg_bytes") is not None}
    return ndv, quantiles, avg_bytes


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
        import batcher._native as _native

        out = _native.tail_quantiles(list(columns), batches, list(probs))
    except Exception:  # pragma: no cover - measurement must never break execution
        return {}
    return {c: v for c, v in out.items() if v}


def tdigest_partial(batches: list[pa.RecordBatch], column: str) -> bytes | None:
    """Build a serialized TDigest over `column` — the partial step of a mergeable
    approximate quantile. None for a non-numeric/empty column. Paired with
    `tdigest_quantile`, so a quantile streams chunk-by-chunk with no full collect."""
    if not batches:
        return None
    try:
        import batcher._native as _native

        return _native.tdigest_partial(column, batches)
    except Exception:  # pragma: no cover - measurement must never break execution
        return None


def tdigest_quantile(sketches: list[bytes], q: float) -> float | None:
    """Merge serialized TDigest `sketches` and return the value at quantile `q` (the
    combine+finalize step). None if no sketch carried data."""
    if not sketches:
        return None
    try:
        import batcher._native as _native

        return _native.tdigest_quantile(list(sketches), float(q))
    except Exception:  # pragma: no cover - measurement must never break execution
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
        import batcher._native as _native

        out = _native.heavy_hitters(list(columns), batches, float(fraction))
    except Exception:  # pragma: no cover - measurement must never break execution
        return {}
    return {c: [(v, int(n)) for v, n in hits] for c, hits in out.items() if hits}
