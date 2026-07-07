"""Post-execution column-statistics learning (Core measures, Kyber persists).

After a query runs, the engine has the base sources' batches in hand; measuring each
column's ndv / quantiles / average byte width / most-common-values once and recording
them into the `MetadataHub` is what lets the *next* run's optimizer size joins,
aggregations, and broadcasts from learned numbers instead of Selinger guesses. The
work is gated to columns not already known, so a column's O(rows) sketch build happens
at most once. Best-effort throughout: a measurement failure never affects a result.

Extracted from `orchestration` along the measurement seam to keep the conductor module
within the size budget; the conductor calls these on its single-node and UDF paths.
"""

from __future__ import annotations

import pyarrow as pa

from batcher.config import active_config
from batcher.io.source import Source, iter_source

__all__ = ["collect_source_metadata", "learn_column_stats"]

# The MetadataHub key under which learned per-column distinct counts live.
_NDV_KEY = "__column_ndv__"


# Row cap for the driver-side column-stat sample (≈ a couple of Parquet row-groups).
# Enough for usable ndv/quantile sketches; small enough that learning never re-scans a
# large input on the driver.
_STATS_SAMPLE_ROWS = 1 << 18


def _stats_sample(src: Source) -> list[pa.RecordBatch]:
    """A bounded row sample of `src` for column-stat learning — NOT the whole source.

    `collect_source_metadata` runs on the driver after a query; on the distributed/UDF
    paths it has no scanned batches in hand and so samples the base source here. Reading
    the *whole* source would re-scan it single-node on the driver — which on a large
    distributed input dwarfs the query itself (sf100: a 140 GB driver re-read that hung
    for minutes *after* a ~60 s distributed agg). The actual cardinalities of that run are
    already learned from the worker metrics; these column sketches are an approximate
    prior the estimator refines across runs, so a bounded sample is the right trade.
    `iter_source` is lazy, so this stops after the first batches past the row cap (an
    in-memory source is already resident and small)."""
    out: list[pa.RecordBatch] = []
    n = 0
    for b in iter_source(src, None, None):
        out.append(b)
        n += b.num_rows
        if n >= _STATS_SAMPLE_ROWS:
            break
    return out


def collect_source_metadata(hub, sources: list[Source]) -> None:
    """Record per-column ndv/quantiles from the base sources (Core collects).

    The UDF and distributed paths don't surface their scanned batches the way the
    native path hands `resolved` to `learn_column_stats`, so this samples the base
    sources directly (see `_stats_sample` — bounded, never a whole-source driver scan).
    It is gated on the cheap `Source.schema` — a source is only read when it has a
    not-yet-measured column — so a file is never re-scanned once its columns are
    learned. Best-effort: learning never breaks a query.
    """
    if hub is None:
        return
    from batcher import kyber

    try:
        known = set(kyber.load_learned_stats(hub).get(_NDV_KEY, {}))
        resolved = [
            _stats_sample(src)
            for src in sources
            if any(c not in known for c in src.schema().names)
        ]
        if resolved:
            learn_column_stats(hub, resolved)
    except Exception:  # pragma: no cover - learning must never break execution
        pass


def learn_column_stats(hub, resolved: list[list[pa.RecordBatch]]) -> None:
    """Measure per-column ndv/quantiles from the just-scanned input and record them.

    Gated to columns not already known, so the O(rows) sketch build happens at most
    once per column — a bounded, one-time cost that sharpens every later plan. Core
    measures (`core.column_statistics`); Kyber persists/consumes. Best-effort: a
    failure here never affects the query result.
    """
    if hub is None:
        return
    from batcher import core, kyber

    try:
        known = set(kyber.load_learned_stats(hub).get(_NDV_KEY, {}))
        min_frac = active_config().optimizer.cardinality.mcv_min_fraction
        ndv_all: dict[str, float] = {}
        quant_all: dict[str, dict[str, list[float]]] = {}
        bytes_all: dict[str, float] = {}
        mcv_all: dict[str, dict[str, float]] = {}
        for batches in resolved:
            if not batches:
                continue
            cols = [c for c in batches[0].schema.names if c not in known]
            if not cols:
                continue
            ndv, quants, avg_bytes = core.column_statistics(batches, cols)
            ndv_all.update(ndv)
            quant_all.update(quants)
            bytes_all.update(avg_bytes)
            total = sum(b.num_rows for b in batches)
            # MCV clears `min_frac` only on low-cardinality columns (ndv ≲ 1/min_frac);
            # skip the per-row Misra-Gries scan on keys/high-ndv columns (always empty).
            mcv_cols = [c for c in cols if ndv.get(c, 1e18) <= 1.0 / min_frac]
            for col_name, hits in core.heavy_hitters(batches, mcv_cols, min_frac).items():
                if total > 0 and hits:
                    mcv_all[col_name] = {str(v): n / total for v, n in hits}
        if ndv_all or quant_all or bytes_all or mcv_all:
            kyber.record_column_stats(hub, ndv_all, quant_all, bytes_all, mcv_all)
    except Exception:  # pragma: no cover - learning must never break execution
        pass
