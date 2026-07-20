"""Distributions and time series over many runs — percentiles and session throughput.

Split from the rest of the analysis because these two are the only pieces that care about
the *shape* of a set of numbers rather than about what the numbers mean.

**Percentiles are reported honestly.** With fewer than a handful of samples a "p95" is the
largest value wearing a lab coat, so `percentiles` reports how many samples it had and the
UI says "3 runs" rather than implying a distribution it cannot see.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

__all__ = ["percentiles", "throughput_series"]

#: Below this many samples a percentile is not meaningfully different from the max.
_MIN_FOR_PERCENTILES = 5
#: Session timeline resolution. 60 buckets keeps the chart readable at any session length.
_SERIES_BUCKETS = 60


def percentiles(values: Iterable[float]) -> dict[str, Any]:
    """p50/p90/p95/p99 plus min, max, mean and the sample count.

    Uses nearest-rank on the sorted sample — no interpolation. Interpolating between two
    observed durations invents a duration that never happened, which is the wrong trade for
    a panel whose whole job is to report measurements.

    Args:
        values: Durations (or any comparable magnitudes).

    Returns:
        A dict of statistics; ``{"count": 0}`` when there is nothing to describe.
    """
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return {"count": 0}
    out: dict[str, Any] = {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "reliable": len(ordered) >= _MIN_FOR_PERCENTILES,
    }
    for label, q in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)):
        rank = max(0, math.ceil(q * len(ordered)) - 1)
        out[label] = ordered[rank]
    return out


def throughput_series(
    queries: list[dict[str, Any]], buckets: int = _SERIES_BUCKETS
) -> dict[str, Any]:
    """Rows/second and run counts over the session, bucketed for a sparkline.

    Buckets are sized from the session's own span rather than a fixed interval, so a
    ten-second session and a ten-hour one both fill the chart. Empty buckets are kept: a gap
    where nothing ran is information, and dropping them would draw a continuous line through
    a period of silence.

    Args:
        queries: Run summaries, newest first (the store's natural order).
        buckets: How many time buckets to produce.

    Returns:
        ``{"buckets": [...], "start": float, "end": float, "bucket_s": float}``.
    """
    finished = [q for q in queries if q.get("status") != "running" and q.get("started_wall")]
    if not finished:
        return {"buckets": [], "start": 0.0, "end": 0.0, "bucket_s": 0.0}
    start = min(q["started_wall"] for q in finished)
    end = max(q["started_wall"] + (q.get("total_ms", 0.0) / 1000.0) for q in finished)
    span = max(end - start, 1e-6)
    width = span / buckets
    slots: list[dict[str, Any]] = [
        {"t": start + i * width, "runs": 0, "rows": 0, "ms": 0.0, "failed": 0}
        for i in range(buckets)
    ]
    for q in finished:
        index = min(buckets - 1, int((q["started_wall"] - start) / width))
        slot = slots[index]
        slot["runs"] += 1
        slot["rows"] += int(q.get("rows", 0))
        slot["ms"] += float(q.get("total_ms", 0.0))
        if q.get("status") == "error":
            slot["failed"] += 1
    for slot in slots:
        slot["rows_per_sec"] = (slot["rows"] / (slot["ms"] / 1000.0)) if slot["ms"] > 0 else 0.0
    return {"buckets": slots, "start": start, "end": end, "bucket_s": width}
