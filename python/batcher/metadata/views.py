"""The bounded derived views over the feedback history.

Four consumers want an *aggregate* over everything Core has measured — Kyber's cost
calibration, its cardinality correction, Carbonite's learned memory model, and the CPU-share
loop — and each wants a different shape of it. Building those shapes is a separate job from
the hub's own, which is being the façade over a backend, so it lives here.

Two views, and the difference between them is the whole point:

* **by kind** buckets rows by operator family for the models fitted in *machine units* —
  nanoseconds per row, bytes per group, utilization. It is restricted to rows measured on this
  machine class, because none of those quantities transfers to different hardware.
* **with signature** keeps rows in chronological order for cardinality correction, and is
  restricted to nothing, because a query's selectivity is a property of the data and is the
  same on every machine.

Both are bounded. Each consumer reduces a view to a median or a regression coefficient, so the
newest few thousand rows decide the fit and older ones only cost memory and parse time; without
a cap the view — and the per-query fit over it — would grow for the life of the process. The
backend still holds the full history.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from batcher._internal.logging import get_logger
from batcher.metadata.hardware_scope import measured_here

__all__ = [
    "PER_KIND_MAX",
    "SIGNED_HISTORY_MAX",
    "bucket_by_kind",
    "chronological_signed",
    "trimmed",
]

_log = get_logger("metadata")

# Cap on the in-memory view of signature-carrying feedback. The consumer averages the last
# handful of observations per signature, so this is orders of magnitude more than it reads; it
# exists only to bound a long-lived session's memory.
SIGNED_HISTORY_MAX = 4096

# Cap on the retained rows *per operator family* in the by-kind view.
PER_KIND_MAX = 4096

# Bounded views are trimmed only once they exceed their cap by this factor, so a trim costs
# O(cap) once every O(cap) records rather than O(cap) on every record.
_TRIM_SLACK = 2


def trimmed(rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    """`rows` bounded to its newest `cap` entries, trimming only past the slack factor.

    Args:
        rows: The view's row list, mutated in place.
        cap: The retention target.

    Returns:
        The same list, for chaining.
    """
    if len(rows) > cap * _TRIM_SLACK:
        del rows[:-cap]
    return rows


def bucket_by_kind(scanned: Iterable[tuple[Any, bytes]]) -> dict[str, list[dict[str, Any]]]:
    """Bucket this machine's feedback rows by operator `kind`, bounded per bucket.

    Filtered by `measured_here`: everything fitted from this view is in machine units, and a
    row from another node describes hardware the next query will not run on. A malformed row
    is skipped rather than raised — a corrupt entry must not break planning.

    Args:
        scanned: `(key, value)` pairs from a backend scan of the `op_stats` table.

    Returns:
        Operator kind to its rows from this machine class, oldest first.
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    try:
        for _key, value in scanned:
            row = json.loads(value)
            if not measured_here(row):
                continue
            buckets.setdefault(row.get("kind", ""), []).append(row)
    except Exception:  # pragma: no cover - calibration must not break planning
        _log.warning("could not scan op_stats", exc_info=True)
    for bucket in buckets.values():
        trimmed(bucket, PER_KIND_MAX)
    return buckets


def chronological_signed(scanned: Iterable[tuple[Any, bytes]]) -> list[dict[str, Any]]:
    """Signature-carrying feedback rows, oldest first, bounded to the newest window.

    Deliberately **not** filtered by machine: this feeds cardinality correction, and a filter's
    selectivity or a join's fan-out is a property of the data, identical wherever it runs.
    Restricting it would fragment the statistics that take longest to collect for no gain.

    Rows without a signature are excluded, notably those a distributed worker reports for its
    sub-plan — their `op_id`s address their own space and correlate with nothing on the driver.

    Args:
        scanned: `(key, value)` pairs from a backend scan of the `op_stats` table.

    Returns:
        The newest `SIGNED_HISTORY_MAX` signature-carrying rows, oldest first.
    """
    ordered: list[tuple[int, dict[str, Any]]] = []
    try:
        for key, value in scanned:
            row = json.loads(value)
            if not row.get("signature"):
                continue
            seq = int(key[1]) if len(key) > 1 else 0
            ordered.append((seq, row))
    except Exception:  # pragma: no cover - learning must not break planning
        _log.warning("could not scan op_stats", exc_info=True)
        return []
    ordered.sort(key=lambda pair: pair[0])
    return [row for _seq, row in ordered[-SIGNED_HISTORY_MAX:]]
