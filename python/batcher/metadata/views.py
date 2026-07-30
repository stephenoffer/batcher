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
    "build_views",
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


# How many decode failures one load reports before it stops logging. A store that is
# wholesale unreadable would otherwise emit one warning per row, which buries the first
# (and only informative) one under tens of thousands of duplicates.
_DECODE_WARN_LIMIT = 3


def _rows(scanned: Iterable[tuple[Any, bytes]]) -> Iterable[tuple[Any, dict[str, Any]]]:
    """Decode a backend scan into `(key, row)` pairs, **skipping** what will not decode.

    The isolation is per row, and that is the whole point. The view builders used to wrap their
    entire loop in one `try`, so a single unparseable entry — a truncated write, a row from a
    build with a different shape, a value another process was mid-write on — did not cost that
    row, it cost *every row after it*. For the signed history that meant returning `[]`, i.e.
    "this session has measured nothing", which silently disables cardinality correction and
    cost calibration wholesale rather than degrading them by one observation.

    A non-dict value is skipped too: JSON's scalars parse without error, and a bare `null` or
    `7` would otherwise reach `row.get` as an `AttributeError` deeper in the caller.

    Args:
        scanned: `(key, value)` pairs from a backend scan.

    Yields:
        `(key, row)` for every entry that decoded to a JSON object.
    """
    failures = 0
    for key, value in scanned:
        try:
            row = json.loads(value)
        except Exception:
            failures += 1
            if failures <= _DECODE_WARN_LIMIT:
                _log.warning("skipped an undecodable op_stats row at key %r", key, exc_info=True)
            continue
        if isinstance(row, dict):
            yield key, row
    if failures > _DECODE_WARN_LIMIT:
        _log.warning("skipped %d further undecodable op_stats rows", failures - _DECODE_WARN_LIMIT)


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


def build_views(
    scanned: Iterable[tuple[Any, bytes]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Both views from **one** pass over the backend scan.

    The two views are read together — Kyber calibrates cost from the by-kind buckets and
    corrects cardinality from the signed history on the same optimize — but each used to load
    itself, so the first query of a session scanned the `op_stats` table twice and ran
    `json.loads` over every stored row twice. The parse is the expensive half (a persisted
    store holds tens of thousands of rows), and it produces the same objects both times.

    The two filters stay exactly as they were, and the difference between them is load-bearing:
    the by-kind buckets keep only rows measured on **this machine class**, because everything
    fitted from them is in machine units, while the signed history keeps rows from anywhere,
    because cardinality is a property of the data. A row can therefore land in both, one, or
    neither. Rows that land in both are *shared*, not copied — every consumer of these views
    reads them, so aliasing costs nothing and halves the memory a long history occupies.

    Args:
        scanned: `(key, value)` pairs from a backend scan of the `op_stats` table.

    Returns:
        `(by_kind, signed)` — the bucketed machine-scoped view and the chronological
        signature-carrying view, each bounded exactly as its single-view builder bounds it.
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    ordered: list[tuple[int, dict[str, Any]]] = []
    for key, row in _rows(scanned):
        if measured_here(row):
            buckets.setdefault(row.get("kind", ""), []).append(row)
        if row.get("signature"):
            try:
                seq = int(key[1]) if len(key) > 1 else 0
            except (TypeError, ValueError):
                seq = 0
            ordered.append((seq, row))
    for bucket in buckets.values():
        trimmed(bucket, PER_KIND_MAX)
    ordered.sort(key=lambda pair: pair[0])
    return buckets, [row for _seq, row in ordered[-SIGNED_HISTORY_MAX:]]
